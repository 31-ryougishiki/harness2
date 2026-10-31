"""SRV-Bench runtime: docker container, vLLM service, and ais_bench execution.

Standard library only.  ``docker`` and ``ais_bench`` are reached exclusively through
:mod:`subprocess`, so this module imports cleanly on a host with no NPU, no docker
and no ais_bench (CONTRACT.md section 7.4).

Failure policy: every entry point that returns a dict reports failure as
``ok=False`` plus a human readable ``error``, and leaves measured fields at
``None`` instead of inventing numbers.  ``container_start`` / ``container_stop``
have no error channel in their contract signature, so start raises
:class:`SrvError` and stop swallows it; :func:`collect` catches both.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

# CONTRACT.md section 1: scenario -> (input_len, output_len), defined here once.
SRV_SCENARIOS: dict[str, tuple[int, int]] = {
    "4k2k": (4096, 2048),
    "16k1k": (16384, 1024),
    "64k1k": (65536, 1024),
    "128k1k": (131072, 1024),
}

# Native context of Qwen3-8B, YaRN disabled (CONTRACT.md section 1).
MODEL_CTX_TOKENS: int = 32768

MODEL_ID = "Qwen/Qwen3-8B"
AISBENCH_BIN = "ais_bench"
AISBENCH_MODEL = "vllm_api_stream_chat"
PERF_DATASET = "synthetic_gen"
ACCURACY_DATASET = "gsm8k_gen_0_shot_cot_chat_prompt"
DEFAULT_KV_CONNECTOR = "P2pNcclConnector"
DEFAULT_PORT = 8000
DEFAULT_HEALTH_TIMEOUT_S = 600.0
MAX_HEALTH_TIMEOUT_S = 600.0
DEFAULT_HEALTH_INTERVAL_S = 2.0
DEFAULT_AISBENCH_TIMEOUT_S = 3600.0
DEFAULT_NUM_PROMPTS = "200"
KEEPALIVE_CMD = "sleep infinity"

# AISBench flag names differ between releases; both templates are overridable at
# call time through SRV_AISBENCH_PERF_TEMPLATE / SRV_AISBENCH_ACC_TEMPLATE so the
# installed version can be matched without touching this file.
PERF_CMD_TEMPLATE = (
    "{bin} --models {model} --datasets {dataset} --mode perf --num-prompts {num_prompts}"
    " --min-input-len {input_len} --max-input-len {input_len}"
    " --min-output-len {output_len} --max-output-len {output_len} --work-dir {workdir}"
)
ACC_CMD_TEMPLATE = (
    "{bin} --models {model} --datasets {dataset} --mode accuracy"
    " --num-prompts {num_prompts} --work-dir {workdir}"
)

_RC_DIR = "/tmp/srv"
_AISBENCH_ROOT = "/tmp/srv-aisbench"


class SrvError(RuntimeError):
    """Runtime failure whose message is already phrased for an ``error`` field."""


def applicable_scenarios(requested: list[str]) -> tuple[list[str], dict[str, str]]:
    """Return (measurable scenarios, {skipped scenario: reason}).

    A scenario is skipped when input_len + output_len exceeds MODEL_CTX_TOKENS.
    """
    usable: list[str] = []
    skipped: dict[str, str] = {}
    for name in requested:
        lens = SRV_SCENARIOS.get(name)
        if lens is None:
            skipped[name] = (
                f"unknown scenario {name!r}; known scenarios: {', '.join(SRV_SCENARIOS)}"
            )
            continue
        input_len, output_len = lens
        total = input_len + output_len
        if total > MODEL_CTX_TOKENS:
            skipped[name] = (
                f"exceeds model context: input_len={input_len} + output_len={output_len}"
                f" = {total} > MODEL_CTX_TOKENS={MODEL_CTX_TOKENS}"
            )
            continue
        usable.append(name)
    return usable, skipped


def container_start(run_id: str, *, image: str) -> str:
    """Idempotently create (or restart) the benchmark container; return its name.

    The container runs ``sleep infinity`` so that ``service_up`` can launch each
    vLLM instance with ``docker exec -d``.  Host networking makes the service
    ports reachable on 127.0.0.1 for the /health probe.

    Raises :class:`SrvError` on docker failure: this signature has no error
    channel, and pretending the container exists would fake a measurement.
    """
    if not image:
        raise SrvError("container_start requires a non-empty image (e.g. $SRV_IMAGE)")
    name = "srv-" + re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id).strip("-.")[:48]
    if name == "srv-":
        raise SrvError(f"run_id {run_id!r} yields an unusable container name")
    proc = _docker(["inspect", "-f", "{{.State.Running}}", name], timeout=120)
    if proc.returncode == 0:
        if proc.stdout.strip() == "true":
            return name
        _docker(["start", name], timeout=600, check=True)
        return name
    # A stale name from an earlier failed run must not block this one.
    _docker(["rm", "-f", name], timeout=300)
    _docker(
        [
            "run", "-d", "--name", name, "--network", "host",
            *_device_flags(),
            "--entrypoint", "bash", image, "-lc", KEEPALIVE_CMD,
        ],
        timeout=600,
        check=True,
    )
    return name


def container_stop(name: str) -> None:
    """Remove the container; best effort, because the signature returns None."""
    try:
        _docker(["rm", "-f", name], timeout=300)
    except SrvError:
        return


def service_up(
    mode: str,
    scenario: str,
    *,
    container: str,
    cfg: dict,
    log_dir: Path,
) -> dict:
    """Start the vLLM service and wait for /health on every instance.

    ``mode='single'`` starts one instance (``d_cmd`` stays None);
    ``mode='pd'`` starts 1P1D in the same container, P with prefix caching
    enabled and D with it disabled.

    Recognised ``cfg`` keys (all optional): model, tp/tensor_parallel_size,
    devices, port, p_port, d_port, max_model_len, max_num_seqs,
    gpu_memory_utilization, extra_args, p_extra_args, d_extra_args,
    kv_connector, kv_transfer, health_host, health_timeout_s,
    health_interval_s.  The health timeout is capped at MAX_HEALTH_TIMEOUT_S
    (600 s); reaching it returns ``ok=False`` with the log tail in ``log``.
    """
    cfg = dict(cfg or {})
    out: dict = {
        "ok": False,
        "p_cmd": "",
        "d_cmd": None,
        "prefix_cache": {"p": "100%", "d": "off"},
        "error": "",
        "log": "",
    }
    logs: dict[str, str] = {}
    try:
        if mode not in ("single", "pd"):
            raise SrvError(f"unknown service mode {mode!r}; expected 'single' or 'pd'")
        input_len, output_len = _scenario_lens(scenario)
        needed = input_len + output_len
        if needed > MODEL_CTX_TOKENS:
            raise SrvError(
                f"scenario {scenario} needs {needed} tokens > MODEL_CTX_TOKENS"
                f" {MODEL_CTX_TOKENS}; filter it through applicable_scenarios()"
            )
        model = str(cfg.get("model") or os.environ.get("SRV_MODEL") or MODEL_ID)
        tp = int(cfg.get("tp") or cfg.get("tensor_parallel_size") or 1)
        max_model_len = int(cfg.get("max_model_len") or needed)
        port = int(cfg.get("port") or DEFAULT_PORT)
        p_port = int(cfg.get("p_port") or port)
        d_port = int(cfg.get("d_port") or p_port + 1)
        shared = [str(arg) for arg in cfg.get("extra_args") or []]

        out["p_cmd"] = _service_cmd(
            cfg,
            model=model,
            port=p_port,
            max_model_len=max_model_len,
            tp=tp,
            prefix_cache=True,
            kv_role="kv_producer" if mode == "pd" else None,
            devices=_device_slice(cfg, tp, "p"),
            extra_args=shared + [str(arg) for arg in cfg.get("p_extra_args") or []],
        )
        if mode == "pd":
            out["d_cmd"] = _service_cmd(
                cfg,
                model=model,
                port=d_port,
                max_model_len=max_model_len,
                tp=tp,
                prefix_cache=False,
                kv_role="kv_consumer",
                devices=_device_slice(cfg, tp, "d"),
                extra_args=shared + [str(arg) for arg in cfg.get("d_extra_args") or []],
            )

        for tag, cmd in (("p", out["p_cmd"]), ("d", out["d_cmd"])):
            if not cmd:
                continue
            logs[tag] = f"{_RC_DIR}/{scenario}.{tag}.log"
            _launch(
                container,
                cmd,
                logs[tag],
                f"{_RC_DIR}/{scenario}.{tag}.rc",
            )

        pending = [{"tag": "p", "port": p_port, "rc": f"{_RC_DIR}/{scenario}.p.rc"}]
        if out["d_cmd"]:
            pending.append({"tag": "d", "port": d_port, "rc": f"{_RC_DIR}/{scenario}.d.rc"})
        timeout = float(cfg.get("health_timeout_s") or DEFAULT_HEALTH_TIMEOUT_S)
        _wait_healthy(
            container,
            pending,
            host=str(cfg.get("health_host") or "127.0.0.1"),
            timeout=min(timeout, MAX_HEALTH_TIMEOUT_S),
            interval=float(cfg.get("health_interval_s") or DEFAULT_HEALTH_INTERVAL_S),
        )
        out["ok"] = True
    except SrvError as exc:
        out["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - no entry point may raise
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        out["log"] = _dump_logs(container, logs, Path(log_dir))
    return out


def aisbench_perf(
    scenario: str,
    *,
    container: str,
    output_len: int | None = None,
) -> dict:
    """Run AISBench in perf mode; ``output_len`` overrides the scenario value.

    ``output_len=1`` is the pure-P (prefill only) probe.  ``raw`` is the report
    path inside the container; :func:`collect` copies it into the run directory
    and rewrites it relative to run_dir, as CONTRACT.md section 4 requires.
    """
    out: dict = {
        "ok": False,
        "throughput_req_s": None,
        "ttft_ms_p90": None,
        "tpot_ms": None,
        "cmd": "",
        "raw": "",
        "error": "",
    }
    try:
        input_len, default_output = _scenario_lens(scenario)
        out_len = default_output if output_len is None else int(output_len)
        if out_len < 1:
            raise SrvError(f"output_len must be >= 1, got {out_len}")
        workdir = _aisbench_workdir(scenario)
        out["cmd"] = _perf_command(scenario, input_len, out_len, workdir)
        _run_aisbench(container, out["cmd"], workdir, scenario=scenario)
        out["raw"], found = _pick_report(container, workdir, _extract_perf)
        out.update(found)
        missing = [key for key, value in found.items() if value is None]
        if missing or not out["raw"]:
            raise SrvError(
                f"perf report under {workdir} has no usable value for {missing}"
                f" (report: {out['raw'] or 'none'})"
            )
        out["ok"] = True
    except SrvError as exc:
        out["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - no entry point may raise
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def aisbench_accuracy(
    *,
    container: str,
    dataset: str = ACCURACY_DATASET,
) -> dict:
    """Run AISBench in accuracy mode and report the measured accuracy."""
    out: dict = {"ok": False, "accuracy": None, "cmd": "", "raw": "", "error": ""}
    try:
        name = str(dataset or ACCURACY_DATASET)
        workdir = f"{_AISBENCH_ROOT}/accuracy"
        out["cmd"] = _acc_command(name, workdir)
        _run_aisbench(container, out["cmd"], workdir, scenario="accuracy")
        out["raw"], found = _pick_report(container, workdir, _extract_accuracy_for(name))
        out["accuracy"] = found.get("accuracy")
        if out["accuracy"] is None or not out["raw"]:
            raise SrvError(
                f"accuracy report under {workdir} has no accuracy value"
                f" (report: {out['raw'] or 'none'})"
            )
        out["ok"] = True
    except SrvError as exc:
        out["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - no entry point may raise
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def collect(container: str, task_dir: Path, run_dir: Path, spec: dict) -> dict:
    """Drive a full measurement for one task spec and write metrics.json.

    Service topology and knobs come from ``spec['mode']`` and
    ``spec['service_cfg']``; accuracy runs when ``spec['accuracy']`` is set
    (a string selects the dataset); ``spec['pure_p']`` adds the output_len=1
    prefill probe.  Never raises: a failed step lands in ``ok``/``error``.

    CONTRACT 9/D8: ``deliverables`` is ``{filename: bool}`` produced by scanning
    run_dir after metrics.json itself is on disk, and ``changed_files`` for a
    ``code`` task is parsed from the agent's own patch file - neither is taken
    from anything the agent wrote into metrics.json.  ``task_dir`` stays part of
    the signature but is not a deliverable root: agent artifacts land in run_dir.
    """
    run_dir = Path(run_dir)
    spec = dict(spec or {})
    metrics: dict = {
        "task_id": str(spec.get("id") or ""),
        "ok": False,
        "error": "",
        "image": str(spec.get("image") or os.environ.get("SRV_IMAGE") or ""),
        "model": str(spec.get("model") or MODEL_ID),
        "service": {
            "ok": False,
            "mode": str(spec.get("mode") or "single"),
            "p_cmd": "",
            "d_cmd": None,
            "prefix_cache": {"p": "100%", "d": "off"},
            "error": "",
            "log": "",
        },
        "scenarios": {},
    }
    try:
        _collect_into(metrics, spec, container=container, run_dir=run_dir)
    except SrvError as exc:
        metrics["ok"] = False
        metrics["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - collect may never raise
        metrics["ok"] = False
        metrics["error"] = f"{type(exc).__name__}: {exc}"
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        if str(spec.get("category") or "") == "code":
            changed = _changed_files_from_patch(spec, run_dir)
            if changed:
                metrics["changed_files"] = changed
        _write_metrics(run_dir, metrics)
        # D8: scan the run directory once metrics.json exists, then persist the result.
        metrics["deliverables"] = _scan_deliverables(spec, run_dir)
        _write_metrics(run_dir, metrics)
    except OSError as exc:
        metrics["ok"] = False
        metrics["error"] = (metrics["error"] + "; " if metrics["error"] else "") + (
            f"cannot write metrics.json: {exc}"
        )
    return metrics


def _collect_into(
    metrics: dict,
    spec: dict,
    *,
    container: str,
    run_dir: Path,
) -> None:
    requested = [str(name) for name in spec.get("scenarios") or ["4k2k", "16k1k"]]
    usable, skipped = applicable_scenarios(requested)
    for name in requested:
        if name in skipped:
            metrics["scenarios"][name] = {"skipped": True, "reason": skipped[name]}
        else:
            metrics["scenarios"][name] = {"skipped": False, "reason": ""}
    if not usable:
        metrics["ok"] = True
        return

    mode = str(spec.get("mode") or "single")
    anchor = max(usable, key=lambda name: sum(SRV_SCENARIOS[name]))
    cfg = dict(spec.get("service_cfg") or {})
    service = service_up(mode, anchor, container=container, cfg=cfg, log_dir=run_dir / "logs")
    metrics["service"] = {
        "ok": bool(service["ok"]),
        "mode": mode,
        "p_cmd": service["p_cmd"],
        "d_cmd": service["d_cmd"],
        "prefix_cache": service["prefix_cache"],
        "error": service["error"],
        "log": service["log"],
    }
    if not service["ok"]:
        metrics["error"] = f"service did not start: {service['error']}"
        for name in usable:
            metrics["scenarios"][name].update(
                {"skipped": True, "reason": f"service did not start: {service['error']}"}
            )
        return

    accuracy = None
    if _needs_accuracy(spec):
        asked = spec.get("accuracy")
        if isinstance(asked, str):
            dataset = asked
        else:
            dataset = str(spec.get("dataset") or ACCURACY_DATASET)
        accuracy = aisbench_accuracy(container=container, dataset=dataset)
        metrics["accuracy"] = {
            "ok": bool(accuracy["ok"]),
            "accuracy": accuracy["accuracy"],
            "cmd": accuracy["cmd"],
            "raw": _materialize(container, accuracy["raw"], run_dir, "aisbench_accuracy.json"),
            "error": accuracy["error"],
        }
        if not accuracy["ok"]:
            metrics["error"] = _join(metrics["error"], f"accuracy run failed: {accuracy['error']}")

    for name in usable:
        block = metrics["scenarios"][name]
        perf = aisbench_perf(name, container=container)
        block["throughput_req_s"] = perf["throughput_req_s"]
        block["ttft_ms_p90"] = perf["ttft_ms_p90"]
        block["tpot_ms"] = perf["tpot_ms"]
        block["accuracy"] = accuracy["accuracy"] if accuracy and accuracy["ok"] else None
        block["aisbench_cmd"] = perf["cmd"]
        block["raw"] = _materialize(container, perf["raw"], run_dir, f"aisbench_{name}.json")
        block["reason"] = perf["error"]
        if not perf["ok"]:
            metrics["error"] = _join(metrics["error"], f"{name}: {perf['error']}")
        if spec.get("pure_p"):
            probe = aisbench_perf(name, container=container, output_len=1)
            block["pure_p"] = {
                "ok": bool(probe["ok"]),
                "output_len": 1,
                "ttft_ms_p90": probe["ttft_ms_p90"],
                "aisbench_cmd": probe["cmd"],
                "raw": _materialize(container, probe["raw"], run_dir, f"pure_p_{name}.json"),
                "error": probe["error"],
            }

    measured = [block for block in metrics["scenarios"].values() if not block.get("skipped")]
    complete = all(
        _is_number(block.get(key))
        for block in measured
        for key in ("throughput_req_s", "ttft_ms_p90", "tpot_ms")
    )
    metrics["ok"] = bool(measured) and complete and not metrics["error"]


# --------------------------------------------------------------------------- #
# docker plumbing
# --------------------------------------------------------------------------- #


def _docker(args: list[str], *, timeout: float = 120.0, check: bool = False):
    cmd = ["docker", *args]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise SrvError(f"docker executable not found: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise SrvError(f"{' '.join(cmd)} timed out after {timeout}s") from exc
    except OSError as exc:
        raise SrvError(f"{' '.join(cmd)} failed: {exc}") from exc
    if check and proc.returncode != 0:
        raise SrvError(f"{' '.join(cmd[:4])} exited {proc.returncode}: {_tail(proc.stderr)}")
    return proc


def _device_flags() -> list[str]:
    paths: list[str] = []
    for path in [*sorted(glob.glob("/dev/davinci*")), "/dev/davinci_manager"]:
        if path not in paths:
            paths.append(path)
    flags: list[str] = []
    for path in paths:
        flags += ["--device", path]
    return flags


def _launch(container: str, cmd: str, log_path: str, rc_path: str) -> None:
    inner = (
        f"rm -f {shlex.quote(rc_path)}"
        f" && mkdir -p {shlex.quote(_posix_dir(log_path))}"
        f" && ( {cmd} ) > {shlex.quote(log_path)} 2>&1"
        f"; echo $? > {shlex.quote(rc_path)}"
    )
    _docker(["exec", "-d", container, "bash", "-lc", inner], timeout=180, check=True)


def _container_int(container: str, path: str) -> int | None:
    try:
        proc = _docker(["exec", container, "cat", path], timeout=60)
    except SrvError:
        return None
    if proc.returncode != 0:
        return None
    text = proc.stdout.strip()
    return int(text) if re.fullmatch(r"-?\d+", text) else None


def _wait_healthy(
    container: str,
    pending: list[dict],
    *,
    host: str,
    timeout: float,
    interval: float,
) -> None:
    deadline = time.monotonic() + max(0.0, float(timeout))
    waiting = list(pending)
    while waiting:
        still: list[dict] = []
        for item in waiting:
            if _http_ok(f"http://{host}:{item['port']}/health"):
                continue
            code = _container_int(container, item["rc"])
            if code is not None:
                raise SrvError(
                    f"instance {item['tag']} exited with code {code} before /health"
                    f" answered on port {item['port']}"
                )
            still.append(item)
        waiting = still
        if not waiting:
            return
        now = time.monotonic()
        if now >= deadline:
            ports = ", ".join(str(item["port"]) for item in waiting)
            raise SrvError(f"/health not ready on port(s) {ports} within {int(timeout)}s")
        time.sleep(max(0.0, min(float(interval), deadline - now)))


def _http_ok(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= int(response.status) < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _dump_logs(container: str, logs: dict[str, str], log_dir: Path) -> str:
    if not logs:
        return ""
    chunks: list[str] = []
    for tag, path in logs.items():
        try:
            proc = _docker(["exec", container, "cat", path], timeout=120)
            body = (
                proc.stdout
                if proc.returncode == 0
                else f"<docker exec cat failed>\n{proc.stderr}"
            )
        except SrvError as exc:
            body = f"<unavailable: {exc}>"
        chunks.append(f"===== {tag}: {path} =====\n{body}")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        target = log_dir / "service.log"
        target.write_text("\n".join(chunks), encoding="utf-8")
        return str(target)
    except OSError:
        return ""


# --------------------------------------------------------------------------- #
# vLLM command construction
# --------------------------------------------------------------------------- #


def _service_cmd(
    cfg: dict,
    *,
    model: str,
    port: int,
    max_model_len: int,
    tp: int,
    prefix_cache: bool,
    kv_role: str | None,
    devices: str,
    extra_args: list[str],
) -> str:
    args = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--served-model-name", model,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--tensor-parallel-size", str(tp),
        "--max-model-len", str(max_model_len),
        "--enable-prefix-caching" if prefix_cache else "--no-enable-prefix-caching",
    ]
    if kv_role:
        payload = _kv_config(cfg, kv_role)
        args += ["--kv-transfer-config", json.dumps(payload, separators=(",", ":"))]
    if cfg.get("max_num_seqs"):
        args += ["--max-num-seqs", str(cfg["max_num_seqs"])]
    if cfg.get("gpu_memory_utilization"):
        args += ["--gpu-memory-utilization", str(cfg["gpu_memory_utilization"])]
    args += extra_args
    body = " ".join(shlex.quote(arg) for arg in args)
    return f"ASCEND_RT_VISIBLE_DEVICES={devices} {body}" if devices else body


def _kv_config(cfg: dict, role: str) -> dict:
    base = dict(cfg.get("kv_transfer") or {})
    connector = base.pop("kv_connector", None) or cfg.get("kv_connector") or DEFAULT_KV_CONNECTOR
    payload = {"kv_connector": str(connector), "kv_role": role}
    payload.update(base)
    return payload


def _device_slice(cfg: dict, tp: int, role: str) -> str:
    raw = cfg.get("devices")
    if raw is None:
        return ""
    if isinstance(raw, str):
        seq = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        seq = [str(part).strip() for part in raw]
    start = 0 if role == "p" else tp
    return ",".join(seq[start:start + tp])


# --------------------------------------------------------------------------- #
# ais_bench plumbing
# --------------------------------------------------------------------------- #


def _perf_command(scenario: str, input_len: int, output_len: int, workdir: str) -> str:
    fields = _template_fields(scenario, workdir, mode="perf")
    fields.update({"input_len": str(input_len), "output_len": str(output_len)})
    template = os.environ.get("SRV_AISBENCH_PERF_TEMPLATE", PERF_CMD_TEMPLATE)
    try:
        return template.format(**fields)
    except (KeyError, IndexError) as exc:
        raise SrvError(f"bad perf command template {template!r}: {exc}") from exc


def _acc_command(dataset: str, workdir: str) -> str:
    fields = _template_fields("accuracy", workdir, mode="accuracy")
    fields["dataset"] = dataset
    template = os.environ.get("SRV_AISBENCH_ACC_TEMPLATE", ACC_CMD_TEMPLATE)
    try:
        return template.format(**fields)
    except (KeyError, IndexError) as exc:
        raise SrvError(f"bad accuracy command template {template!r}: {exc}") from exc


def _template_fields(scenario: str, workdir: str, *, mode: str) -> dict:
    dataset = PERF_DATASET if mode == "perf" else ACCURACY_DATASET
    dataset = os.environ.get(
        "SRV_AISBENCH_PERF_DATASET" if mode == "perf" else "SRV_AISBENCH_ACC_DATASET",
        dataset,
    )
    return {
        "bin": os.environ.get("SRV_AISBENCH_BIN", AISBENCH_BIN),
        "model": os.environ.get("SRV_AISBENCH_MODEL", AISBENCH_MODEL),
        "dataset": dataset,
        "mode": mode,
        "num_prompts": os.environ.get("SRV_AISBENCH_NUM_PROMPTS", DEFAULT_NUM_PROMPTS),
        "input_len": "0",
        "output_len": "0",
        "scenario": scenario,
        "workdir": workdir,
    }


def _run_aisbench(container: str, cmd: str, workdir: str, *, scenario: str) -> None:
    inner = f"mkdir -p {shlex.quote(workdir)} && {cmd}"
    timeout = float(os.environ.get("SRV_AISBENCH_TIMEOUT_S") or DEFAULT_AISBENCH_TIMEOUT_S)
    proc = _docker(["exec", container, "bash", "-lc", inner], timeout=timeout)
    if proc.returncode != 0:
        raise SrvError(
            f"ais_bench ({scenario}) exited {proc.returncode}: {_tail(proc.stdout + proc.stderr)}"
        )


def _pick_report(container: str, workdir: str, extract) -> tuple[str, dict]:
    best_path, best_found, best_hits = "", {}, -1
    for path in _json_files(container, workdir):
        try:
            payload = _container_json(container, path)
        except SrvError:
            continue
        found = extract(_flatten(payload))
        hits = sum(1 for value in found.values() if value is not None)
        if hits > best_hits:
            best_path, best_found, best_hits = path, found, hits
        if hits == len(found):
            return path, found
    return best_path, best_found


def _json_files(container: str, workdir: str) -> list[str]:
    inner = (
        f"find {shlex.quote(workdir)} -type f -name '*.json'"
        " -printf '%T@ %p\n' | sort -rn | head -40"
    )
    try:
        proc = _docker(["exec", container, "bash", "-lc", inner], timeout=180)
    except SrvError:
        return []
    paths: list[str] = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split(" ", 1)
        if len(parts) == 2 and parts[1].strip():
            paths.append(parts[1].strip())
    return paths


def _container_json(container: str, path: str) -> dict:
    proc = _docker(["exec", container, "cat", path], timeout=180, check=True)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SrvError(f"{path} is not JSON: {exc}") from exc


def _extract_perf(flat: list[tuple[str, str, object]]) -> dict:
    return {
        "throughput_req_s": _pick(flat, ("request_throughput", "req_s", "requests_per_sec",
                                        "request_per_second", "throughput")),
        "ttft_ms_p90": _pick_ttft_p90(flat),
        "tpot_ms": _pick(flat, ("average_tpot_ms", "mean_tpot_ms", "tpot_ms", "tpot")),
    }


def _extract_accuracy_for(dataset: str):
    stem = re.sub(r"[^a-z0-9]+", "_", str(dataset).split(".")[0].lower()).strip("_")

    def extract(flat: list[tuple[str, str, object]]) -> dict:
        patterns = []
        if stem:
            patterns += [f"{stem}_accuracy", f"{stem}_acc", stem]
        patterns += ["accuracy", "acc", "score"]
        return {"accuracy": _pick(flat, tuple(patterns))}

    return extract


def _flatten(payload: object, prefix: str = "") -> list[tuple[str, str, object]]:
    flat: list[tuple[str, str, object]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            flat += _flatten(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            flat += _flatten(value, f"{prefix}[{index}]")
    else:
        flat.append((prefix, _norm_key(prefix), payload))
    return flat


def _norm_key(path: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", path.lower()).strip("_")


def _pick(flat: list[tuple[str, str, object]], patterns: tuple[str, ...]) -> float | None:
    for pattern in patterns:
        for _, norm, value in flat:
            if pattern in norm:
                number = _as_float(value)
                if number is not None:
                    return number
    return None


def _pick_ttft_p90(flat: list[tuple[str, str, object]]) -> float | None:
    for _, norm, value in flat:
        if "ttft" in norm and ("p90" in norm or "90_percentile" in norm or "pctl90" in norm):
            number = _as_float(value)
            if number is not None:
                return number
    return None


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", value)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


# --------------------------------------------------------------------------- #
# small shared helpers
# --------------------------------------------------------------------------- #


def _scenario_lens(scenario: str) -> tuple[int, int]:
    try:
        return SRV_SCENARIOS[scenario]
    except KeyError:
        raise SrvError(
            f"unknown scenario {scenario!r}; known scenarios: {', '.join(SRV_SCENARIOS)}"
        ) from None


def _aisbench_workdir(scenario: str) -> str:
    return f"{_AISBENCH_ROOT}/{scenario}"


def _posix_dir(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else "."


def _tail(text: str, limit: int = 800) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "..." + text[-limit:]


def _join(existing: str, extra: str) -> str:
    return extra if not existing else f"{existing}; {extra}"


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _materialize(container: str, remote: str, run_dir: Path, name: str) -> str:
    if not remote:
        return ""
    try:
        proc = _docker(["exec", container, "cat", remote], timeout=180)
        if proc.returncode != 0:
            return ""
        target = Path(run_dir) / "artifacts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(proc.stdout, encoding="utf-8")
        return target.relative_to(Path(run_dir)).as_posix()
    except (SrvError, OSError, ValueError):
        return ""


def _needs_accuracy(spec: dict) -> bool:
    """Accuracy runs when the spec asks for it or scores it (thresholds.accuracy)."""
    if spec.get("accuracy"):
        return True
    thresholds = spec.get("thresholds")
    return isinstance(thresholds, dict) and "accuracy" in thresholds


DELIVERABLE_DIRS = ("", "deliverables", "workspace/deliverables")
PATCH_NAMES = ("patch.diff", "patch.patch", "changes.diff", "changes.patch", "diff.patch")


def _write_metrics(run_dir: Path, metrics: dict) -> None:
    target = Path(run_dir) / "metrics.json"
    target.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _scan_deliverables(spec: dict, run_dir: Path) -> dict[str, bool]:
    """CONTRACT 9/D8: what is on disk in run_dir, not what the agent claims."""
    roots = [Path(run_dir)]
    if spec.get("work_dir"):
        roots.append(Path(str(spec["work_dir"])))
    found: dict[str, bool] = {}
    for name in spec.get("deliverables") or []:
        item = str(name)
        found[item] = any(
            (root / sub / item).is_file() for root in roots for sub in DELIVERABLE_DIRS
        )
    return found


def _changed_files_from_patch(spec: dict, run_dir: Path) -> list[str]:
    """CONTRACT 9/D8: changed paths parsed from the patch/diff the agent produced."""
    candidates: list[Path] = []
    if spec.get("patch_path"):
        candidates.append(Path(str(spec["patch_path"])))
    for base in (run_dir, run_dir / "deliverables", run_dir / "workspace"):
        for name in PATCH_NAMES:
            candidates.append(base / name)
    for path in candidates:
        try:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        files = _parse_patch_paths(text)
        if files:
            return files
    return []


def _parse_patch_paths(text: str) -> list[str]:
    paths: list[str] = []
    for line in text.splitlines():
        for prefix in ("+++ ", "--- "):
            if not line.startswith(prefix):
                continue
            raw = line[len(prefix):].split("\t")[0].strip()
            if not raw or raw == "/dev/null":
                continue
            for strip in ("a/", "b/"):
                if raw.startswith(strip):
                    raw = raw[len(strip):]
                    break
            raw = raw.removeprefix("./").strip("/")
            if raw and raw not in paths:
                paths.append(raw)
    return paths
