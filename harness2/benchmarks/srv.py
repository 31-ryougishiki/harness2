# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SRV adapter: inference-service work on an Ascend vLLM stack, on either substrate.

The machine work lives behind two modules that are imported LAZILY and never at import
time: `srv_runtime` (container, service bring-up, aisbench) and `srv_grade` (the marked
verdict). This module therefore imports, discovers tasks, builds its spec and renders its
prompts on a host with no docker, no NPU and no aisbench installed.

`task_prompt` reads instruction.md and nothing else: task.json carries `baseline` and
`thresholds`, which are judge-side material and never leave `_task_json`.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
from dataclasses import replace
from pathlib import Path

from .. import config
from ..core import runmark as RM
from ..core import tree as HT
from ..core.spec import BenchPrompts, BenchSpec, resolve_channel_model
from ..substrate.base import Install, RunRequest
from ..substrate.codex import CODEX_SURFACE, CodexSubstrate
from ..substrate.opencode import OPENCODE_SURFACE, OpencodeSubstrate
from .tasks import discover, select

SRV_DIR = config.DEPENDENCIES / "srv-bench"
KIT_DIR = SRV_DIR / "base_harness"
KIT_NAMES = ("run_service.sh", "aisbench_example.sh", "README.md")
CATEGORIES = ("config", "measure", "code")
MODES = ("single", "pd", "patch")
MODEL = "Qwen/Qwen3-8B"
MODEL_CTX_TOKENS = 32768
IMAGE_ENV = "SRV_IMAGE"
DELIV_CAP = 40_000
DELIV_DIRS = ("deliverables", "workspace/deliverables")
TEXT_SUFFIXES = frozenset(
    {".md", ".txt", ".json", ".sh", ".yaml", ".yml", ".py", ".log", ".csv", ".toml"}
)

_BASE_LOCK = threading.Lock()
_BASE_DONE: set[Path] = set()


SRV_BASE = BenchPrompts(
    work=(
        "Inference service work for one fixed model on an Ascend host: launching vLLM-Ascend "
        "as a single instance or as one prefill plus one decode instance inside the same "
        "container, driving a load client at the sequence lengths the assignment names, and "
        "changing launch configuration or serving code so that latency and throughput improve "
        "while answer accuracy holds. The result is a service that comes up, a record of what "
        "was actually observed on each attempt, and the exact artifacts the assignment names."
    ),
    domain_word="serving concern",
    deliverable_word="service artifact",
    inv_levers=(
        "- read the launch kit and the assignment's constraints before changing anything; write "
        "down the image, the model, the context window and the sequence lengths in force\n"
        "- keep one attempt log per change: the exact command, whether the service reported "
        "healthy, and the client call you actually ran\n"
        "- change one launch parameter at a time and repeat the same client configuration "
        "before comparing two attempts\n"
        "- when the service does not come up, read its log tail before changing anything else\n"
        "- never report a figure you did not observe, and say plainly which scenario you could "
        "not serve and why"
    ),
    dom_levers=(
        "- which parallelism, batch, memory-fraction, scheduling and cache settings the device "
        "and a 32768-token window actually allow\n"
        "- how the client's dataset, concurrency, request rate and sequence lengths map onto the "
        "named scenarios, and which of them do not fit the window at all\n"
        "- where the serving path spends its time, and which code change is small enough to ship "
        "without breaking correctness\n"
        "- whether a prefill/decode split earns its transfer cost for this workload, and which "
        "cache settings each side needs"
    ),
    yardstick=(
        "Did the run bring the service up on the required image and model, drive every scenario "
        "it could serve at that scenario's own lengths, record commands it actually ran, keep "
        "answer accuracy intact, and produce every artifact the assignment named?"
    ),
    selection_checks=(
        "Prefer the run whose service actually came up and served the required scenarios; whose "
        "recorded commands and artifacts agree with each other; whose figures appear in its own "
        "artifacts rather than in prose; that keeps accuracy inside the stated tolerance; that "
        "reports a scenario it could not serve as skipped with a concrete reason; and that stays "
        "inside the assignment's constraints."
    ),
)


SRV_CATEGORIES: dict[str, str] = {
    "config": (
        "SERVING CONCERN - LAUNCH CONFIGURATION. Establish what the device can hold: the weight "
        "and cache footprint of the model at 32768 tokens, the parallelism the card count "
        "supports, and the batch and memory-fraction ceilings. Change launch settings one at a "
        "time, and re-drive the same client configuration before believing any difference. A "
        "setting that makes the service refuse to start is a finding, not a result."
    ),
    "measure": (
        "SERVING CONCERN - DRIVING THE CLIENT. The work is the measurement path itself: a "
        "launcher that brings the service up, a client call that names the right dataset, mode, "
        "concurrency and sequence lengths, and a report that separates what ran from what could "
        "not run. A scenario that does not fit the 32768-token window is reported as skipped with "
        "its reason; a client run that failed is reported as failed, never as a figure."
    ),
    "code": (
        "SERVING CONCERN - SERVING CODE. Locate the hot path before editing: scheduling, "
        "batching, cache handling or the client loop. Keep the change reviewable, keep behaviour "
        "for the untested scenarios unchanged, and prove the result by re-running the same client "
        "configuration that produced the starting figures."
    ),
}


def srv_prompts(category: str = "") -> BenchPrompts:
    """The SRV profile plus the category overlay that describes this kind of work."""
    body = SRV_CATEGORIES.get(category, "")
    if not body:
        return SRV_BASE
    guidance = "\n\n".join(p for p in (SRV_BASE.domain_guidance, body) if p.strip())
    return replace(SRV_BASE, domain_guidance=guidance)


def srv_spec(substrate_name: str) -> BenchSpec:
    """The SRV BenchSpec for one substrate cell. A value, not a code path."""
    if substrate_name not in ("oc", "cx"):
        raise ValueError(f"substrate must be 'oc' or 'cx', got {substrate_name!r}")
    channel = resolve_channel_model(None, None, effort="high")
    return BenchSpec(
        name="srv",
        runs_root=config.RUNS / "srv",
        install_mode="workspace_config" if substrate_name == "oc" else "codex_root",
        prompts=SRV_BASE,
        channels={"proposer": channel, "selector": channel},
        domain_word="serving concern",
        leak_globs=("task.json", "verify.py", "reference/*", "baseline/*"),
        allow_subagents=False,
        proposer_deliverables="contents",
        obs_trunc=50_000,
    )


def srv_image() -> str:
    """The container image, from SRV_IMAGE only: no tag is ever written down here."""
    image = config.env(IMAGE_ENV, "")
    if not image:
        raise config.ConfigError(
            f"{IMAGE_ENV} is required for srv (the vllm-ascend image this host runs); set it in "
            f".env or the shell, e.g. {IMAGE_ENV}=vllm-ascend:0.23.0-a2"
        )
    return str(image)


def strip_front_matter(text: str) -> str:
    """The instruction body without any leading YAML front-matter block."""
    body = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = body.split("\n")
    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                return "\n".join(lines[index + 1 :]).strip()
    return body.strip()


def _runtime():
    """`srv_runtime`, or a RuntimeError naming the missing half of the installation."""
    try:
        from . import srv_runtime
    except ImportError as exc:
        raise RuntimeError(
            f"srv_runtime is not available ({exc}); it owns the container, the service "
            f"bring-up and the client runs, and ships with the Ascend host image. Discovery, "
            f"the dry-run plan and task authoring need neither it nor docker."
        ) from exc
    return srv_runtime


def _grader():
    """`srv_grade`, or None. A missing marker means unscored, never zero."""
    try:
        from . import srv_grade
    except ImportError:
        return None
    return srv_grade


def _json_or_none(p: Path) -> dict | None:
    try:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _run_detached(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout: int,
) -> None:
    """Run the substrate in its own process group and kill the group on a wall timeout.

    A non-zero exit is a failed rollout, not a silent one: the caller turns it into a dead run
    that is never cached as a result.
    """
    stdout_path = Path(stdout_path)
    stderr_path = Path(stderr_path)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        stdout_path.open("w", encoding="utf-8") as out,
        stderr_path.open("w", encoding="utf-8") as err,
    ):
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (AttributeError, OSError, ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait()
            raise
    if proc.returncode != 0:
        raise RuntimeError(
            f"substrate runner exited with status {proc.returncode} ({stderr_path.name})"
        )


def _base_body(heading: str) -> str:
    """The reference always-on body: what the work is, what is fixed, where the kit is."""
    return f"""# YOU RUN AN INFERENCE SERVICE

{heading}

## WHAT IS FIXED

- Model `{MODEL}`, native context {MODEL_CTX_TOKENS} tokens. YaRN is off, so a scenario whose
  input plus output length exceeds {MODEL_CTX_TOKENS} cannot be served: report it as skipped,
  with the reason, instead of running it.
- The container image comes from the `{IMAGE_ENV}` environment variable. Never write a tag down.
- The hardware is Ascend A2/A3. There is no CUDA path here.
- Accuracy is part of the work: the assignment names the dataset and the tolerance it allows.

## WHERE THE WORK HAPPENS

- `launch_kit/run_service.sh` - reference launcher: one instance, or one prefill plus one
  decode instance inside the same container. Edit it, or replace it, as the assignment requires.
- `launch_kit/aisbench_example.sh` - reference client call, with the flags that name the
  dataset, the mode and the input/output lengths.
- `launch_kit/README.md` - what the two files do and which environment variables they read.

## DELIVERABLES

Write every file the assignment names to `deliverables/<filename>`. Files left anywhere else
are not part of the delivered work.

## HOW TO WORK

- Read the assignment and the launch kit before changing anything.
- Keep one log per attempt: the exact command, whether the service reported healthy, and the
  client call you actually ran.
- Change one launch parameter at a time, and repeat the same client configuration before
  comparing two attempts.
- When the service does not come up, read its log tail before changing anything else.
- Never write a figure you did not observe. A service that never became healthy is a failure
  and must be reported as one.
"""


class SrvAdapter:
    """The BenchmarkAdapter surface for SRV, on either substrate.

    The adapter owns the seam - task reading, harness staging, run-directory bookkeeping,
    grading - and delegates the machine work to `srv_runtime`.
    """

    def __init__(
        self,
        *,
        substrate: str = "oc",
        model: str | None = None,
        timeout: int | None = None,
        solver_effort: str = "high",
        split_file: Path | None = None,
    ):
        self.split_file = split_file
        self.spec = srv_spec(substrate)
        if substrate == "oc":
            self.substrate = OpencodeSubstrate(launcher=self._launch)
            self.model = model or config.SOLVER_MODEL_OC
        else:
            self.substrate = CodexSubstrate(launcher=self._launch)
            self.model = model or config.SOLVER_MODEL_CX
        self._results_root = self.spec.runs_root / "results" / substrate
        self.timeout = int(timeout or 3600)
        self.solver_effort = solver_effort
        self._deliverables: dict[str, dict[str, str]] = {}
        self._task_of: dict[str, str] = {}

    def tasks(self, split: str = "all") -> list[str]:
        return select(discover("srv"), split, self.split_file, benchmark="srv")

    def domain_of(self, task_id: str) -> str:
        return task_id.split("/", 1)[0]

    def reference_domain_order(self) -> list[str]:
        return sorted(set(discover("srv").values()))

    def prompts_for(self, task_id: str) -> BenchPrompts:
        return srv_prompts(self.domain_of(task_id))

    def _task_dir(self, task_id: str) -> Path:
        return SRV_DIR / "tasks" / task_id

    def _task_json(self, task_id: str) -> dict:
        """task.json WITH the judge-side keys. Only execute/score may call this."""
        return json.loads((self._task_dir(task_id) / "task.json").read_text(encoding="utf-8"))

    def task_prompt(self, task_id: str, *, max_chars: int = 12_000) -> str:
        """instruction.md only: no front-matter, no threshold, no baseline."""
        body = strip_front_matter(
            (self._task_dir(task_id) / "instruction.md").read_text(encoding="utf-8")
        )
        return body[:max_chars]

    def task_files(self, task_id: str) -> Path | None:
        """None: a task directory carries reference/ and verify.py, so none of it is shipped."""
        return None

    def _run_dir(self, run_id: str) -> Path:
        return self._results_root / run_id

    def _workspace(self, run_dir: Path) -> Path:
        return Path(run_dir) / "workspace"

    def execute(
        self, task_id: str, harness_dir: Path | None, run_id: str, *, timeout: int | None = None
    ) -> Path:
        out = self._run_dir(run_id)
        self._task_of[out.name] = task_id
        if RM.is_complete(out, harness_dir=harness_dir) and (out / "metrics.json").is_file():
            return out
        if out.is_dir() and any(out.iterdir()):
            print(
                f"  [srv-{self.substrate.name}] re-rolling {run_id}: "
                f"{RM.why_not(out, harness_dir=harness_dir)}",
                flush=True,
            )
            self._safe_rmtree(out)
        self._deliverables.pop(str(out.resolve()), None)
        out.mkdir(parents=True, exist_ok=True)
        install = self._install(harness_dir)
        try:
            self._rollout(task_id, install, out, int(timeout or self.timeout))
        finally:
            if install.extras.get("harness") == "staged":
                shutil.rmtree(install.staging_dir, ignore_errors=True)
        self._measure(task_id, out)
        if not (out / "metrics.json").is_file():
            raise RuntimeError(
                f"srv run produced no metrics.json: {out} - srv_runtime.collect writes that "
                f"file, and a run without it is unscored (never cached as a zero)"
            )
        RM.mark_complete(out, run_id=run_id, harness_dir=harness_dir)
        return out

    def _rollout(self, task_id: str, install: Install, run_dir: Path, timeout: int) -> None:
        """The agent's own pass over the task, through the substrate's launcher seam."""
        self.substrate.execute(task_id, install, [Path(run_dir).name], timeout)

    def _measure(self, task_id: str, run_dir: Path) -> None:
        """Bring a container up, drive the task's measurement, and stop it again."""
        rt = _runtime()
        image = srv_image()
        container = rt.container_start(Path(run_dir).name, image=image)
        try:
            rt.collect(container, self._task_dir(task_id), Path(run_dir), self._task_json(task_id))
        finally:
            rt.container_stop(container)

    def _launch(self, req: RunRequest, install: Install) -> list[Path]:
        """Start the substrate for this rollout: build the command here, run it detached, and
        return the run directories. The same shape as the LAB launcher - the adapter owns the
        runner choice, the environment and the read-back locations, and `srv_runtime` keeps to
        the container, the service, the client and `collect`.
        """
        if not self.model:
            raise config.ConfigError(
                "HARNESS2_SOLVER_MODEL is unset: an srv rollout needs the provider-prefixed "
                "model id the substrate runs"
            )
        prompt = self._compose_prompt(req.task_ref, install)
        stream_name = (
            "opencode_session.jsonl" if self.substrate.name == "oc" else "codex_session.jsonl"
        )
        dirs: list[Path] = []
        for run_id in req.run_ids:
            run_id = str(run_id)
            run_dir = self._run_dir(run_id)
            workspace = self._workspace(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            self._seed_workspace(install.staging_dir, workspace)
            if self.substrate.name == "oc":
                cmd = self.substrate.run_argv(prompt, workspace, self.model, title=f"srv {run_id}")
                env = self.substrate.run_env(os.environ, workspace)
            else:
                cmd = self.substrate.exec_argv(
                    prompt, workspace, self.model, reasoning_effort=self.solver_effort
                )
                env = self.substrate.exec_env(os.environ, codex_home=config.CODEX_HOME_DIR)
            record = {
                "task_id": req.task_ref,
                "run_id": run_id,
                "substrate": self.substrate.name,
                "model": self.model,
                "effort": self.solver_effort,
                "install_dir": str(install.staging_dir),
                "workspace": str(workspace),
                "stream": str(run_dir / stream_name),
                "timeout_s": int(req.timeout),
                "cmd": cmd,
            }
            (run_dir / "runner.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
            log = run_dir / "runner.log"
            try:
                _run_detached(
                    cmd,
                    cwd=config.ROOT,
                    env=env,
                    stdout_path=run_dir / stream_name,
                    stderr_path=log,
                    timeout=int(req.timeout),
                )
            except subprocess.TimeoutExpired:
                print(
                    f"  [srv-{self.substrate.name}] runner wall-timeout for "
                    f"{req.task_ref} @ {run_id}",
                    flush=True,
                )
            except RuntimeError as exc:
                raise RuntimeError(f"{exc}; see {log}") from None
            dirs.append(run_dir)
        return dirs

    def _compose_prompt(self, task_id: str, install: Install) -> str:
        """The harness's always-on text, then the assignment. The staged tree is the harness
        under test, so its text is the one that travels, not the base tree's."""
        surface = self.substrate.surface()
        body = HT.always_on_text(surface, HT.read_tree(surface, install.staging_dir)).strip()
        assignment = "# ASSIGNMENT\n\n" + self.task_prompt(task_id).strip()
        return "\n\n---\n\n".join(part for part in (body, assignment) if part.strip())

    def _seed_workspace(self, staging: Path, workspace: Path) -> None:
        """The staged harness on disk, minus the always-on bodies: those travel in the prompt,
        and a second copy on disk would be injected twice."""
        skip = set(self.substrate.surface().always_on)
        staging = Path(staging)
        if not staging.is_dir():
            return
        for src in sorted(staging.rglob("*")):
            if not src.is_file():
                continue
            rel = src.relative_to(staging).as_posix()
            if rel in skip or rel.startswith("_composed/"):
                continue
            dest = workspace / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    def _install(self, harness_dir: Path | None) -> Install:
        """BASE (None) is a value: the reference tree plus the launch kit it ships with."""
        if harness_dir is None:
            return Install(
                staging_dir=self.base_harness(), fingerprint="base", extras={"harness": "base"}
            )
        inst = self.substrate.materialise(Path(harness_dir), self.spec, {})
        self._copy_kit(inst.staging_dir)
        return Install(
            staging_dir=inst.staging_dir,
            fingerprint=inst.fingerprint,
            extras={**inst.extras, "harness": "staged"},
        )

    def _copy_kit(self, dest: Path) -> None:
        """The launch kit rides along with every staging: it is not on the c1..c8 surface."""
        if KIT_DIR.is_dir():
            shutil.copytree(KIT_DIR, Path(dest) / "launch_kit", dirs_exist_ok=True)

    def _safe_rmtree(self, d: Path) -> None:
        d = Path(d).resolve()
        root = self._results_root.resolve()
        if root != d and root not in d.parents:
            raise RuntimeError(f"refusing to remove {d}: outside the SRV results root {root}")
        shutil.rmtree(d, ignore_errors=True)

    def _deliverable_dirs(self, run_dir: Path) -> list[Path]:
        dirs = [Path(run_dir) / rel for rel in DELIV_DIRS]
        return [d for d in dirs if d.is_dir()]

    def _named_deliverables(self, run_dir: Path) -> list[str]:
        task_id = self._task_of.get(Path(run_dir).name, "")
        if not task_id:
            return []
        try:
            names = self._task_json(task_id).get("deliverables") or []
        except (OSError, json.JSONDecodeError):
            return []
        return [str(n) for n in names if isinstance(n, str) and n.strip()]

    def delivered(self, run_dir: Path) -> bool:
        """Every artifact the task named is present, or any file when it named none."""
        dirs = self._deliverable_dirs(run_dir)
        if not dirs:
            return False
        names = self._named_deliverables(run_dir)
        if names:
            return all(any((d / n).is_file() for d in dirs) for n in names)
        return any(p.is_file() for d in dirs for p in d.rglob("*"))

    def trajectory(self, run_dir: Path, *, obs_trunc: int) -> str:
        name = "opencode_session.jsonl" if self.substrate.name == "oc" else "codex_session.jsonl"
        stream = Path(run_dir) / name
        if not stream.is_file():
            return f"(no trajectory recorded for {Path(run_dir).name})"
        return self.substrate.read_trajectory(stream, obs_trunc=obs_trunc, max_chars=10**9)

    def deliverable_files(self, run_dir: Path) -> dict[str, str]:
        """filename -> readable text, from agent-authored bytes only."""
        key = str(Path(run_dir).resolve())
        if key in self._deliverables:
            return dict(self._deliverables[key])
        files: dict[str, str] = {}
        for base in self._deliverable_dirs(run_dir):
            for p in sorted(q for q in base.rglob("*") if q.is_file()):
                rel = p.relative_to(base).as_posix()
                if rel in files:
                    continue
                try:
                    if p.suffix.lower() in TEXT_SUFFIXES:
                        text = p.read_text(encoding="utf-8", errors="replace")
                    else:
                        text = f"(binary file: {p.name}, {p.stat().st_size} bytes)"
                except OSError as exc:
                    text = f"(error reading {rel}: {exc})"
                if len(text) > DELIV_CAP:
                    more = len(text) - DELIV_CAP
                    text = text[:DELIV_CAP] + f"\n\n... [truncated, {more} more chars]"
                files[rel] = text
        self._deliverables[key] = files
        return dict(files)

    def deliverable_text(self, run_dir: Path) -> str:
        files = self.deliverable_files(run_dir)
        if not files:
            return "(this run produced NO deliverables)"
        return "\n\n".join(
            f"----- FILE: {name} -----\n{text.strip() or '(empty file)'}"
            for name, text in files.items()
        )

    def base_harness(self) -> Path:
        """The unmodified tree this cell starts from: the always-on body plus the launch kit.

        Materialised once per process and per pid, because a sweep's threads share the path.
        """
        d = self.spec.runs_root / f"_base_harness_{self.substrate.name}_{os.getpid()}"
        heading = (
            "You are the engineer on call for this service. The assignment names the scenario, "
            "the constraints and the artifacts."
        )
        with _BASE_LOCK:
            if d not in _BASE_DONE:
                if self.substrate.name == "oc":
                    HT.write_tree(
                        OPENCODE_SURFACE,
                        {
                            "systemprompt.md": _base_body(heading),
                            "guardrails.md": "",
                            "LongTermMEMORY.md": "",
                            "tool_descriptions/tool_guidance.md": "",
                        },
                        d,
                    )
                else:
                    HT.write_tree(CODEX_SURFACE, {"AGENTS.md": _base_body(heading)}, d)
                self._copy_kit(d)
                _BASE_DONE.add(d)
        return d

    def validate_tree(self, tree: dict[str, str]) -> None:
        """SRV invariants a harness must keep: the kit path, the deliverable path, no leak."""
        body = HT.always_on_text(self.substrate.surface(), tree)
        low = body.lower()
        errs = []
        if "launch_kit/" not in body:
            errs.append(
                "the always-on body must name 'launch_kit/run_service.sh' - the reference "
                "launcher the agent is expected to edit lives there, and nothing else tells "
                "the agent how to bring a service up"
            )
        if "deliverables/" not in body:
            errs.append(
                "the always-on body must say deliverables go to 'deliverables/<filename>' "
                "explicitly - anything left elsewhere is invisible to delivery"
            )
        for bad in ("thresholds", "reference/", "baseline.json"):
            if bad in low:
                errs.append(
                    f"the always-on body names {bad!r}, which is judge-side material; a harness "
                    f"that ships it hands the agent the answer key"
                )
        if errs:
            raise ValueError(" | ".join(errs))

    def preflight(self) -> None:
        """Everything this host can check offline. The machine checks live in the runtime."""
        problems = []
        if not SRV_DIR.is_dir():
            problems.append(f"SRV task tree missing at {SRV_DIR}; run harness2-setup srv")
        missing = [n for n in KIT_NAMES if not (KIT_DIR / n).is_file()]
        if missing:
            problems.append(
                f"launch kit incomplete under {KIT_DIR}: missing {', '.join(missing)}; restore "
                f"them from the repository (they ship with the benchmark)"
            )
        try:
            srv_image()
        except config.ConfigError as exc:
            problems.append(str(exc))
        if _grader() is None:
            problems.append(
                "srv_grade is not importable: a run would be measured and then marked as "
                "nothing; bring the srv benchmark modules in before spending a rollout"
            )
        if problems:
            raise RuntimeError("; ".join(problems))

    def score(self, run_dir: Path, task_id: str) -> dict | None:
        """'srv_grade.grade' over the metrics 'srv_runtime.collect' wrote.

        task.json is handed over as-is: a measure task may carry an empty 'thresholds' (D1) and
        a baseline value may be null, one number, or a per-scenario map (D4). The adapter never
        interprets either, and a judge failure is None, never 0, and nothing is cached.
        """
        metrics = _json_or_none(Path(run_dir) / "metrics.json")
        if metrics is None:
            print(f"  [srv] no readable metrics.json in {run_dir}: unscored (not 0)", flush=True)
            return None
        grader = _grader()
        if grader is None:
            print("  [srv] srv_grade is not importable: unscored (not 0)", flush=True)
            return None
        try:
            got = grader.grade(metrics, self._task_json(task_id))
        except Exception as exc:
            print(
                f"  [srv] grade raised {type(exc).__name__}: {exc}: unscored (not 0)", flush=True
            )
            return None
        if not isinstance(got, dict) or got.get("score") is None:
            print(f"  [srv] {task_id} carries no grade (needs calibration): unscored", flush=True)
            return None
        frac = min(1.0, max(0.0, float(got["score"])))
        return {
            "n_passed": round(100 * frac),
            "n_criteria": 100,
            "all_pass": bool(got.get("passed")),
            "metric": "srv_score",
            "score": frac,
            "detail": got.get("detail") or {},
        }


__all__ = [
    "SrvAdapter",
    "srv_spec",
    "srv_prompts",
    "srv_image",
    "strip_front_matter",
    "SRV_BASE",
    "SRV_CATEGORIES",
    "SRV_DIR",
    "KIT_DIR",
    "KIT_NAMES",
    "CATEGORIES",
    "MODES",
    "MODEL",
    "MODEL_CTX_TOKENS",
    "IMAGE_ENV",
]
