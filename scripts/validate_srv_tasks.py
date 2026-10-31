#!/usr/bin/env python3
"""Validate every srv task package offline: reading, schema, and the package's own verify.py.

    python scripts/validate_srv_tasks.py [--timeout 300] [--quiet]

Each package is checked here for what the adapter relies on -- id/directory agreement, the
category, the mode, the deliverable names, a non-empty instruction -- and then its own
verify.py is run. Package self-checks are not a single CLI: the config and measure packages
take the task directory as an argument, the code packages default to their own directory, so
a run that rejects the argument is retried without it (two attempts, the same content).

A tree with no packages prints one line and exits 0: a fresh checkout has none, and that is
not a failure. Only the standard library is imported, and srv_runtime / srv_grade stay
optional -- without them the scenario check reports that it was skipped rather than guessing
at their constants.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness2.benchmarks.srv import (  # noqa: E402
    CATEGORIES,
    MODES,
    SRV_DIR,
    strip_front_matter,
)

TASKS = SRV_DIR / "tasks"
ID_PREFIX = {"config": "cfg-", "measure": "msr-", "code": "kod-"}


def declared_scenarios() -> tuple[dict | None, str]:
    """SRV_SCENARIOS from the one place that defines it, plus a note when it is unavailable."""
    try:
        from harness2.benchmarks import srv_runtime
    except ImportError as exc:
        return None, f"srv_runtime not importable ({exc}); scenario check skipped"
    except Exception as exc:  # a runtime that breaks on import is a defect worth naming
        return None, f"srv_runtime failed to import: {type(exc).__name__}: {exc}"
    got = getattr(srv_runtime, "SRV_SCENARIOS", None)
    if not isinstance(got, dict):
        return None, "srv_runtime declares no SRV_SCENARIOS dict; scenario check skipped"
    return got, ""


CONTRACT_TOP = (
    "task_id", "ok", "error", "image", "model", "service", "deliverables", "scenarios",
)
CONTRACT_TOP_CODE = ("changed_files",)
CONTRACT_SERVICE = ("mode", "p_cmd", "d_cmd", "prefix_cache")
CONTRACT_SCENARIO = (
    "skipped", "reason", "throughput_req_s", "ttft_ms_p90", "tpot_ms", "accuracy",
    "aisbench_cmd", "raw",
)
CONTRACT_CORE = ("throughput_req_s", "ttft_ms_p90", "tpot_ms")
CONTRACT_SOURCE = "CONTRACT.md section 9 (D9)"
CONTRACT_RUNTIME_FILLED = ("deliverables", "changed_files")
FENCE = chr(96) * 3
KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _fenced(text: str) -> list[str]:
    """The bodies of the fenced blocks, in order, from their first JSON-looking line.

    A fence may carry the info string on the fence line or on a line of its own, which is what
    a pasted template looks like, so the body starts at the first line opening an object or an
    array.
    """
    out = []
    for body in text.split(FENCE)[1::2]:
        lines = body.splitlines()
        while lines and not lines[0].lstrip().startswith(("{", "[")):
            lines.pop(0)
        out.append("\n".join(lines))
    return out


def _looks_like_metrics(block: str) -> bool:
    return any(f'"{key}"' in block for key in ("task_id", "scenarios", "ok"))


def _json_metrics(block: str) -> dict | None:
    """The block as a dict, tolerating the // comments people type in a template."""
    body = "\n".join(ln for ln in block.splitlines() if not ln.strip().startswith("//"))
    try:
        got = json.loads(body)
    except json.JSONDecodeError:
        return None
    return got if isinstance(got, dict) else None


def _section(region: str, name: str) -> str | None:
    """The inside of the first balanced <name>{...}, or None when it is absent."""
    m = re.search(rf"\b{name}\s*\{{", region)
    if not m:
        return None
    index, depth = m.end(), 1
    while index < len(region) and depth:
        if region[index] == "{":
            depth += 1
        elif region[index] == "}":
            depth -= 1
        index += 1
    return region[m.end() : index - 1] if depth == 0 else None


def _brace_keys(region: str) -> tuple[set[str], set[str], set[str], set[str]]:
    """Key names out of a prose enumeration: top keys, service{...}, scenarios{...}, and any
    key written as "name（explanation）" on a line of its own."""
    service: set[str] = set()
    scenario: set[str] = set()
    top: set[str] = set()
    named: set[str] = set()
    for line in region.splitlines():
        m = re.match(r"^\s*([a-z][a-z0-9_]*)\s*[（(]", line)
        if m:
            named.add(m.group(1))
    for name, bucket in (("service", service), ("scenarios", scenario)):
        top.add(name)
        inner = _section(region, name)
        if inner is None:
            continue
        for chunk in re.split(r"[,、]", re.sub(r"[{}\"]", " ", inner)):
            token = chunk.strip().strip("<>").split(":")[0].strip()
            if KEY_RE.match(token):
                bucket.add(token)
    head = region
    brace = head.find("{")
    if brace >= 0:
        head = head[:brace]
    for chunk in re.split(r"[,、，：:]", head):
        token = chunk.strip().strip("*")
        if KEY_RE.match(token):
            top.add(token)
            continue
        # "deliverables（what the run handed over）" is how a bilingual brief lists a key
        m = re.match(r"^([a-z][a-z0-9_]*)[（(]", chunk.strip())
        if m:
            named.add(m.group(1))
    return top, service, scenario, named


def _from_json(spec: dict) -> tuple[set[str], set[str], set[str], set[str]]:
    top = {k for k in spec if KEY_RE.match(str(k))}
    service: set[str] = set()
    if isinstance(spec.get("service"), dict):
        service = {k for k in spec["service"] if KEY_RE.match(str(k))}
    scenario: set[str] = set()
    scenarios = spec.get("scenarios")
    if isinstance(scenarios, dict):
        for value in scenarios.values():
            if isinstance(value, dict):
                scenario |= {k for k in value if KEY_RE.match(str(k))}
    return top, service, scenario, set()


def _prose_region(text: str) -> str:
    """The lines around a prose key list: a "metrics.json" line plus the block under it, and
    any line that opens a service {...} or scenarios {...} section."""
    lines = text.splitlines()
    keep: list[str] = []
    for index, line in enumerate(lines):
        if "metrics.json" in line:
            for follow in lines[index : index + 8]:
                if not follow.strip():
                    break
                keep.append(follow)
        elif re.search(r"\b(scenarios|service)\s*\{", line):
            keep.append(line)
    return "\n".join(keep)


def check_metrics_contract(text: str) -> tuple[list[tuple[bool, str, str]], list[str]]:
    """instruction.md's metrics.json keys against CONTRACT.md section 4.

    The performance keys are required: an assignment that never names them leaves the agent
    guessing at the artifact the whole task is read from. 'deliverables' and 'changed_files' are
    filled by the runtime, so their absence from the text is not a defect (D8/D9), and keys the
    contract does not define are warnings -- a task may add a scene-local note while the reader
    upstream parses only the contracted ones.
    """
    items: list[tuple[bool, str, str]] = []
    warnings: list[str] = []
    blocks = [b for b in _fenced(text) if _looks_like_metrics(b)]
    top: set[str] = set()
    service: set[str] = set()
    scenario: set[str] = set()
    named: set[str] = set()
    if blocks:
        for block in blocks:
            parsed = _json_metrics(block)
            a, b, c, d = _from_json(parsed) if parsed else _brace_keys(block)
            top, service, scenario, named = top | a, service | b, scenario | c, named | d
    else:
        region = _prose_region(text)
        if not region.strip():
            return [(False, "metrics.json template present", "no fenced template, no key list")], []
        parsed = _json_metrics(region)
        top, service, scenario, named = (
            _from_json(parsed) if parsed else _brace_keys(region)
        )

    flat = top | service | scenario | named
    missing = [k for k in CONTRACT_CORE if k not in flat]
    items.append(
        (
            not missing,
            "metrics.json names every performance key",
            f"missing {missing}" if missing else ", ".join(CONTRACT_CORE),
        )
    )
    items.append(
        (
            "scenarios" in top,
            "metrics.json template keeps the scenario map",
            "" if "scenarios" in top else "no 'scenarios' key found in the template",
        )
    )
    known_top = set(CONTRACT_TOP) | set(CONTRACT_TOP_CODE) | set(CONTRACT_RUNTIME_FILLED)
    if "deliverables" not in (top | named):
        warnings.append(
            "the metrics.json template does not name 'deliverables'; collect() fills it by "
            "scanning the run directory (D8/D9), so this is a hint at most -- naming it would "
            "make the artifact easier to read"
        )
    for where, extra in (
        ("top level", sorted(top - known_top)),
        ("service", sorted(service - set(CONTRACT_SERVICE))),
        ("scenario", sorted(scenario - set(CONTRACT_SCENARIO))),
    ):
        if extra:
            warnings.append(
                f"instruction.md names {extra} under {where}, which {CONTRACT_SOURCE} does not "
                f"define; a reader that parses only the contract will ignore them"
            )
    return items, warnings


def check_schema(task_dir: Path, scenarios: dict | None) -> list[tuple[bool, str, str]]:
    items: list[tuple[bool, str, str]] = []
    path = task_dir / "task.json"
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [(False, "task.json parses", f"{path}: {exc}")]
    if not isinstance(spec, dict):
        return [(False, "task.json is an object", type(spec).__name__)]

    category = str(spec.get("category") or "")
    items.append(
        (spec.get("id") == task_dir.name, "id matches the directory",
         f"id={spec.get('id')!r} dir={task_dir.name!r}")
    )
    items.append(
        (category == task_dir.parent.name, "category matches the parent directory",
         f"category={category!r} parent={task_dir.parent.name!r}")
    )
    items.append(
        (spec.get("domain") == category, "domain equals category", f"domain={spec.get('domain')!r}")
    )
    items.append((category in CATEGORIES, "category is known", category))
    items.append((spec.get("mode") in MODES, "mode is known", str(spec.get("mode"))))
    prefix = ID_PREFIX.get(category, "")
    items.append((task_dir.name.startswith(prefix), f"directory name starts with {prefix!r}",
                  task_dir.name))

    wanted = spec.get("scenarios")
    ok = isinstance(wanted, list) and bool(wanted) and all(isinstance(s, str) and s for s in wanted)
    detail = "" if ok else repr(wanted)
    if ok and scenarios is not None:
        unknown = [s for s in wanted if s not in scenarios]
        ok = not unknown
        detail = f"not declared by the runtime: {unknown}" if unknown else ""
    items.append((ok, "scenarios are ones the runtime declares", detail))

    names = spec.get("deliverables")
    ok = isinstance(names, list) and all(isinstance(n, str) and n.strip() for n in names)
    items.append((ok, "deliverables is a list of names", "" if ok else repr(names)))

    instruction = task_dir / "instruction.md"
    text = ""
    if instruction.is_file():
        text = instruction.read_text(encoding="utf-8", errors="replace")
    items.append((bool(text.strip()), "instruction.md exists and is not empty", str(instruction)))
    body = strip_front_matter(text)
    items.append((bool(body.strip()), "the body survives front-matter stripping",
                  f"{len(body)} chars"))
    return items


def _run(cmd: list[str], task_dir: Path, timeout: int):
    """UTF-8 in and out: the packages print their native language, and a Windows console
    code page is not the authority on what they wrote."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        cmd,
        cwd=str(task_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )


def _norm_path(value: str) -> str:
    return str(value).strip().replace("\\", "/").strip("/")


def _path_list(spec: dict, name: str) -> list[str]:
    value = spec.get(name)
    if not isinstance(value, list):
        return []
    return [_norm_path(v) for v in value if isinstance(v, str) and _norm_path(v)]


def check_paths(spec: dict) -> tuple[list[tuple[bool, str, str]], list[str]]:
    """D7: forbidden_paths and protected_paths are two spellings of one constraint.

    The union is what applies, so a task may ship either or both; what must not happen is a path
    that is guarded and explicitly allowed at the same time, because then the patch-scope rule
    has no answer for it.
    """
    items: list[tuple[bool, str, str]] = []
    warnings: list[str] = []
    forbidden = _path_list(spec, "forbidden_paths")
    protected = _path_list(spec, "protected_paths")
    allowed = _path_list(spec, "allowed_paths")
    known = (("forbidden_paths", forbidden), ("protected_paths", protected))
    spellings = [name for name, paths in known if paths]
    if spellings:
        items.append((True, "a guarded-path constraint is declared (either spelling)",
                      ", ".join(spellings)))
    elif str(spec.get("category") or "") == "code":
        # Only a patch task has a scope to guard; config and measure tasks have none.
        items.append((False, "a guarded-path constraint is declared (either spelling)",
                      "neither forbidden_paths nor protected_paths on a code task"))
    if not spellings:
        return items, warnings
    guarded = sorted(set(forbidden) | set(protected))
    both = sorted(set(forbidden) & set(protected))
    if both:
        warnings.append(
            f"both spellings guard {both}; the union applies, so this is harmless but one "
            f"spelling is enough"
        )
    clash = sorted(set(guarded) & set(allowed))
    items.append(
        (
            not clash,
            "no path is both guarded and allowed",
            f"{clash} is guarded and allowed at once; the patch-scope rule cannot decide it"
            if clash
            else "",
        )
    )
    nested = sorted(
        f"{path} (allowed) sits under {guard} (guarded)"
        for path in allowed
        for guard in guarded
        if path.startswith(guard + "/")
    )
    if nested:
        warnings.append(
            "allowed path under a guarded directory: " + "; ".join(nested[:5])
            + " -- resolvable, but say in the task which rule wins"
        )
    return items, warnings


def run_verify(task_dir: Path, timeout: int) -> tuple[bool, str, str]:
    """The package's own offline self-check, tolerating both CLIs its categories ship."""
    script = task_dir / "verify.py"
    if not script.is_file():
        return True, "package self-check", "no verify.py in this package; skipped"
    try:
        r = _run([sys.executable, str(script), str(task_dir)], task_dir, timeout)
        if r.returncode != 0 and "unrecognized arguments" in (r.stderr or ""):
            r = _run([sys.executable, str(script)], task_dir, timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, "verify.py runs", f"{type(exc).__name__}: {exc}"
    if r.returncode == 0:
        lines = [ln for ln in (r.stdout or "").strip().splitlines() if ln.strip()]
        return True, "verify.py passes", lines[-1] if lines else ""
    blob = (r.stdout or "") + (r.stderr or "")
    fails = [
        ln.strip()
        for ln in blob.splitlines()
        if ln.strip().startswith(("[FAIL]", "FAIL", "Traceback", "  File", "Error"))
    ]
    return False, "verify.py passes", "\n      ".join(fails[:8]) or blob.strip()[-400:]


def check_runtime_agreement() -> tuple[list[tuple[bool, str, str]], list[str]]:
    """D14: one context window, two copies.

    srv.py restates MODEL and MODEL_CTX_TOKENS so that the base harness body and the dry-run
    plan render on a host that cannot import srv_runtime. Two copies drift, and a drifted copy
    makes "which scenarios are measurable" disagree with what the plan claims, so the copies are
    compared here: the constants are allowed to exist twice, they are not allowed to differ.
    """
    from harness2.benchmarks import srv as S

    try:
        from harness2.benchmarks import srv_runtime
    except ImportError as exc:
        return [], [
            f"srv_runtime not importable ({exc}); the context-window agreement check was skipped"
        ]
    except Exception as exc:  # a runtime that breaks on import is a defect worth naming
        return [
            (False, "srv_runtime imports", f"{type(exc).__name__}: {exc}")
        ], []

    items: list[tuple[bool, str, str]] = []
    theirs = getattr(srv_runtime, "MODEL_CTX_TOKENS", None)
    ours = S.MODEL_CTX_TOKENS
    if not isinstance(theirs, int):
        items.append(
            (
                False,
                "srv.py MODEL_CTX_TOKENS equals srv_runtime.MODEL_CTX_TOKENS",
                f"srv_runtime declares none (srv.py has {ours}); CONTRACT.md section 5 requires it",
            )
        )
    else:
        items.append(
            (
                theirs == ours,
                "srv.py MODEL_CTX_TOKENS equals srv_runtime.MODEL_CTX_TOKENS",
                ""
                if theirs == ours
                else f"srv.py={ours} srv_runtime={theirs}: the skip rule and the dry-run plan "
                f"would disagree about what fits the window",
            )
        )
    their_model = getattr(srv_runtime, "MODEL_ID", None)
    if their_model is not None:
        items.append(
            (
                their_model == S.MODEL,
                "srv.py MODEL equals srv_runtime.MODEL_ID",
                "" if their_model == S.MODEL else f"srv.py={S.MODEL} srv_runtime={their_model}",
            )
        )
    return items, []


def main(argv: list[str] | None = None) -> int:
    # The packages print their own language, so the console code page is not the authority on
    # what this script may print back.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timeout", type=int, default=300, help="seconds allowed per verify.py run")
    p.add_argument("--quiet", action="store_true", help="print only failures and the summary")
    args = p.parse_args(argv)

    packages = sorted(TASKS.glob("*/*/task.json")) if TASKS.is_dir() else []
    scenarios, note = declared_scenarios()
    if note:
        print(f"[INFO] {note}")
    agreement, agreement_warnings = check_runtime_agreement()
    for passed, name, detail in agreement:
        mark = "ok" if passed else "FAIL"
        print(f"[{mark}] {name}" + (f" - {detail}" if detail else ""))
    for message in agreement_warnings:
        print(f"[INFO] {message}")
    module_failed = sum(1 for passed, _, _ in agreement if not passed)
    if not packages:
        print(f"[INFO] no task packages under {TASKS} (an empty tree is not a failure)")
        print(
            f"result: 0 tasks, {module_failed} failure(s) outside the task tree, "
            f"{len(agreement_warnings)} warning(s)"
        )
        return 1 if module_failed else 0

    failed = module_failed
    warned = len(agreement_warnings)
    for path in packages:
        task_dir = path.parent
        task_id = f"{task_dir.parent.name}/{task_dir.name}"
        items = check_schema(task_dir, scenarios)
        instruction = task_dir / "instruction.md"
        text = ""
        if instruction.is_file():
            text = instruction.read_text(encoding="utf-8", errors="replace")
        contract_items, warnings = check_metrics_contract(text)
        items.extend(contract_items)
        try:
            spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            spec = {}
        if isinstance(spec, dict):
            path_items, path_warnings = check_paths(spec)
            items.extend(path_items)
            warnings.extend(path_warnings)
        items.append(run_verify(task_dir, args.timeout))
        bad = [it for it in items if not it[0]]
        verdict = "FAIL" if bad else "ok"
        tail = f", {len(warnings)} warning(s)" if warnings else ""
        print(f"[{verdict}] {task_id}: {len(items) - len(bad)} passed, {len(bad)} failed{tail}")
        for passed, name, detail in items:
            if passed and args.quiet:
                continue
            mark = "PASS" if passed else "FAIL"
            print(f"    [{mark}] {name}" + (f" - {detail}" if detail else ""))
        for message in warnings:
            print(f"    [WARN] {message}")
        failed += 1 if bad else 0
        warned += len(warnings)

    total = len(packages)
    print(
        f"result: {total - failed} task(s) passed, {failed} failed, {warned} warning(s) "
        f"(of {total})"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
