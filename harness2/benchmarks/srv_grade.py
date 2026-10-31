"""SRV-Bench grading: pure functions over metrics.json and a task spec.

CONTRACT.md section 6 plus the v1.1 arbitration record (section 8):

* D1 - a ``measure`` task with ``thresholds: {}`` is legal; measure never reads
  thresholds at all (it grades process, not numbers).
* D3 - accuracy is a 0..100 percentage; ``tolerance_pct`` is percentage points and
  nothing is divided by 100 on either side.
* D4 - a baseline metric may be null (uncalibrated), a number (shared by every
  compared scenario) or a {scenario: number} map.  Every comparable scenario must
  pass on its own; zero comparable scenarios means unscored (None, never 0).

No IO, no network, no subprocess, so ``verify.py`` files can self-test offline with
the same code the judge runs.  ``score`` is the fraction of passed checks.
"""

from __future__ import annotations

from .srv_runtime import (
    ACCURACY_DATASET,
    AISBENCH_MODEL,
    MODEL_CTX_TOKENS,
    MODEL_ID,
    PERF_DATASET,
    applicable_scenarios,
)

CATEGORIES = ("config", "code", "measure")

# CONTRACT.md section 6, row ``measure``: five equally weighted checks.
MEASURE_CHECKS = (
    "service_up",
    "aisbench_cmd",
    "report_parsed",
    "scenarios_coverage",
    "deliverables",
)

# Spec keys that ask for an accuracy run; thresholds are deliberately excluded (D1).
ACCURACY_KEYS = ("accuracy", "require_accuracy", "dataset", "accuracy_dataset")

_LOWER_IS_BETTER = ("ttft", "tpot", "latency", "p90", "_ms")
_ACCURACY = "accuracy"
_PERF_METRICS = ("throughput_req_s", "ttft_ms_p90", "tpot_ms")


def grade(metrics: dict, spec: dict) -> dict | None:
    """Score one run; None means unscored (never cached as 0)."""
    try:
        return _grade(metrics, spec)
    except Exception:  # noqa: BLE001 - a judge must not crash on a bad metrics.json
        return None


def _grade(metrics: dict, spec: dict) -> dict | None:
    if not isinstance(metrics, dict) or not isinstance(spec, dict):
        return None
    category = str(spec.get("category") or "").strip()
    if category not in CATEGORIES:
        return None
    if metrics.get("ok") is not True:
        return None
    if "deliverables" not in metrics:
        # CONTRACT 9/D8: the key is runtime-filled; its absence is a runtime gap,
        # not evidence that the agent delivered nothing.
        return None
    scenarios = metrics.get("scenarios")
    if not isinstance(scenarios, dict) or not scenarios:
        return None
    blocks = [block for block in scenarios.values() if isinstance(block, dict)]
    if not blocks or all(block.get("skipped") for block in blocks):
        return None
    if category == "measure":
        return _grade_measure(metrics, spec, scenarios)
    return _grade_numeric(metrics, spec, scenarios, category)


# --------------------------------------------------------------------------- #
# config / code
# --------------------------------------------------------------------------- #


def _grade_numeric(metrics: dict, spec: dict, scenarios: dict, category: str) -> dict | None:
    baseline = spec.get("baseline") if isinstance(spec.get("baseline"), dict) else {}
    thresholds = spec.get("thresholds") if isinstance(spec.get("thresholds"), dict) else {}
    under_test = _scenarios_under_test(spec, scenarios)
    notes: list[str] = []
    checks: list[dict] = []

    violation = ""
    patch_check: dict | None = None
    if category == "code":
        violation, patch_check, patch_notes, unscored = _patch_check(metrics, spec)
        notes += patch_notes
        if unscored:
            # CONTRACT 9/D8: no changed_files means the runtime reported no patch,
            # so the path constraint is unverifiable - unscored, never a violation.
            return None

    if violation:
        return {
            "score": 0.0,
            "passed": False,
            "detail": {
                "category": category,
                "violation": violation,
                "checks": [],
                "failed": ["patch_paths"],
                "notes": notes,
                "per_scenario": {},
            },
        }

    for raw_metric, rule in thresholds.items():
        metric = str(raw_metric)
        if not isinstance(rule, dict):
            notes.append(f"threshold for {metric} is not an object; ignored")
            continue
        kind = str(rule.get("kind") or "")
        if metric == _ACCURACY:
            checks += _accuracy_checks(metrics, rule, baseline, scenarios, under_test, notes)
        elif kind == "rel_gain":
            checks += _rel_gain_checks(rule, metric, baseline, scenarios, under_test, notes)
        elif kind == "not_below":
            checks += _not_below_checks(rule, metric, baseline, scenarios, under_test, notes)
        elif kind == "budget":
            checks += _budget_checks(rule, metric, baseline, scenarios, under_test, notes)
        else:
            notes.append(f"unknown threshold kind {kind!r} for {metric}; ignored")

    if not checks:
        # No comparable scenario (D4): baseline still null, or every scenario skipped.
        # The patch check alone cannot score a task whose numbers are uncalibrated.
        return None
    if patch_check is not None:
        checks.append(patch_check)
    failed = [check["label"] for check in checks if not check["ok"]]
    return {
        "score": round((len(checks) - len(failed)) / len(checks), 6),
        "passed": not failed,
        "detail": {
            "category": category,
            "violation": "",
            "checks": checks,
            "failed": failed,
            "notes": notes,
            "per_scenario": _per_scenario_summary(checks),
        },
    }


def _rel_gain_checks(
    rule: dict,
    metric: str,
    baseline: dict,
    scenarios: dict,
    under_test: list[str],
    notes: list[str],
) -> list[dict]:
    min_pct = _as_float(rule.get("min_pct"))
    if min_pct is None:
        notes.append(f"{metric}: rel_gain without min_pct; ignored")
        return []
    direction = str(rule.get("direction") or _default_direction(metric))
    checks: list[dict] = []
    for name in under_test:
        base = _baseline_for(baseline, metric, name)
        if base is None:
            continue
        measured = _as_float((scenarios.get(name) or {}).get(metric))
        label = f"{metric}@{name}"
        if base == 0:
            notes.append(f"{label}: baseline 0 cannot express rel_gain; skipped")
            continue
        if measured is None:
            checks.append(
                {
                    "name": metric,
                    "label": label,
                    "scenario": name,
                    "kind": "rel_gain",
                    "baseline": base,
                    "measured": None,
                    "direction": direction,
                    "gain_pct": None,
                    "min_pct": min_pct,
                    "ok": False,
                    "detail": f"{metric} not measured for scenario {name}",
                }
            )
            continue
        if direction == "lower_is_better":
            gain = (base - measured) / base * 100.0
        else:
            gain = (measured - base) / base * 100.0
        checks.append(
            {
                "name": metric,
                "label": label,
                "scenario": name,
                "kind": "rel_gain",
                "baseline": base,
                "measured": measured,
                "direction": direction,
                "gain_pct": round(gain, 6),
                "min_pct": min_pct,
                "ok": gain >= min_pct,
                "detail": (
                    f"{_formula(direction, metric)} = {gain:.4f}% vs min_pct {min_pct}%"
                ),
            }
        )
    return checks


def _not_below_checks(
    rule: dict,
    metric: str,
    baseline: dict,
    scenarios: dict,
    under_test: list[str],
    notes: list[str],
) -> list[dict]:
    tolerance = _as_float(rule.get("tolerance_pct"))
    if tolerance is None:
        notes.append(f"{metric}: not_below without tolerance_pct; ignored")
        return []
    checks: list[dict] = []
    for name in under_test:
        base = _baseline_for(baseline, metric, name)
        if base is None:
            continue
        measured = _as_float((scenarios.get(name) or {}).get(metric))
        floor = base * (1.0 - tolerance / 100.0)
        ok = measured is not None and measured >= floor - 1e-9
        checks.append(
            {
                "name": metric,
                "label": f"{metric}@{name}",
                "scenario": name,
                "kind": "not_below",
                "baseline": base,
                "measured": measured,
                "delta_pct": round((measured - base * (1.0 - tolerance / 100.0)), 6)
                if measured is not None
                else None,
                "floor": round(floor, 6),
                "tolerance_pct": tolerance,
                "ok": ok,
                "detail": f"measured {measured} vs floor {floor:.6f}",
            }
        )
    return checks


def _budget_checks(
    rule: dict,
    metric: str,
    baseline: dict,
    scenarios: dict,
    under_test: list[str],
    notes: list[str],
) -> list[dict]:
    max_s = _as_float(rule.get("max_s"))
    if max_s is None:
        notes.append(f"{metric}: budget without max_s; ignored")
        return []
    checks: list[dict] = []
    for name in under_test:
        measured = _as_float((scenarios.get(name) or {}).get(metric))
        if measured is None:
            continue
        checks.append(
            {
                "name": metric,
                "label": f"{metric}@{name}",
                "scenario": name,
                "kind": "budget",
                "baseline": None,
                "measured": measured,
                "max_s": max_s,
                "ok": measured <= max_s,
                "detail": f"measured {measured}s vs budget {max_s}s",
            }
        )
    return checks


def _accuracy_checks(
    metrics: dict,
    rule: dict,
    baseline: dict,
    scenarios: dict,
    under_test: list[str],
    notes: list[str],
) -> list[dict]:
    tolerance = _as_float(rule.get("tolerance_pct"))
    if tolerance is None:
        notes.append("accuracy: threshold without tolerance_pct; using 0.5 percentage points")
        tolerance = 0.5
    checks: list[dict] = []
    for name in under_test:
        base = _baseline_for(baseline, _ACCURACY, name)
        if base is None:
            continue
        measured = _measured_accuracy(metrics, scenarios, name)
        floor = base - tolerance
        ok = measured is not None and measured >= floor - 1e-9
        checks.append(
            {
                "name": _ACCURACY,
                "label": f"{_ACCURACY}@{name}",
                "scenario": name,
                "kind": "not_below",
                "unit": "percent",
                "baseline": base,
                "measured": measured,
                "delta_pct": round(measured - floor, 6) if measured is not None else None,
                "floor_pct": round(floor, 6),
                "tolerance_pct": tolerance,
                "ok": ok,
                "detail": (
                    f"accuracy {measured}% vs floor {floor:.4f}%"
                    f" (baseline {base} - tolerance {tolerance} percentage points)"
                ),
            }
        )
    return checks


def _patch_check(metrics: dict, spec: dict) -> tuple[str, dict | None, list[str], bool]:
    """Return (violation text, patch check or None, notes, unscored)."""
    allowed = [_norm_path(p) for p in spec.get("allowed_paths") or [] if str(p).strip()]
    # CONTRACT 9/D7: both spellings are equivalent; either one alone must bind.
    blocked_paths = [
        _norm_path(path)
        for key in ("protected_paths", "forbidden_paths")
        for path in spec.get(key) or []
        if str(path).strip()
    ]
    notes: list[str] = []
    if not allowed and not blocked_paths:
        notes.append("no allowed_paths/protected_paths/forbidden_paths declared; not checked")
        return "", None, notes, False
    changed = _changed_files(metrics)
    if not changed:
        notes.append("metrics.json reports no changed_files; patch location unverifiable")
        return "", None, notes, True
    for path in changed:
        norm = _norm_path(path)
        for blocked in blocked_paths:
            if norm == blocked or norm.startswith(blocked + "/"):
                return f"patch touches forbidden path {path}", None, notes, False
        if allowed and not any(norm == item or norm.startswith(item + "/") for item in allowed):
            return f"patch outside allowed paths: {path}", None, notes, False
    return (
        "",
        {
            "name": "patch_paths",
            "label": "patch_paths",
            "ok": True,
            "detail": f"{len(changed)} changed file(s) inside allowed paths",
        },
        notes,
        False,
    )


# --------------------------------------------------------------------------- #
# measure (D1: thresholds are never read here)
# --------------------------------------------------------------------------- #


def _grade_measure(metrics: dict, spec: dict, scenarios: dict) -> dict | None:
    required = [str(name) for name in spec.get("scenarios") or list(scenarios)]
    measured = [
        name for name in required
        if isinstance(scenarios.get(name), dict) and not scenarios[name].get("skipped")
    ]
    if not measured:
        measured = [
            str(name) for name, block in scenarios.items()
            if isinstance(block, dict) and not block.get("skipped")
        ]
    checks = [
        _service_check(metrics),
        _command_check(metrics, spec, scenarios, measured),
        _report_check(metrics, spec, scenarios, measured),
        _coverage_check(spec, scenarios, required),
        _deliverable_check(spec, metrics),
    ]
    failed = [check["name"] for check in checks if not check["ok"]]
    return {
        "score": round((len(checks) - len(failed)) / len(checks), 6),
        "passed": not failed,
        "detail": {
            "category": "measure",
            "violation": "",
            "checks": checks,
            "failed": failed,
            "notes": [],
            "per_scenario": {},
        },
    }


def _service_check(metrics: dict) -> dict:
    service = metrics.get("service")
    if not isinstance(service, dict):
        return _check("service_up", False, "metrics.service is missing")
    if "ok" in service:
        ok = bool(service.get("ok"))
        detail = f"service.ok={service.get('ok')!r} (mode {service.get('mode')!r})"
    else:
        ok = bool(metrics.get("ok")) and bool(service.get("p_cmd"))
        detail = "service.ok absent; inferred from p_cmd + metrics.ok"
    if not ok and service.get("error"):
        detail += f"; error: {service['error']}"
    return _check("service_up", ok, detail)


def _command_check(metrics: dict, spec: dict, scenarios: dict, measured: list[str]) -> dict:
    perf_dataset = str(spec.get("perf_dataset") or PERF_DATASET)
    tokens = [
        str(token)
        for token in (spec.get("aisbench_model"), spec.get("model"), AISBENCH_MODEL, MODEL_ID)
        if token
    ]
    problems: list[str] = []
    if not measured:
        problems.append("no measured scenario to audit")
    for name in measured:
        cmd = str((scenarios.get(name) or {}).get("aisbench_cmd") or "").strip()
        if not cmd:
            problems.append(f"{name}: aisbench_cmd missing")
            continue
        if perf_dataset not in cmd:
            problems.append(f"{name}: cmd does not use dataset {perf_dataset}")
        if "perf" not in cmd:
            problems.append(f"{name}: cmd is not in perf mode")
        if not any(token in cmd for token in tokens):
            problems.append(f"{name}: cmd names no expected model ({', '.join(tokens)})")
    block = metrics.get("accuracy") if isinstance(metrics.get("accuracy"), dict) else {}
    acc_cmd = str(block.get("cmd") or "").strip()
    if _accuracy_required(spec) and not acc_cmd:
        problems.append("accuracy: aisbench_cmd missing")
    if acc_cmd:
        dataset = str(spec.get("dataset") or spec.get("accuracy_dataset") or ACCURACY_DATASET)
        if dataset not in acc_cmd:
            problems.append(f"accuracy: cmd does not use dataset {dataset}")
        if "accuracy" not in acc_cmd:
            problems.append("accuracy: cmd is not in accuracy mode")
        if not any(token in acc_cmd for token in tokens):
            problems.append("accuracy: cmd names no expected model")
    return _check(
        "aisbench_cmd",
        not problems,
        (
            "unrecorded, wrong dataset/mode/model: " + "; ".join(problems)
            if problems
            else "recorded aisbench commands carry dataset, mode and model"
        ),
    )


def _report_check(metrics: dict, spec: dict, scenarios: dict, measured: list[str]) -> dict:
    problems: list[str] = []
    if not measured:
        problems.append("no measured scenario to parse")
    for name in measured:
        block = scenarios.get(name) or {}
        for key in _PERF_METRICS:
            if not _positive(block.get(key)):
                problems.append(f"{name}: {key} missing or not a positive number")
        if not str(block.get("raw") or "").strip():
            problems.append(f"{name}: raw report path missing")
    block = metrics.get("accuracy") if isinstance(metrics.get("accuracy"), dict) else {}
    if _accuracy_required(spec) or block:
        if _as_float(block.get("accuracy")) is None:
            problems.append("accuracy: measured accuracy missing")
        if not str(block.get("raw") or "").strip():
            problems.append("accuracy: raw report path missing")
    return _check(
        "report_parsed",
        not problems,
        (
            "aisbench report not usable: " + "; ".join(problems)
            if problems
            else "every measured scenario has numeric metrics and a raw report path"
        ),
    )


def _coverage_check(spec: dict, scenarios: dict, required: list[str]) -> dict:
    usable, _ = applicable_scenarios(required)
    problems: list[str] = []
    for name in required:
        block = scenarios.get(name)
        if not isinstance(block, dict):
            problems.append(f"{name}: absent from metrics.scenarios")
            continue
        if block.get("skipped"):
            reason = str(block.get("reason") or "").strip()
            if not reason:
                problems.append(f"{name}: skipped without a reason")
            elif name in usable:
                problems.append(
                    f"{name}: skipped although it fits in {MODEL_CTX_TOKENS} tokens ({reason})"
                )
        elif not all(_positive(block.get(key)) for key in _PERF_METRICS):
            problems.append(f"{name}: not skipped but carries no measured values")
    return _check(
        "scenarios_coverage",
        not problems,
        (
            "coverage gaps: " + "; ".join(problems)
            if problems
            else f"required scenarios covered: {', '.join(required)}"
        ),
    )


def _deliverable_check(spec: dict, metrics: dict) -> dict:
    """Every declared file the runtime found on disk (never the agent's own claim)."""
    wanted = [str(item) for item in spec.get("deliverables") or []]
    if not wanted:
        return _check("deliverables", True, "no deliverables declared")
    present = _present_files(metrics)
    missing = [item for item in wanted if not _has_file(present, item)]
    return _check(
        "deliverables",
        not missing,
        (
            "missing deliverables: " + ", ".join(missing)
            if missing
            else f"all deliverables present: {', '.join(wanted)}"
        ),
    )


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #


def _check(name: str, ok: bool, detail: str) -> dict:
    return {"name": name, "label": name, "ok": bool(ok), "detail": detail}


def _scenarios_under_test(spec: dict, scenarios: dict) -> list[str]:
    """Scenarios a numeric category compares, in spec order (D4).

    A scenario that metrics marks skipped is out; a scenario the spec lists but
    metrics never reported still counts when it is measurable, so a missing
    measurement fails instead of disappearing.
    """
    requested = [str(name) for name in spec.get("scenarios") or []]
    if requested:
        usable, _ = applicable_scenarios(requested)
        ordered: list[str] = []
        for name in requested:
            block = scenarios.get(name)
            if isinstance(block, dict):
                if not block.get("skipped"):
                    ordered.append(name)
            elif name in usable:
                ordered.append(name)
        if ordered:
            return ordered
    return [
        str(name) for name, block in scenarios.items()
        if isinstance(block, dict) and not block.get("skipped")
    ]


def _baseline_for(baseline: dict, metric: str, scenario: str) -> float | None:
    """null -> None, number -> shared by every scenario, {scenario: number} -> per scenario."""
    value = (baseline or {}).get(metric)
    if isinstance(value, dict):
        return _as_float(value.get(scenario))
    return _as_float(value)


def _measured_accuracy(metrics: dict, scenarios: dict, scenario: str) -> float | None:
    block = scenarios.get(scenario) if isinstance(scenarios.get(scenario), dict) else {}
    value = _as_float(block.get("accuracy"))
    if value is not None:
        return value
    block = metrics.get("accuracy") if isinstance(metrics.get("accuracy"), dict) else {}
    return _as_float(block.get("accuracy"))


def _per_scenario_summary(checks: list[dict]) -> dict:
    """Per scenario: measured / baseline / gain / ok for every graded metric (D4)."""
    summary: dict[str, dict] = {}
    for check in checks:
        scenario = str(check.get("scenario") or "")
        if not scenario or check.get("name") == "patch_paths":
            continue
        entry = summary.setdefault(scenario, {})
        entry[check["name"]] = {
            "measured": check.get("measured"),
            "baseline": check.get("baseline"),
            "gain_pct": check.get("gain_pct", check.get("delta_pct")),
            "ok": bool(check["ok"]),
        }
    return summary


def _default_direction(metric: str) -> str:
    name = metric.lower()
    lower = any(token in name for token in _LOWER_IS_BETTER)
    return "lower_is_better" if lower else "higher_is_better"


def _formula(direction: str, metric: str) -> str:
    if direction == "lower_is_better":
        return f"(baseline - measured)/baseline*100 for {metric}"
    return f"(measured - baseline)/baseline*100 for {metric}"


def _accuracy_required(spec: dict) -> bool:
    """Measure only: an explicit spec key asks for accuracy, thresholds never do (D1)."""
    return any(spec.get(key) for key in ACCURACY_KEYS)


def _changed_files(metrics: dict) -> list[str]:
    for key in ("changed_files", "files_changed"):
        value = metrics.get(key)
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
    for holder, key in (("patch", "changed_files"), ("patch", "files"), ("diff", "files")):
        block = metrics.get(holder)
        if isinstance(block, dict):
            value = block.get(key)
            if isinstance(value, (list, tuple)):
                return [str(item) for item in value]
    return []


def _present_files(metrics: dict) -> set[str]:
    """Files the runtime actually found (CONTRACT 9/D8: {filename: bool})."""
    names: set[str] = set()
    value = metrics.get("deliverables")
    if isinstance(value, dict):
        for name, flag in value.items():
            if isinstance(flag, str):
                names.add(flag)
            elif flag:
                names.add(str(name))
    elif isinstance(value, (list, tuple)):
        names.update(str(item) for item in value)
    elif isinstance(value, str):
        names.add(value)
    return names


def _has_file(present: set[str], wanted: str) -> bool:
    target = _norm_path(wanted)
    base = target.rsplit("/", 1)[-1]
    for item in present:
        norm = _norm_path(item)
        if norm == target or norm.rsplit("/", 1)[-1] == base:
            return True
    return False


def _norm_path(value: object) -> str:
    return str(value).replace("\\", "/").strip("/")


def _positive(value: object) -> bool:
    number = _as_float(value)
    return number is not None and number > 0


def _as_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None
