#!/usr/bin/env python3
"""Offline self-check for the SRV-Bench class-3 task .

What it does:
  * validates task.json against the CONTRACT.md section 3 schema (plus the
    allowed_paths / forbidden_paths extension CONTRACT.md section 6 requires
    for the code category);
  * validates that instruction.md cannot leak the judge side;
  * proves the patch-scope rule is executable, not merely declared, by running
    the scope checker over synthetic accept/reject cases and over a real diff.

It imports neither srv_runtime nor srv_grade: those belong to the runtime task
(A) and are absent from a checkout that has not been set up, so the scenario
rule of CONTRACT.md section 5 is mirrored below as a pure local function and
self-tested. No third-party dependency either; nothing here needs docker, an
NPU or network access.

    python verify.py                  # full offline self-check, exit 0 = pass
    python verify.py --patch X.diff    # enforce the scope rule on a real patch
    python verify.py --changed A B     # enforce the scope rule on path arguments
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
TASK_ID = "kod-moe-or-attn-comm"
SCENARIOS = ("4k2k", "16k1k")
# CONTRACT.md section 5 (single source of truth is srv_runtime.SRV_SCENARIOS;
# this copy exists only so the check runs without that module).
SRV_SCENARIOS = {
    "4k2k": (4096, 2048),
    "16k1k": (16384, 1024),
    "64k1k": (65536, 1024),
    "128k1k": (131072, 1024),
}
MODEL_CTX_TOKENS = 32768

REQUIRED_KEYS = (
    "id", "category", "domain", "title", "mode", "timeout_s", "scenarios",
    "baseline", "thresholds", "constraints", "deliverables",
    "allowed_paths", "forbidden_paths",
)
BASELINE_KEYS = ("throughput_req_s", "ttft_ms_p90", "tpot_ms", "accuracy")
REQUIRED_DELIVERABLES = ("patch.diff", "metrics.json", "report.md")
REQUIRED_FORBIDDEN = ("tests/", "benchmarks/", "scripts/", "csrc/", "cmake/")
LEAK_TOKENS = (
    "min_pct", "tolerance_pct", "rel_gain", "not_below", "thresholds",
    "reference/", "pick_perf_prs", "baseline/README", "pull/",
)
PR_REF_RE = re.compile(
    r"(?i)(?:\bPR\s*#?\d{2,}\b)|(?:#\d{3,}\b)|(?:github\.com/[\w.-]+/[\w.-]+/pull/)"
)
IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+(docker|requests|urllib3|srv_runtime|srv_grade)\b", re.M
)


# --------------------------------------------------------------------------- #
# pure helpers (the executable form of the scope rule)
# --------------------------------------------------------------------------- #
def norm(path):
    """Repo-relative path with forward slashes and no leading './' or '/'."""
    text = str(path).replace(chr(92), "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def scope_matches(path, scope):
    """True when path is the scope itself or lives underneath it.

    A scope is either an exact file ("vllm_ascend/worker/block_table.py") or a
    directory ("vllm_ascend/ops/fused_moe" / ".../fused_moe/"); the trailing
    slash is optional and a sibling with a shared prefix never matches.
    """
    path = norm(path)
    scope = norm(scope).rstrip("/")
    if not path or not scope:
        return False
    return path == scope or path.startswith(scope + "/")


def scope_overlaps(left, right):
    """True when the two scopes share any path, in either direction."""
    return scope_matches(left, right) or scope_matches(right, left)


def parse_diff_paths(text):
    """Repo-relative paths a unified diff touches (git diff, or bare diff)."""
    paths = set()
    header = re.compile(r"^diff --git a/(\S+) b/(\S+)$")
    for line in text.splitlines():
        match = header.match(line)
        if match:
            paths.add(norm(match.group(1)))
            paths.add(norm(match.group(2)))
            continue
        for prefix, marker in (("--- ", "-"), ("+++ ", "+")):
            if line.startswith(prefix):
                value = line[len(prefix):].strip()
                if value == "/dev/null":
                    break
                if value[:2] in ("a/", "b/"):
                    value = value[2:]
                if value and " " not in value:
                    paths.add(norm(value))
                break
    return sorted(p for p in paths if p)


def check_patch(paths, allowed, forbidden):
    """Violations of the scope rule for one change set (empty list = compliant)."""
    violations = []
    if not paths:
        violations.append("patch touches no files")
    for raw in paths:
        path = norm(raw)
        if not path:
            violations.append("empty path in change set")
            continue
        if ".." in path.split("/"):
            violations.append("%s: path escapes the repository root" % path)
            continue
        hit = [scope for scope in forbidden if scope_matches(path, scope)]
        if hit:
            violations.append("%s: forbidden path (%s)" % (path, hit[0]))
            continue
        if not [scope for scope in allowed if scope_matches(path, scope)]:
            violations.append("%s: outside allowed_paths" % path)
    return violations


def threshold_text(value):
    """One-decimal form of a threshold, so the leak check looks for what a
    reader of task.json would copy."""
    try:
        return "%.1f" % float(value)
    except (TypeError, ValueError):
        return str(value)


def applicable_scenarios(requested):
    """CONTRACT.md section 5: keep the scenarios that fit the model context."""
    kept, skipped = [], {}
    for name in requested:
        if name not in SRV_SCENARIOS:
            skipped[name] = "unknown scenario"
            continue
        input_len, output_len = SRV_SCENARIOS[name]
        if input_len + output_len > MODEL_CTX_TOKENS:
            skipped[name] = "exceeds model context %d" % MODEL_CTX_TOKENS
            continue
        kept.append(name)
    return kept, skipped


# --------------------------------------------------------------------------- #
# checker
# --------------------------------------------------------------------------- #
class Checker:
    def __init__(self, quiet=False):
        self.quiet = quiet
        self.passed = 0
        self.failures = []

    def ok(self, label, condition, detail=""):
        if condition:
            self.passed += 1
            if not self.quiet:
                print("ok   " + label)
            return True
        self.failures.append(label)
        print("FAIL %s%s" % (label, (" -- " + detail) if detail else ""))
        return False

    def equal(self, label, got, want):
        return self.ok(label, got == want, "got %r want %r" % (got, want))


def load_json(path, checker):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        checker.ok("%s is readable" % path.name, False, str(exc))
    except ValueError as exc:
        checker.ok("%s is valid JSON" % path.name, False, str(exc))
    return None


def check_scope_rule(checker, allowed, forbidden):
    """Prove check_patch accepts and rejects the cases it claims to."""
    checker.ok("scope_matches accepts an exact file",
               scope_matches("vllm_ascend/worker/block_table.py",
                             "vllm_ascend/worker/block_table.py"))
    checker.ok("scope_matches accepts a file under a directory scope",
               scope_matches("vllm_ascend/ops/fused_moe/token_dispatcher.py",
                             "vllm_ascend/ops/fused_moe"))
    checker.ok("scope_matches tolerates a trailing slash",
               scope_matches("vllm_ascend/ops/fused_moe/token_dispatcher.py",
                             "vllm_ascend/ops/fused_moe/"))
    checker.ok("scope_matches rejects a shared-prefix sibling",
               not scope_matches("vllm_ascend/ops/fused_moe_x/y.py",
                                 "vllm_ascend/ops/fused_moe"))
    checker.ok("scope_overlaps sees nesting in both directions",
               scope_overlaps("vllm_ascend/ops/fused_moe",
                              "vllm_ascend/ops/fused_moe/moe_mlp.py")
               and not scope_overlaps("vllm_ascend/attention",
                                      "vllm_ascend/ops/fused_moe"))

    sample = "\n".join([
        "diff --git a/vllm_ascend/worker/block_table.py b/vllm_ascend/worker/block_table.py",
        "--- a/vllm_ascend/worker/block_table.py",
        "+++ b/vllm_ascend/worker/block_table.py",
        "@@ -1 +1 @@",
        "-old line",
        "+new line",
    ])
    checker.equal("parse_diff_paths reads a git diff header",
                 parse_diff_paths(sample), ["vllm_ascend/worker/block_table.py"])
    checker.equal("parse_diff_paths drops /dev/null",
                 parse_diff_paths("--- /dev/null\n+++ b/vllm_ascend/new_file.py"),
                 ["vllm_ascend/new_file.py"])

    checker.ok("parse_diff_paths reads a header-only diff",
               parse_diff_paths("diff --git a/vllm_ascend/a.py b/vllm_ascend/a.py")
               == ["vllm_ascend/a.py"])
    checker.ok("PR_REF_RE catches a PR number", bool(PR_REF_RE.search("see PR #1234")))
    checker.ok("PR_REF_RE catches a pull request URL",
               bool(PR_REF_RE.search("https://github.com/o/r/pull/77")))
    checker.ok("PR_REF_RE ignores ordinary numbers",
               not PR_REF_RE.search("budget 2400 s, context 32768, 3 rounds"))
    checker.ok("IMPORT_RE catches a docker import",
               bool(IMPORT_RE.search("import docker\n")))
    checker.ok("IMPORT_RE catches a srv_runtime import",
               bool(IMPORT_RE.search("from srv_runtime import x\n")))

    scope = allowed[0]
    inside = scope + "probe_module.py" if scope.endswith("/") else scope
    checker.ok("an allowed path is accepted",
               not check_patch([inside], allowed, forbidden),
               str(check_patch([inside], allowed, forbidden)))

    tests_probe = "tests/test_scope_probe.py"
    violations = check_patch([tests_probe], allowed, forbidden)
    checker.ok("a path under tests/ is rejected as forbidden",
               bool(violations) and "forbidden" in violations[0], str(violations))

    csrc_probe = "csrc/mc2/probe.cpp"
    violations = check_patch([csrc_probe], allowed, forbidden)
    checker.ok("a path under csrc/ is rejected as forbidden",
               bool(violations) and "forbidden" in violations[0], str(violations))

    outside = "vllm_ascend/platform.py"
    if scope_matches(outside, inside):
        outside = "vllm_ascend/_unrelated/probe.py"
    violations = check_patch([outside], allowed, forbidden)
    checker.ok("a path outside allowed_paths is rejected",
               bool(violations) and "outside allowed_paths" in violations[0],
               str(violations))

    violations = check_patch([inside + "/../../etc/passwd"], allowed, forbidden)
    checker.ok("a path that escapes the repository root is rejected",
               bool(violations) and "escapes" in violations[0], str(violations))

    checker.ok("an empty change set is rejected", bool(check_patch([], allowed, forbidden)))

    if scope.endswith(".py"):
        sibling = scope + ".bak"
        violations = check_patch([sibling], allowed, forbidden)
        checker.ok("a decorated copy of an allowed file is rejected",
                   bool(violations), str(violations))
    else:
        sibling = scope.rstrip("/") + "-sibling/probe.py"
        violations = check_patch([sibling], allowed, forbidden)
        checker.ok("a sibling of an allowed directory is rejected",
                   bool(violations), str(violations))


def check_scenario_rule(checker):
    checker.equal("applicable_scenarios keeps 4k2k", applicable_scenarios(["4k2k"])[0], ["4k2k"])
    checker.equal("applicable_scenarios keeps 16k1k", applicable_scenarios(["16k1k"])[0], ["16k1k"])
    kept, skipped = applicable_scenarios(["64k1k", "128k1k"])
    checker.ok("applicable_scenarios skips both long scenarios",
               kept == [] and set(skipped) == {"64k1k", "128k1k"}, str(skipped))
    checker.ok("the skip reason names the model context",
               "32768" in skipped.get("64k1k", ""), skipped.get("64k1k", ""))


def check_self_imports(checker):
    source = Path(__file__).read_text(encoding="utf-8")
    hits = sorted({m.group(1) for m in IMPORT_RE.finditer(source)})
    checker.ok("verify.py imports no docker/requests/srv_runtime/srv_grade",
               not hits, str(hits))


def check_instruction(checker, spec):
    path = TASK_DIR / "instruction.md"
    if not checker.ok("instruction.md exists", path.is_file()):
        return
    text = path.read_text(encoding="utf-8")
    checker.ok("instruction.md is a real task brief (>= 40 lines)",
               len(text.splitlines()) >= 40, "%d lines" % len(text.splitlines()))
    checker.ok("instruction.md states the 2400 s budget", "2400" in text)
    checker.ok("instruction.md names every deliverable",
               all(item in text for item in REQUIRED_DELIVERABLES),
               str([i for i in REQUIRED_DELIVERABLES if i not in text]))
    checker.ok("instruction.md forbids touching tests and benchmark scripts",
               "不得修改测试与基准脚本" in text)
    raw_allowed = spec.get("allowed_paths") or []
    checker.ok("instruction.md lists every allowed path",
               all(item.strip("/") in text or item in text for item in raw_allowed),
               str([i for i in raw_allowed if i not in text]))
    checker.ok("instruction.md lists every forbidden path",
               all(item in text for item in (spec.get("forbidden_paths") or [])),
               str([i for i in (spec.get("forbidden_paths") or []) if i not in text]))
    scenarios = spec.get("scenarios") or []
    checker.ok("instruction.md names every scenario",
               all(item in text for item in scenarios),
               str([i for i in scenarios if i not in text]))
    leaks = [token for token in LEAK_TOKENS if token in text]
    checker.ok("instruction.md leaks no threshold/reference token", not leaks, str(leaks))
    thresholds = spec.get("thresholds") or {}
    numbers = [threshold_text((thresholds.get("throughput_req_s") or {}).get("min_pct")),
               threshold_text((thresholds.get("accuracy") or {}).get("tolerance_pct"))]
    leaked = [number for number in numbers
              if re.search(r"(?<![\d.])" + re.escape(number) + r"(?![\d.])", text)]
    checker.ok("instruction.md leaks no exact threshold value", not leaked, str(leaked))
    match = PR_REF_RE.search(text)
    checker.ok("instruction.md references no upstream PR", match is None,
               match.group(0) if match else "")


def check_reference(checker):
    path = TASK_DIR / "reference" / "README.md"
    if not checker.ok("reference/README.md exists", path.is_file()):
        return
    text = path.read_text(encoding="utf-8")
    checker.ok("reference/README.md points at pick_perf_prs.py", "pick_perf_prs.py" in text)
    checker.ok("reference/README.md leaves TODO markers", "TODO" in text)
    checker.ok("reference/README.md states the human-confirmation rule",
               "人工确认" in text)


def run_self_check(quiet=False):
    checker = Checker(quiet=quiet)
    spec = load_json(TASK_DIR / "task.json", checker)
    if spec is None:
        print()
        print("FAILED: %d check(s): %s" % (len(checker.failures), ", ".join(checker.failures)))
        return 1
    check_self_imports(checker)
    missing = [key for key in REQUIRED_KEYS if key not in spec]
    checker.ok("task.json has every required key", not missing, "missing %s" % missing)
    checker.equal("id matches the directory name", spec.get("id"), TASK_DIR.name)
    checker.equal("id is a kod- class-3 id", TASK_ID, TASK_DIR.name)
    checker.equal("category is code", spec.get("category"), "code")
    checker.equal("domain equals category", spec.get("domain"), spec.get("category"))
    checker.equal("mode is patch", spec.get("mode"), "patch")
    checker.equal("timeout_s is 2400", spec.get("timeout_s"), 2400)
    checker.ok("title is a non-empty string",
               isinstance(spec.get("title"), str) and bool(spec.get("title").strip()))

    scenarios = spec.get("scenarios") or []
    checker.ok("scenarios is a non-empty list of known keys",
               isinstance(scenarios, list) and bool(scenarios)
               and all(item in SRV_SCENARIOS for item in scenarios), str(scenarios))
    kept, skipped = applicable_scenarios([s for s in scenarios if s in SRV_SCENARIOS])
    checker.ok("every scenario fits the 32768 context", not skipped, str(skipped))
    checker.ok("4k2k is measured", "4k2k" in scenarios, str(scenarios))
    checker.equal("scenarios match the packaged expectation", tuple(scenarios), SCENARIOS)

    baseline = spec.get("baseline") or {}
    checker.equal("baseline has the four metric keys", sorted(baseline), sorted(BASELINE_KEYS))
    checker.ok("baseline is entirely null (not calibrated yet)",
               all(value is None for value in baseline.values()), str(baseline))

    thresholds = spec.get("thresholds") or {}
    throughput = thresholds.get("throughput_req_s") or {}
    checker.equal("throughput threshold kind is rel_gain", throughput.get("kind"), "rel_gain")
    checker.ok("throughput min_pct is a positive number",
               isinstance(throughput.get("min_pct"), (int, float))
               and not isinstance(throughput.get("min_pct"), bool)
               and throughput.get("min_pct") > 0, str(throughput))
    accuracy = thresholds.get("accuracy") or {}
    checker.equal("accuracy threshold kind is not_below", accuracy.get("kind"), "not_below")
    checker.equal("accuracy tolerance_pct is the contract 0.5",
                 accuracy.get("tolerance_pct"), 0.5)

    deliverables = spec.get("deliverables") or []
    checker.ok("deliverables cover the three required artefacts",
               all(item in deliverables for item in REQUIRED_DELIVERABLES), str(deliverables))

    constraints = spec.get("constraints") or []
    checker.ok("constraints is a non-empty list of strings",
               isinstance(constraints, list) and bool(constraints)
               and all(isinstance(item, str) and item.strip() for item in constraints))
    checker.ok("constraints forbid touching tests and benchmark scripts",
               any("测试" in item and "基准" in item for item in
                   [c for c in constraints if isinstance(c, str)]))

    raw_allowed = spec.get("allowed_paths") or []
    raw_forbidden = spec.get("forbidden_paths") or []
    allowed = [norm(item) for item in raw_allowed]
    forbidden = [norm(item) for item in raw_forbidden]
    checker.ok("allowed_paths is a non-empty list", bool(allowed), str(raw_allowed))
    checker.ok("allowed_paths entries are already normalised",
               all(item == norm(item) for item in raw_allowed), str(raw_allowed))
    checker.ok("allowed_paths stay inside the source tree",
               all(item.startswith("vllm_ascend/") for item in allowed), str(allowed))
    checker.ok("allowed_paths has no duplicates", len(set(allowed)) == len(allowed))
    checker.ok("forbidden_paths is a non-empty list", bool(forbidden), str(raw_forbidden))
    checker.ok("forbidden_paths cover the graded protected areas",
               all(item in forbidden for item in REQUIRED_FORBIDDEN),
               str([i for i in REQUIRED_FORBIDDEN if i not in forbidden]))
    overlaps = [(left, right) for left in allowed for right in forbidden
                if scope_overlaps(left, right)]
    checker.ok("allowed_paths and forbidden_paths do not overlap", not overlaps,
               str(overlaps[:1]))

    check_scope_rule(checker, allowed, forbidden)
    check_scenario_rule(checker)
    check_instruction(checker, spec)
    check_reference(checker)

    print()
    if checker.failures:
        print("FAILED: %d check(s) failed for %s: %s"
              % (len(checker.failures), TASK_ID, ", ".join(checker.failures)))
        return 1
    print("OK: %d checks passed for task %s" % (checker.passed, TASK_ID))
    return 0


def scopes_from_spec():
    spec = json.loads((TASK_DIR / "task.json").read_text(encoding="utf-8"))
    allowed = [norm(item) for item in spec.get("allowed_paths") or []]
    forbidden = [norm(item) for item in spec.get("forbidden_paths") or []]
    return allowed, forbidden


def report_violations(paths, allowed, forbidden, quiet=False):
    violations = check_patch(paths, allowed, forbidden)
    if not quiet:
        print("changed files (%d):" % len(paths))
        for item in paths:
            print("  " + item)
    if violations:
        print("VIOLATIONS (%d):" % len(violations))
        for item in violations:
            print("  " + item)
        return 1
    print("OK: every changed file stays inside allowed_paths and avoids forbidden_paths")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="offline self-check for task %s" % TASK_ID)
    parser.add_argument("--patch", default=None,
                        help="unified diff file to check against the scope rule")
    parser.add_argument("--changed", nargs="*", default=None,
                        help="repo-relative paths to check against the scope rule")
    parser.add_argument("--quiet", action="store_true",
                        help="only print failures and the summary")
    args = parser.parse_args(argv)
    if args.patch or args.changed is not None:
        allowed, forbidden = scopes_from_spec()
        if args.patch:
            text = Path(args.patch).read_text(encoding="utf-8")
            return report_violations(parse_diff_paths(text), allowed, forbidden, args.quiet)
        return report_violations([norm(item) for item in args.changed], allowed, forbidden,
                                 args.quiet)
    return run_self_check(args.quiet)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main())
