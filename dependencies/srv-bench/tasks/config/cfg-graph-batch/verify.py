#!/usr/bin/env python3
"""srv-bench 任务包离线自检 (只用标准库, 不 import srv_runtime / srv_grade)。

校验项: task.json 合法性与契约 schema、id 与目录名一致、scenarios 都在
SRV_SCENARIOS 内、thresholds 结构、baseline 形态(契约 D4)、instruction.md 不含
D6 禁用词且写明时间预算与交付物、reference/README.md 存在。

用法:  python verify.py [--quiet] [task_dir]      (task_dir 省略时用本脚本所在目录)
返回:  0 全部通过 / 1 存在失败项
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

_ = (ast, json, re, sys, Path)  # 显式保留全部导入, 便于静态检查

USAGE = "用法: python verify.py [--quiet] [task_dir]   (task_dir 省略时用本脚本所在目录)"

CATEGORIES = ("config", "measure", "code")
ID_PREFIX = {"config": "cfg-", "measure": "msr-", "code": "kod-"}
MODES = ("single", "pd", "patch")
REQUIRED_KEYS = (
    "id", "category", "domain", "title", "mode", "timeout_s",
    "scenarios", "baseline", "thresholds", "constraints", "deliverables",
)
# CONTRACT 10/D13: collect() 用这几个键判断是否要跑 gsm8k 精度。
OPTIONAL_KEYS = ("accuracy", "require_accuracy", "dataset", "accuracy_dataset")
BASELINE_KEYS = ("throughput_req_s", "ttft_ms_p90", "tpot_ms", "accuracy")
METRIC_KEYS = BASELINE_KEYS
KIND_FIELD = {"rel_gain": "min_pct", "not_below": "tolerance_pct", "budget": "max_s"}

# SRV_SCENARIOS / MODEL_CTX_TOKENS 单点定义在 srv_runtime.py; 这里优先从源码解析,
# 解析不到 (文件还没落地) 才回退到下面的镜像。镜像仅用于自检, 不参与运行时。
FALLBACK_SCENARIOS = {
    "4k2k": (4096, 2048),
    "16k1k": (16384, 1024),
    "64k1k": (65536, 1024),
    "128k1k": (131072, 1024),
}
FALLBACK_MODEL_CTX = 32768

# 下发给被测 agent 的文本里绝不允许出现的字样 (CONTRACT 8/D6)。
ASCII_BANNED = (
    "rubric", "score", "threshold", "baseline", "tolerance", "reference",
    "judge", "grader", "golden", "min_pct", "rel_gain", "needs calibration",
)
CJK_BANNED = (
    "评分", "评测", "判分", "打分", "得分", "(?<!百)分数", "基线", "阈值", "容差",
    "验收口径", "不公开", "参考配置",
)


class Checker:
    """收集 (ok, 名字, 细节) 三元组, 最后统一打印。"""

    def __init__(self):
        self.items = []
        self.notes = []

    def add(self, ok, name, detail=""):
        self.items.append((bool(ok), name, detail))
        return bool(ok)

    def note(self, text):
        self.notes.append(text)

    def failures(self):
        return [it for it in self.items if not it[0]]


def _banned_hits(text):
    hits = []
    lowered = text.lower()
    for word in ASCII_BANNED:
        pattern = r"(?<![0-9a-z_])" + re.escape(word) + r"(?![0-9a-z_])"
        if re.search(pattern, lowered):
            hits.append(word)
    for pattern in CJK_BANNED:
        if re.search(pattern, text):
            hits.append(pattern)
    return hits


def _repo_root(start):
    for cand in (start, *start.parents):
        if (cand / "harness2" / "benchmarks").is_dir():
            return cand
    return None


def _parse_runtime(src):
    try:
        tree = ast.parse(src.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    scenarios, ctx = None, None
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, value_node = node.target.id, node.value
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name)):
            name, value_node = node.targets[0].id, node.value
        else:
            continue
        if name not in ("SRV_SCENARIOS", "MODEL_CTX_TOKENS") or value_node is None:
            continue
        try:
            value = ast.literal_eval(value_node)
        except (ValueError, SyntaxError):
            continue
        if name == "SRV_SCENARIOS" and isinstance(value, dict):
            norm = {}
            for key, pair in value.items():
                if (isinstance(key, str) and isinstance(pair, tuple) and len(pair) == 2
                        and all(isinstance(n, int) and not isinstance(n, bool) for n in pair)):
                    norm[key] = (pair[0], pair[1])
            if norm:
                scenarios = norm
        if name == "MODEL_CTX_TOKENS" and isinstance(value, int) and not isinstance(value, bool):
            ctx = value
    if scenarios is None:
        return None
    return scenarios, (FALLBACK_MODEL_CTX if ctx is None else ctx)


def _runtime_consts(task_dir):
    root = _repo_root(task_dir)
    if root is not None:
        src = root / "harness2" / "benchmarks" / "srv_runtime.py"
        if src.is_file():
            parsed = _parse_runtime(src)
            if parsed is not None:
                return parsed[0], parsed[1], str(src)
            return dict(FALLBACK_SCENARIOS), FALLBACK_MODEL_CTX, "内置镜像 (srv_runtime.py 无可解析的 SRV_SCENARIOS)"
    return dict(FALLBACK_SCENARIOS), FALLBACK_MODEL_CTX, "内置镜像 (未找到 srv_runtime.py)"


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _report(checker, quiet=False):
    for ok, name, detail in checker.items:
        if quiet and ok:
            continue
        line = "[PASS] " + name if ok else "[FAIL] " + name
        print(line + (" — " + detail if detail else ""))
    for text in checker.notes:
        print("[INFO] " + text)
    failures = checker.failures()
    print("结果: " + str(len(checker.items) - len(failures)) + " 通过 / " + str(len(failures)) + " 失败")
    return 0 if not failures else 1


def _check_optional_keys(checker, spec):
    problems = []
    for key in ("accuracy", "require_accuracy"):
        if key in spec and not isinstance(spec[key], bool):
            problems.append(key + ": 必须是 true/false, 实际 " + repr(spec[key]))
    for key in ("dataset", "accuracy_dataset"):
        if key in spec and not (isinstance(spec[key], str) and spec[key].strip()):
            problems.append(key + ": 必须是非空字符串, 实际 " + repr(spec[key]))
    checker.add(not problems, "可选精度触发键类型正确 (D13)", "; ".join(problems))


def _check_baseline(checker, baseline, scenarios):
    if not isinstance(baseline, dict):
        checker.add(False, "baseline 是对象", repr(baseline))
        return
    keys = set(baseline)
    missing = sorted(set(BASELINE_KEYS) - keys)
    unknown = sorted(keys - set(BASELINE_KEYS))
    checker.add(not missing and not unknown, "baseline 键集合与契约一致",
                "缺 " + str(missing) + " 多 " + str(unknown))
    known_scenarios = set(scenarios) if isinstance(scenarios, list) else set()
    problems = []
    for key, value in baseline.items():
        if value is None or _is_number(value):
            continue
        if isinstance(value, dict) and value and all(
                isinstance(s, str) and _is_number(n) for s, n in value.items()):
            stray = sorted(set(value) - known_scenarios)
            if stray:
                problems.append(key + ": 场景键不在 task.scenarios 内 " + str(stray))
            continue
        problems.append(key + ": 取值须为 null / 数字 / 逐场景数字对象, 实际 " + repr(value))
    checker.add(not problems, "baseline 形态合法 (契约 D4)", "; ".join(problems))
    nulls = [k for k in BASELINE_KEYS if baseline.get(k) is None]
    checker.add(len(nulls) in (0, len(BASELINE_KEYS)), "baseline 要么全 null 要么全填",
                "仍为 null: " + str(nulls))
    if len(nulls) == len(BASELINE_KEYS):
        checker.note("baseline 全为 null → needs calibration: 该任务尚未标定, judge 侧记 unscored (不是 0)")
    elif not nulls:
        checker.note("baseline 已标定: 判分按 thresholds 计算")


def _check_thresholds(checker, thresholds, category):
    if not isinstance(thresholds, dict):
        checker.add(False, "thresholds 是对象", repr(thresholds))
        return
    if category == "measure":
        checker.add(True, "measure 类不比数字: thresholds 允许为空 (契约 D1)",
                    "keys=" + str(sorted(thresholds)))
    else:
        checker.add(bool(thresholds), "非 measure 类 thresholds 非空", "keys=" + str(sorted(thresholds)))
    problems = []
    for metric, rule in thresholds.items():
        if metric not in METRIC_KEYS:
            problems.append(metric + ": 不是 metrics.json 指标名")
            continue
        if not isinstance(rule, dict):
            problems.append(metric + ": 规则不是对象")
            continue
        kind = rule.get("kind")
        if kind not in KIND_FIELD:
            problems.append(metric + ": kind=" + repr(kind) + " 非法")
            continue
        field = KIND_FIELD[kind]
        if not _is_number(rule.get(field)):
            problems.append(metric + ": kind=" + kind + " 缺少数字字段 " + field)
        extra = sorted(set(rule) - {"kind", field, "direction"})
        if extra:
            problems.append(metric + ": 多余字段 " + str(extra))
        if "direction" in rule and rule["direction"] not in ("higher_is_better", "lower_is_better"):
            problems.append(metric + ": direction 非法 " + repr(rule["direction"]))
    checker.add(not problems, "thresholds 结构与契约一致", "; ".join(problems))


def _check_instruction(checker, task_dir, timeout_s):
    path = task_dir / "instruction.md"
    if not checker.add(path.is_file(), "instruction.md 存在", str(path)):
        return
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        checker.add(False, "instruction.md 可读为 UTF-8", str(exc))
        return
    checker.add(bool(text.strip()), "instruction.md 非空")
    hits = _banned_hits(text)
    checker.add(not hits, "instruction.md 不含 D6 禁用词", "命中: " + ", ".join(hits) if hits else "")
    budget = str(timeout_s) if isinstance(timeout_s, int) and not isinstance(timeout_s, bool) else ""
    checker.add(bool(budget) and budget in text,
                "instruction.md 写明与 timeout_s 一致的时间预算", "需出现 " + repr(budget))
    for name in ("metrics.json", "run.sh", "report.md"):
        checker.add(name in text, "instruction.md 写明交付物 " + name)


def _check_reference(checker, task_dir):
    path = task_dir / "reference" / "README.md"
    if not checker.add(path.is_file(), "reference/README.md 存在 (judge 侧)", str(path)):
        return
    try:
        body = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        checker.add(False, "reference/README.md 可读为 UTF-8", str(exc))
        return
    checker.add(len(body.strip()) > 200, "reference/README.md 有实质内容",
                "字符数 " + str(len(body.strip())))


def _parse_argv(argv):
    """-> (target|None|"help", quiet, bad)。无 target 时用脚本所在目录。"""
    quiet = False
    target = None
    for arg in argv:
        if arg in ("--quiet", "-q"):
            quiet = True
        elif arg in ("--help", "-h"):
            return "help", quiet, False
        elif not arg.startswith("-") and target is None:
            target = arg
        else:
            return None, quiet, True
    return target, quiet, False


def main(argv):
    target, quiet, bad = _parse_argv(argv)
    if bad:
        print(USAGE)
        return 1
    if target == "help":
        print(USAGE)
        return 0
    task_dir = Path(target).expanduser() if target else Path(__file__).resolve().parent
    task_dir = task_dir.resolve()
    checker = Checker()
    if not checker.add(task_dir.is_dir(), "任务目录存在", str(task_dir)):
        return _report(checker, quiet)

    task_path = task_dir / "task.json"
    spec = None
    if checker.add(task_path.is_file(), "task.json 存在", str(task_path)):
        try:
            spec = json.loads(task_path.read_text(encoding="utf-8"))
            checker.add(True, "task.json 可被 json 解析")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            checker.add(False, "task.json 可被 json 解析", str(exc))
    if not checker.add(isinstance(spec, dict), "task.json 顶层为对象"):
        return _report(checker, quiet)

    missing = [k for k in REQUIRED_KEYS if k not in spec]
    checker.add(not missing, "必填键齐全",
                "缺 " + str(missing) if missing else str(len(REQUIRED_KEYS)) + " 个键")
    extra = sorted(set(spec) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS))
    checker.add(not extra, "无 schema 外未知键", "多 " + str(extra) if extra else "")
    _check_optional_keys(checker, spec)

    category = spec.get("category")
    checker.add(category in CATEGORIES, "category 合法", repr(category))
    checker.add(spec.get("domain") == category, "domain 恒等于 category",
                repr(spec.get("domain")) + " vs " + repr(category))
    checker.add(task_dir.parent.name == category, "category 与上级目录名一致",
                repr(task_dir.parent.name) + " vs " + repr(category))

    task_id = spec.get("id")
    prefix = ID_PREFIX.get(category, "")
    checker.add(isinstance(task_id, str) and bool(prefix) and task_id.startswith(prefix),
                "id 前缀与 category 一致", repr(task_id) + " 需以 " + repr(prefix) + " 开头")
    checker.add(task_id == task_dir.name, "id 与目录名一致",
                repr(task_id) + " vs " + repr(task_dir.name))

    title = spec.get("title")
    checker.add(isinstance(title, str) and bool(title.strip()), "title 为非空字符串", repr(title))

    mode = spec.get("mode")
    checker.add(mode in MODES, "mode 合法", repr(mode))
    if category == "config":
        checker.add(mode == "single", "config 类 mode=single", repr(mode))
    if category == "measure":
        checker.add(mode in ("single", "pd"), "measure 类 mode 为 single/pd", repr(mode))

    timeout_s = spec.get("timeout_s")
    checker.add(isinstance(timeout_s, int) and not isinstance(timeout_s, bool) and timeout_s >= 60,
                "timeout_s 为正整数且不小于 60", repr(timeout_s))

    scenarios = spec.get("scenarios")
    sc_ok = checker.add(
        isinstance(scenarios, list) and bool(scenarios) and all(isinstance(s, str) for s in scenarios),
        "scenarios 为非空字符串列表", repr(scenarios))
    if sc_ok:
        checker.add(len(set(scenarios)) == len(scenarios), "scenarios 无重复", repr(scenarios))
        known, ctx_tokens, source = _runtime_consts(task_dir)
        unknown = [s for s in scenarios if s not in known]
        checker.add(not unknown, "scenarios 都在 SRV_SCENARIOS 内",
                    "未知 " + str(unknown) + "; 来源 " + source)
        measurable = [s for s in scenarios if s in known and sum(known[s]) <= ctx_tokens]
        checker.add(bool(measurable), "至少一个场景不超过模型上下文",
                    "可测 " + str(measurable) + "; 上下文 " + str(ctx_tokens))

    _check_baseline(checker, spec.get("baseline"), scenarios)
    _check_thresholds(checker, spec.get("thresholds"), category)

    constraints = spec.get("constraints")
    checker.add(
        isinstance(constraints, list) and bool(constraints)
        and all(isinstance(x, str) and bool(x.strip()) for x in constraints),
        "constraints 为非空字符串列表", repr(constraints))

    deliverables = spec.get("deliverables")
    if checker.add(
            isinstance(deliverables, list) and bool(deliverables)
            and all(isinstance(x, str) for x in deliverables),
            "deliverables 为非空字符串列表", repr(deliverables)):
        checker.add("metrics.json" in deliverables, "deliverables 含 metrics.json")

    _check_instruction(checker, task_dir, timeout_s)
    _check_reference(checker, task_dir)
    return _report(checker, quiet)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main(sys.argv[1:]))

