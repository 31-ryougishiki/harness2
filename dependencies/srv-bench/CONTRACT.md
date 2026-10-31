# SRV-Bench 接口契约 (v1)

本文件是并行开发的唯一依据。任何实现偏离本契约即视为缺陷。

## 0. 目的

给 harness2 新增一个 benchmark: `srv`，用于测量并递归提升 agent 的**推理服务调优能力**。
三类任务: `config`(配置寻优) / `measure`(aisbench 调测) / `code`(代码修改)。

**边界**: 不改 `harness2/core/` 的方法逻辑(recursion/prompts/selection/library)。
只做 benchmark 接缝(adapter)与任务集。唯一例外是 `core/report.py` 增加 srv 的报告口径分支。

## 1. 不可变锚点

| 项 | 值 |
|---|---|
| 模型 | `Qwen/Qwen3-8B` |
| 原生上下文 | 32768 tokens（**不启 YaRN**） |
| 镜像 | vllm-ascend 0.23.0（由 `SRV_IMAGE` 环境变量注入，不写死 tag） |
| 硬件 | Ascend A2/A3 |
| 场景 | `4k2k`=(4096,2048) `16k1k`=(16384,1024) 可测；`64k1k`=(65536,1024) `128k1k`=(131072,1024) **超上下文 → skipped** |
| PD 拓扑 | 同机 1P1D，同一容器内两个 vLLM 实例 |
| 精度数据集 | `gsm8k_gen_0_shot_cot_chat_prompt.py`（容差 ±0.5%） |

`scenario -> (input_len, output_len)` 由 `srv_runtime.SRV_SCENARIOS` 单点定义，别处不得重复硬编码。

## 2. 目录布局（root = `/d/code/free/harness2`）

```
harness2/benchmarks/srv.py            # SPEC + SrvAdapter            [D]
harness2/benchmarks/srv_runtime.py    # 容器/服务/aisbench 执行器      [A]
harness2/benchmarks/srv_grade.py      # 判分，被 srv.py 调用           [A]
harness2/benchmarks/tasks.py          # + discover('srv')             [D]
harness2/cli/_common.py               # + 工厂                        [D]
harness2/preflight.py                 # + srv 检查                    [D]
harness2/core/report.py               # + srv 报告口径                 [D]
harness2/install.py                   # + setup srv                   [D]
dependencies/srv-bench/
  CONTRACT.md                          # 本文件
  README.md                            # 标定与使用说明                 [D]
  tasks/<category>/<task_id>/task.json
                             /instruction.md
                             /verify.py
                             /reference/            # judge 侧可见，绝不下发
  baseline/README.md                   # 标定流程                       [B]
scripts/pick_perf_prs.py               # 类3 PR 挑选                    [C]
docs/SRV_BENCH.md                      # 用户文档                       [D]
```

`<category>` ∈ `config` | `measure` | `code`。`<task_id>` 前缀与目录一致: `cfg-*` / `msr-*` / `kod-*`。
`task_id` = `<category>/<dirname>`（由 `discover()` 拼出，与 lab/jb 一致）。

## 3. task.json schema（严格，`verify.py` 会校验）

```json
{
  "id": "cfg-tp-choice",
  "category": "config",
  "domain": "config",
  "title": "一句话任务名",
  "mode": "single",
  "timeout_s": 2400,
  "scenarios": ["4k2k", "16k1k"],
  "baseline": {
    "throughput_req_s": null,
    "ttft_ms_p90": null,
    "tpot_ms": null,
    "accuracy": null
  },
  "thresholds": {
    "throughput_req_s": {"kind": "rel_gain", "min_pct": 3.0},
    "accuracy": {"kind": "not_below", "tolerance_pct": 0.5}
  },
  "constraints": ["不得修改模型权重", "..."],
  "deliverables": ["metrics.json", "run.sh", "report.md"]
}
```

- `mode`: `single`(单实例) | `pd`(1P1D) | `patch`(改代码后重测)。
- `domain` 恒等于 `category`（library 分组单位）。
- `baseline` 允许全 `null`：表示**尚未标定**。此时该任务在 judge 侧记为 unscored（不是 0），
  并在报告里单列 "needs calibration"。`baseline/README.md` 说明如何填实测值。
- `thresholds.kind` ∈ `rel_gain`(min_pct) | `not_below`(tolerance_pct) | `budget`(max_s)。

## 4. metrics.json schema（运行时产物，唯一判分输入）

```json
{
  "task_id": "cfg-tp-choice",
  "ok": true,
  "error": "",
  "image": "vllm-ascend:0.23.0-a2",
  "model": "Qwen/Qwen3-8B",
  "service": {
    "mode": "single",
    "p_cmd": "...",
    "d_cmd": null,
    "prefix_cache": {"p": "100%", "d": "off"}
  },
  "scenarios": {
    "4k2k": {
      "skipped": false,
      "reason": "",
      "throughput_req_s": 1.23,
      "ttft_ms_p90": 456.7,
      "tpot_ms": 12.3,
      "accuracy": null,
      "aisbench_cmd": "ais_bench --models ...",
      "raw": "artifacts/aisbench_4k2k.json"
    },
    "64k1k": {"skipped": true, "reason": "exceeds model context 32768"}
  }
}
```

- `raw` 为相对 run_dir 的路径，缺失即该场景无效。
- `ok=false` 或 `scenarios` 全 `skipped` → judge 记 **unscored（None）**，绝不记 0。
- 服务没起来 / aisbench 异常退出 → `ok=false` + `error` 写清原因，**不伪造数值**。

## 5. `srv_runtime.py` API（A 实现，B/C/D 只调用）

```python
SRV_SCENARIOS: dict[str, tuple[int, int]] = {
    "4k2k": (4096, 2048), "16k1k": (16384, 1024),
    "64k1k": (65536, 1024), "128k1k": (131072, 1024),
}
MODEL_CTX_TOKENS: int = 32768

def applicable_scenarios(requested: list[str]) -> tuple[list[str], dict[str, str]]:
    """-> (可测场景列表, {被跳过场景: 原因})。input_len + output_len > MODEL_CTX_TOKENS 即跳过。"""

def container_start(run_id: str, *, image: str) -> str:
    """幂等创建并启动容器, 返回容器名。device 透传 /dev/davinci* 与 /dev/davinci_manager。"""

def container_stop(name: str) -> None: ...

def service_up(mode: str, scenario: str, *, container: str, cfg: dict, log_dir: Path) -> dict:
    """拉起服务并等到 /health 200。
    -> {"ok": bool, "p_cmd": str, "d_cmd": str|None, "prefix_cache": {"p": str, "d": str},
        "error": str, "log": str}
    mode=='pd' 时 P 开 prefix cache 100%、D 关闭；mode=='single' 时只起一个实例。"""

def aisbench_perf(scenario: str, *, container: str, output_len: int | None = None) -> dict:
    """跑 perf 模式。output_len 覆盖用于纯P(output_len=1)。
    -> {"ok": bool, "throughput_req_s": float|None, "ttft_ms_p90": float|None,
        "tpot_ms": float|None, "cmd": str, "raw": str, "error": str}"""

def aisbench_accuracy(*, container: str, dataset: str = "gsm8k_gen_0_shot_cot_chat_prompt.py") -> dict:
    """-> {"ok": bool, "accuracy": float|None, "cmd": str, "raw": str, "error": str}"""

def collect(container: str, task_dir: Path, run_dir: Path, spec: dict) -> dict:
    """按 task spec 驱动完整测量, 落 metrics.json 到 run_dir, 返回该 dict。"""
```

A 只用标准库 + `subprocess`，不引入新依赖。

## 6. `srv_grade.py` 判分契约（A 实现）

```python
def grade(metrics: dict, spec: dict) -> dict | None:
    """-> {"score": float, "passed": bool, "detail": {...}} 或 None(unscored)
    纯函数, 无 IO, 便于单测。"""
```

判分口径:

| category | 判分 |
|---|---|
| `config` | 每个 `rel_gain` 指标: `(实测 - baseline)/baseline * 100 >= min_pct`；`accuracy` 若启用则须 `>= baseline_acc - tolerance`。全过 = passed |
| `code` | 同 `config`，另加约束: patch 必须落在允许路径、不得改动测试/基准脚本（违者 0 分并记 `violation`） |
| `measure` | **不比数字**。逐项检查: ①服务确实起来了（`service.ok`）②实际执行的 `aisbench_cmd` 含正确 dataset/mode/model ③报告可解析 ④`scenarios` 覆盖了要求集合，skipped 必须带非空 reason ⑤`deliverables` 齐全。每项等权，全过才 passed |

`score` 为 0..1。unscored 一律返回 `None`，调用方不得当作 0 缓存。

## 7. 禁止事项（对每个子代理都成立）

1. **绝不下发** `reference/`、`baseline.json` 的具体数值、`thresholds`、PR patch 给 agent。`task_prompt()` 只返回 `instruction.md` 正文。
2. **不伪造测量数字**。拿不到就是 `ok=false`。任何 `baseline` 未标定就写 `null`。
3. **不改** `harness2/core/` 除 `report.py` 加入口以外的地方。
4. 不引入新第三方依赖。Windows 上无法执行的路径必须能被 `--dry-run` 静态校验（不 import docker/requests）。
5. 单元可测: `grade()` 与 `applicable_scenarios()` 必须是纯函数，B/C 的 `verify.py` 用它们做离线自测。

## 8. 仲裁记录（v1.1，B 交付后裁定；A/C/D 必须遵守）

**D1 measure 类 thresholds = {} 是合法的。**
`grade()` 对 `category=="measure"` 完全忽略 thresholds（契约 §6 已定"不比数字"）。
`srv.py`/verify 均**不得**要求 measure 任务的 thresholds 非空。

**D3 accuracy 单位 = 0~100 百分数。**
`accuracy: 82.5` 表示 82.5%。`tolerance_pct: 0.5` 表示 **0.5 个百分点**。
`srv_grade.grade()` 必须按百分点比较：`measured >= baseline_acc - tolerance_pct`，双方都不做 /100 归一。

**D4 baseline 三种形态 + 逐场景判定。**
```jsonc
"baseline": {"throughput_req_s": null}                           // 未标定 -> unscored
"baseline": {"throughput_req_s": 1.23}                           // 单值: 所有被比对场景共用
"baseline": {"throughput_req_s": {"4k2k": 1.23, "16k1k": 0.41}}  // 按场景取值(正式标定推荐)
```
- 判分**逐场景**比对；一个"可比对场景" = 该场景未 skip 且其 baseline 值非 null。
- **所有可比对场景都达标才算 passed**（不允许用 4k2k 的增益掩盖 16k1k 的退化）。
- 可比对场景数为 0 → 返回 `None`（unscored），不是 0。
- `detail` 必须逐场景列出 实测/baseline/增益百分比/是否达标。

**D5 msr-accuracy 跑单实例（mode=single）。** 精度与拓扑无关，PD 拓扑只会引入额外不确定性。

**D6 instruction.md 禁用词表。** 下发文本中不得出现： rubric / 评分 / 评测 / 判分 / 打分 / 得分 / score /
threshold / 阈值 / 容差 / tolerance / min_pct / rel_gain / 基线 / baseline 数值 / reference / needs calibration /
验收口径 / 不公开。描述"工作"用中性表达：如"服务已按参考配置就绪""服务拓扑由 harness 固定""该场景吞吐相对参考配置有提升"。

## 9. 仲裁记录 v1.2（C 交付后裁定；A/D 必须遵守）

**背景**: C 读了 A 已落地的 srv_grade.py，发现判分端读的键在 CONTRACT §4 里不存在；同时路径约束拼写不一致导致约束静默失效。
两条都是真缺陷，不是文档问题。

**D7 路径约束: `forbidden_paths` 与 `protected_paths` 等价。**
`grade()` 必须读两者**并集**（任一拼写单独出现都要生效）。当前 srv_grade.py:326 只读 `protected_paths`，
导致 C 的 3 个 kod-* 任务上「不得改动 tests/ 与 benchmarks/」于判分端完全不生效 —— 这是必须修的静默失效。

**D8 `deliverables` 与 `changed_files` 由运行时填充，不是 agent 自报。**
- `collect()` 必须在 metrics.json 写入:
  - `deliverables`: `{文件名: bool}`，**通过扫描 run_dir 得到**（agent 在 metrics.json 里自报的不可信）。
  - `changed_files`: 字符串数组，从 agent 实际产出的 patch / git diff 提取。
- `grade()` 保持纯函数，只读这两个键:
  - `deliverables` 键**缺失** → 返回 `None`(unscored)，**不得**判成 "agent 没交付"（那是运行时缺件，不是 agent 的错）。
  - `changed_files` 缺失且该任务声明了路径约束 → `None`(unscored)，**不得**记 violation。

**D9 metrics.json 键集合 v1.2（取代 §4 的同名字段清单）**
必填: `task_id` `ok` `error` `image` `model` `service` `deliverables` `scenarios`
条件: `changed_files`（仅 category=="code"）
- config/measure 的 instruction.md **无需**列出 `deliverables`/`changed_files`（运行时填）。
- code 类可在模板里列出，作为提示，但判分不依赖 agent 自报。

**任务边界说明**: C 已按旧 §4 在 kod-* 的 instruction.md 里补了这两个字段，属于无害的提示，不要求回退。

## 10. 仲裁记录 v1.3（A 交付后裁定）

**D10 rel_gain 必须区分指标方向（采纳 A 的实现）。**
§6 的字面公式 `(实测-baseline)/baseline*100 >= min_pct` 只对"越大越好"成立。对 `ttft_ms_p90` / `tpot_ms` 这类延迟指标，
变好会算出负增益而永不可通过。裁定:
- 方向默认由指标名推断: throughput* = higher_is_better；*_ms* / latency* = lower_is_better。
- 增益一律归一到"正数=变好": higher 用 `(实测-baseline)/baseline`，lower 用 `(baseline-实测)/baseline`。
- task.json 的 thresholds 可用 `"direction"` 显式覆盖。
- `detail` 必须写出所用公式与 direction，便于评审复核。

**D11 §8 D3 的公式不变；我给 A 的 81.9 用例是错的。**
`measured >= baseline_acc - tolerance_pct`，baseline 82.0 / tolerance 0.5 → floor 81.5。
故 81.9 通过是正确行为，81.4 不通过、81.5 恰好达标。以 A 的用例为准。

**D12 substrate 启动归 adapter，srv_runtime 不提供 run_agent。**
既有惯例见 `lab.py:_launch_oc/_launch_cx`: adapter 自己构造并执行 substrate 命令（`run_detached`）。
D 在 srv.py:373 要求 `srv_runtime.run_agent(payload)` 是多设计的一层间接，去掉它:
`SrvAdapter._launch` 直接按 lab 惯例起 substrate，runtime 只负责容器/服务/aisbench/collect 五件事。
`srv_runtime` 的 API 面回到 §5 原样（不含 run_agent）。

**D13 msr-accuracy 必须能被机器识别为"要跑精度"。**
A 指出: `collect()` 只认 spec 的 `accuracy`/`require_accuracy`/`dataset`/`accuracy_dataset`，而 B 的
`msr-accuracy/task.json` 是 `thresholds={}` 且无 accuracy 键 → 与 `msr-pure-p` 在规格层无法区分，精度跑不会触发。
裁定: 在 `msr-accuracy/task.json` 加 `"accuracy": true`（B 负责改，A 不改 B 的文件）。
