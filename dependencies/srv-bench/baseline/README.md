# baseline 标定说明（srv-bench）

本目录是 `dependencies/srv-bench/tasks/*/*/task.json` 里 `baseline` 字段的标定说明。

## 1. 为什么现在全是 null

仓库里**没有** NPU、没有 docker、没有 ais_bench，任何写死的吞吐/延迟/精度数字都只能是伪造的。
因此 6 个任务的 `baseline` 一律留 `null`，语义是 **needs calibration（尚未标定）**：

- `srv_grade.grade()` 在 baseline 全为 `null` 时返回 `None`（unscored），**不是 0 分**，调用方不得把 None 当 0 缓存；
- 报告里要单列一行 needs calibration，说明该任务在当前机器上还没跑过标定；
- `thresholds` 与 baseline **互不替代**：thresholds 是验收门槛（已写全），baseline 是参照点（待填）。

## 2. 四个字段的单位与口径

| 字段 | 含义 | 单位 | 来源 |
|---|---|---|---|
| `throughput_req_s` | 每秒完成请求数 | req/s | `metrics.json` 的 `scenarios["4k2k"].throughput_req_s` |
| `ttft_ms_p90` | 首 token 时延 P90 | ms | 同上，取 `ttft_ms_p90` |
| `tpot_ms` | 每输出 token 平均耗时 | ms | 同上，取 `tpot_ms` |
| `accuracy` | gsm8k 准确率 | 0~100 的百分数 | 同上，取 `accuracy` |

baseline 的每个字段可以是**单值**（所有被比对场景共用）或**逐场景对象**（契约 8/D4 推荐），见下文 §3.5。
注意 `accuracy` 统一用 **百分数（0~100）**，所以 `thresholds.accuracy.tolerance_pct = 0.5` 表示 0.5 个百分点，两者必须同单位。
如果 `scenarios["4k2k"]` 是 `skipped: true`（超上下文被跳过），该场景不产生 baseline，不要用别的场景的代填。

## 3. 标定步骤

1. **固定环境**：目标机型（Ascend A2/A3 8 卡）、镜像 tag（由 `SRV_IMAGE` 注入）、模型 `Qwen/Qwen3-8B`、`MODEL_CTX_TOKENS=32768` 不启 YaRN。基线换机器就要重标，不能跨机型复用。
2. **选参考配置**：每个任务用哪套配置当参照点，写在它自己的 `reference/README.md` 的「标定注意」一节。
   - config 类的参照点**必须是非最优配置**（例如 `cfg-tp-choice` 用出厂默认 TP=8），否则 `rel_gain` 无从谈起；
   - measure 类不参与 `rel_gain` 判分，`baseline` 保持全 `null`，不要填。
3. **跑一次完整测量**：用 `srv_runtime.collect()` 走完整流程（起服务 → 两个场景 perf → 精度），产物落在 `runs/<run_id>/metrics.json`。
   - `metrics.ok` 必须为 `true`；为 `false` 的那次不能用来标定；
   - 用参考配置**原样**跑，不要在标定过程中顺手调参。
4. **重复取中位数**：同一配置至少跑 3 次，每个字段取中位数；3 次的离散度超过约 5% 说明机器不稳（别的进程抢卡、温度/降频），先排除再标定。
5. **填进 task.json**：只改 `baseline` 的四个值，`thresholds` 不动。
6. **自检**：

~~~bash
python -m json.tool dependencies/srv-bench/tasks/config/cfg-tp-choice/task.json > /dev/null
python dependencies/srv-bench/tasks/config/cfg-tp-choice/verify.py dependencies/srv-bench/tasks/config/cfg-tp-choice
~~~

   `verify.py` 会打印 `[INFO] baseline 全为 null → needs calibration` 或 `[INFO] baseline 已标定`；它同时会拒绝**只填了一半**的 baseline（要么全 null，要么四个都填）。

## 3.5 两种 baseline 形态（契约 8/D4）

~~~jsonc
"baseline": {"throughput_req_s": null}                             // 未标定 -> unscored
"baseline": {"throughput_req_s": 1.23}                             // 单值: 所有被比对场景共用
"baseline": {"throughput_req_s": {"4k2k": 1.23, "16k1k": 0.41}}    // 逐场景取值(正式标定推荐)
~~~

- 判分**逐场景**比对；"可比对场景" = 该场景未 skip 且其 baseline 值非 null；
- **所有可比对场景都达标才算 passed**，不允许用 4k2k 的增益掩盖 16k1k 的退化；
- 可比对场景数为 0 → unscored（None），不是 0；
- 单值时，"实测"取该场景自己的实测值，与这个共享参照点比；
- 本目录的 `verify.py` 两种形态都接受，逐场景对象的键必须在 task.json 的 `scenarios` 里。

## 4. 把 metrics.json 填进 task.json 的参考脚本

纯标准库，不做任何数值推断：缺字段就留 null。

~~~python
import json
from pathlib import Path

run_metrics = Path("runs/<run_id>/metrics.json")
task_json = Path("dependencies/srv-bench/tasks/config/cfg-tp-choice/task.json")

metrics = json.loads(run_metrics.read_text(encoding="utf-8"))
spec = json.loads(task_json.read_text(encoding="utf-8"))
if not metrics.get("ok"):
    raise SystemExit("metrics.ok=false，这次不能用来标定: " + str(metrics.get("error")))
scenes = {name: blk for name, blk in metrics["scenarios"].items() if not blk.get("skipped")}
if not scenes:
    raise SystemExit("所有场景都被跳过，不产生 baseline")
filled = {}
for field in ("throughput_req_s", "ttft_ms_p90", "tpot_ms", "accuracy"):
    vals = {name: blk.get(field) for name, blk in scenes.items()}
    filled[field] = None if all(v is None for v in vals.values()) else vals
spec["baseline"].update(filled)
task_json.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
~~~

多次标定取中位数时，先用同样的方式把 3 次结果读出来算中位数，再写一次，不要手敲数字。

## 5. 标定提交的规范

- 提交说明里写清：机器型号与卡数、镜像 tag、三个中位数、以及每个任务用的是哪套参考配置（必要时贴完整启动命令）；
- 不得提交「照着别的机型推出来的」数值——宁可保持 `null`；
- 不得为了让任务看起来能过而调低 `thresholds`：门槛是验收标准，不是标定产物；
- 单次标定结果只在当次有效：镜像、驱动、机器任一项变化后需要重标，旧值要连同原因一起改。

## 6. 当前状态一览

| task_id | category | baseline | 说明 |
|---|---|---|---|
| `config/cfg-tp-choice` | config | 全 null | needs calibration；参照配置 = 出厂默认 TP=8 |
| `config/cfg-graph-batch` | config | 全 null | needs calibration；参照配置 = 关闭图捕获 + 默认并发 |
| `config/cfg-mem-budget` | config | 全 null | needs calibration；参照配置 = 出厂默认启动参数 |
| `measure/msr-pure-p` | measure | 全 null | 不参与 rel_gain，保持 null |
| `measure/msr-pure-d` | measure | 全 null | 不参与 rel_gain，保持 null |
| `measure/msr-accuracy` | measure | 全 null | 不参与 rel_gain，保持 null |

标定完成后，把上表对应行的「全 null」改成「已标定（日期 + 机器）」，方便下一个人判断值是否还有效。
