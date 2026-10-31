# reference: msr-pure-p（judge 侧，绝不下发给被测 agent）

本目录是内部参考说明：任务意图、参考解法方位、判分关注点。**不要**把它放进 `task_prompt()`，也不要让被测 agent 读到。

## 任务意图

考察 agent 能不能按需求改装测量方式：这次只想要 prefill 视角的数据，做法是用 ais_bench 的 perf 模式把输出长度压到 1。
判分不看数字大小，只看流程是否可核对：服务是否真起来、命令是否真的覆盖了 output_len、产物是否存在且路径对得上。

## 参考解法要点（不含任何实测数字）

- 单实例起服务，模型固定 `Qwen/Qwen3-8B`，不启 YaRN。
- 两个场景各跑一次 perf，命令里显式覆盖输出长度为 1；覆盖方式以镜像内 `ais_bench --help` 为准。
- 命令必须落到 `metrics.json` 的 `scenarios[*].aisbench_cmd`，`run.sh` 里给出同一条命令。
- 原始产物写到 `artifacts/aisbench_<scenario>.json` 之类的路径，`raw` 用相对 run 目录的相对路径指向它，且文件必须真实存在。
- 输出长度被覆盖后 `tpot_ms` 语义意义有限，可填实测值也可留 `null`，但不得编造。

## 判分关注点

- measure 类**不比数字**。逐项检查：`service.ok`、`aisbench_cmd` 是否含正确的 dataset/mode/model 与 output_len=1、`raw` 是否可解析、`scenarios` 是否覆盖 4k2k+16k1k（skipped 必须带非空 reason）、`deliverables` 是否齐全。
- 每项等权，全过才 passed；任何一项不满足即未通过，而不是按数字扣分。

## 常见失败模式

- 命令里没有 output_len 覆盖，却声称测的是纯 P。
- `aisbench_cmd` 写成占位符，或与实际执行不一致。
- `raw` 指向不存在的文件。
- 只跑 `4k2k`，`16k1k` 既不跑也不标 skipped。
- 服务没起来但 `ok` 仍为 `true`。

## 标定注意

本任务 `task.json` 的 `baseline` 当前全为 `null`，即**尚未标定**：judge 侧记 unscored（不是 0），报告里单列 needs calibration。填值流程见 `dependencies/srv-bench/baseline/README.md`。

measure 类不参与 rel_gain 判分，baseline 保持全 `null` 即可；标定只用于人工看趋势，不要填进 thresholds。
如果将来想给 measure 类加时间预算，用 `thresholds.kind=budget`（`max_s`），不要复用 rel_gain。
