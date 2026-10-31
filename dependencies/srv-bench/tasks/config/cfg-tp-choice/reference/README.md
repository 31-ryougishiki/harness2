# reference: cfg-tp-choice（judge 侧，绝不下发给被测 agent）

本目录是内部参考说明：任务意图、参考解法方位、判分关注点。**不要**把它放进 `task_prompt()`，也不要让被测 agent 读到。

## 任务意图

考察 agent 会不会默认「张量并行度越大越好」。题干把默认配置固定在 TP=8，但在这类 8 卡 Ascend 机器上 TP=8 往往不是最优：
每层都要跨 8 卡做一次归约，而每卡分到的算子规模很小，通信占比高。真正的最优点要靠实测确定，且可能随场景变化。

期望的行为链：读现状 → 提出一个可证伪的假设（TP=4 通信更少）→ 用实测验证 → 记录失败与代价（单卡显存、启动时间）→ 固定配置并交付可复现脚本。

## 参考解法要点（不含任何实测数字）

- 至少对比 `TP=8`（现状）与 `TP=4`，可再加 `TP=2`、`TP=1` 做边界观察。
- 并行度是否可用取决于模型头数（以镜像内模型 `config.json` 的注意力头数与 KV 头数为准），服务的 `--tensor-parallel-size` 必须整除它们，否则直接起不来。
- 并行度变小时单卡权重变多，需要确认显存仍然够用；如果因此调整了 `--gpu-memory-utilization` 或 `--max-model-len`，必须在 `report.md` 里说明这是配套调整而不是第二个自变量。
- 两个场景必须用同一套配置测完，否则对比被污染。
- 精度：同一套服务上至少跑一次 gsm8k，把结果写进两个场景的 `accuracy`。
- 本目录不给任何实测数值：没有机器时写死数字就是造假，一律留 `null`。

## 判分关注点

- 逐指标判断：`throughput_req_s` 用 `rel_gain`（实测相对 baseline 的百分比增益不小于 `min_pct`），`accuracy` 用 `not_below`（不低于 baseline 减 `tolerance_pct` 个百分点）。
- 全部通过才 `passed`；`metrics.ok=false` 或场景全 skipped → unscored（None），不得记 0。
- 题面把 TP=8 声明为默认配置，因此 baseline 也必须取自 TP=8 的实测值——用最优配置当参照会让 `rel_gain` 失去意义。

## 常见失败模式

- 只跑 TP=8 就交差（没有可对比的第二个点）。
- 换了并行度但没确认 KV 头数整除性，服务起不来又不记录报错。
- 两个场景用了不同并行度，或多改了无关开关，导致归因错误。
- 只报吞吐、accuracy 留 `null`；或直接把表里的数抄进 metrics.json。
- 用 `skipped: true` 跳过某个可测场景且 reason 为空。

## 标定注意

本任务 `task.json` 的 `baseline` 当前全为 `null`，即**尚未标定**：judge 侧记 unscored（不是 0），报告里单列 needs calibration。填值流程见 `dependencies/srv-bench/baseline/README.md`。

标定用配置：出厂默认（`--tensor-parallel-size 8`，其余参数不动）。
把该配置在目标机器上跑出的 `metrics.json` 中 `4k2k` 场景的 `throughput_req_s` / `ttft_ms_p90` / `tpot_ms` / `accuracy` 填入 `baseline`。
若 `4k2k` 与 `16k1k` 都要参与判分，必须先确认 `srv_grade` 的口径是「任一场景达标」还是「全部场景达标」，再决定是否需要分别标定（当前 baseline 结构是单值，见 CONTRACT.md 第 3 节）。
