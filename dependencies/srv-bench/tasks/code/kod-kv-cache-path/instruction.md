# 任务: KV cache 块表与 slot mapping 的 Python 层优化

## 背景
服务已经按参考配置就绪。你的任务不是改配置，而是改代码：先用 profiling 找出 KV cache 块管理里的 Python 侧开销，再交出一处能在现有镜像内直接生效、且不需要重新编译算子的改动。

## 环境（固定，不可改）
- 镜像: vllm-ascend 0.23.0，由环境变量 SRV_IMAGE 注入；不要写死 tag
- 模型: Qwen/Qwen3-8B，原生上下文 32768 tokens，不启 YaRN
- 硬件: Ascend A2/A3，同机部署
- 服务拓扑与 prefix cache 策略由 harness 固定，你不得修改启动参数、拓扑或容器配置
- 容器内没有编译 Ascend 算子的工具链：任何需要重新编译 csrc/ 或 cmake/ 的方案都不可行

## 测量场景
| 场景 | input_len | output_len |
|---|---|---|
| 4k2k | 4096 | 2048 |
| 16k1k | 16384 | 1024 |

单个请求 input_len + output_len 超过 32768 的场景属于超上下文，不测；要跳过必须写清原因。
精度数据集: gsm8k_gen_0_shot_cot_chat_prompt.py，改动前后用同一数据集复核精度。

## 你要改什么
目标是 KV cache 块管理这条路径：每个 decode step 重建 block table / slot mapping 的 Python 循环、
块 id 的重复张量构造与拷贝、prefix cache 命中后的块复用与零拷贝机会，以及输入批量里与 block 相关的簿记。
典型可做的事：把逐 step 的 Python 侧重建换成增量维护或一次向量化、去掉重复的 host 侧拷贝、
避免对未变化的块重复写入。不要改底层算子的语义，也不要依赖尚未合入的其它模块改造。

## 允许修改的路径（相对仓库根）
- vllm_ascend/worker/block_table.py
- vllm_ascend/worker/v2/block_table.py
- vllm_ascend/worker/npu_input_batch.py
- vllm_ascend/core/single_type_kv_cache_manager.py
- vllm_ascend/core/kv_cache_interface.py

只允许改这些路径下的文件；你新增的文件也必须落在这里面。

## 禁止修改
- tests/、benchmarks/、scripts/ 下的任何内容
- csrc/、cmake/ 下的任何内容（改了就需要重新编译）
- docs/、examples/、tools/、.github/、setup.py、pyproject.toml
- 上面没有列出的其它任何文件

不得修改测试与基准脚本。改动必须落在允许路径内，越界改动视为无效提交。

## 交付物（缺一不可）
1. patch.diff：相对仓库根的 git diff，只包含允许路径内的改动
2. report.md，必须写清五件事
   - profiling 证据：用的什么工具或钩子、怎么采集、热点函数与耗时占比靠前的清单
   - 改动点与理由：改了什么、省掉了哪一段开销、为什么这段开销是真的
   - 为什么不需要重新编译
   - 风险与回滚：什么输入、拓扑或负载下可能失效，怎么退回原状
   - 改动前后在同一场景的实测对比，数字必须来自你自己的测量
3. metrics.json，字段如下（测不到就填 null，跳过的场景必须写 reason，不要编数字）：
   task_id、ok、error、image、model、service{mode,p_cmd,d_cmd,prefix_cache}、
   changed_files（本次改动的文件路径列表）、
   deliverables（本次实际交出的文件名，例如 patch.diff、report.md、metrics.json）、
   scenarios{"<场景>": {skipped, reason, throughput_req_s, ttft_ms_p90, tpot_ms, accuracy, aisbench_cmd, raw}}

## 时间预算
2400 s，含 profiling、改码、重测和写报告。预算用完后不再接受新提交，交付物不全视为未完成。
最多 3 次完整的"改一版-测一轮"循环，别把预算耗在反复重装上。

## 完成定义
- 目标场景的吞吐相对参考配置有提升，且精度不下降
- 改动全部落在允许路径内，没有触碰禁止修改的内容
- report.md 里的数字能和 metrics.json 对上，profile 证据能支撑你选的改动点
- 交一处能解释清楚的改动；多处叠加却说不清收益来源，复核时无法归因

## 建议做法（不强制）
- 先量后改：先拉出一个 step 的 Python 侧时间线，找出占比最高的那几段
- 改动要在 Python 层独立成立，不要依赖尚未合入的其它模块改造
- 小步验证：改完先跑最短路径确认功能没坏，再跑完整场景
- KV 相关收益通常在长上下文更明显：16k1k 的块规模比 4k2k 大，先用它定位，再看 4k2k 是否同步变好
