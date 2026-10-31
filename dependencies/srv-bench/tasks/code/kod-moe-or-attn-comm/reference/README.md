# reference（judge 侧，绝不下发）

本任务的 reference 由 scripts/pick_perf_prs.py 选出后人工确认：脚本只负责"基线 commit 之后 merged
+ performance 关键词命中 + 可通过 Python-only 改动重测"的候选筛选；方向是否与本任务匹配、
改动是否落在 allowed_paths 内，必须由 judge 侧人工核对后填写。

## TODO（标定前必须填完）
- TODO PR 编号（pick_perf_prs.py --out 报告里的 number）:
- TODO 上游 PR 标题:
- TODO merge_commit:
- TODO changed_files 是否全部落在本任务 allowed_paths 内:
- TODO 该 PR 的改动能否只用 Python 部分复现（needs_recompile 必须为 false）:
- TODO 复测结论：该改动在 2400 s 预算内能否复现收益（是/否 + 证据路径）:
- TODO 与另外两个 kod-* 任务的方向去重确认:

## 人工确认清单
1. needs_recompile 为 false：没有触碰 csrc/ 与 cmake/
2. changed_files 全部落在本任务 allowed_paths 内，且与 forbidden_paths 无交集
3. 方向匹配：attention / MoE 前后处理与通信重叠相关的 Python 层开销
4. 改动能在 Python 层独立成立，不依赖尚未合入的其它模块
5. 与 kod-kv-cache-path / kod-sched-batch / kod-moe-or-attn-comm 三者的方向互不重叠

## 拿候选的命令
python scripts/pick_perf_prs.py --repo vllm-project/vllm-ascend --since <基线 commit> --limit 50 --py-only --out candidates.json

## 绝不外泄
本目录的任何内容、候选 PR 编号、PR patch、baseline 数值与 thresholds 都不得进入
instruction.md 或任何下发给被评测 agent 的文本。

## 已知缺口
- 服务拓扑（单实例 / 同机 1P1D）最终由 srv_runtime.service_up 决定；落定后需要在 instruction.md 里写实。
- allowed_paths 依据 vllm-ascend v0.23.0 的仓库树挑选，标定阶段需要用镜像里的实际路径再核对一次。
