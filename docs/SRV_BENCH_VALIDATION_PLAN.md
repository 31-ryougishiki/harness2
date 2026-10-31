# SRV-Bench 环境验证计划

在 NPU 宿主机上按阶段推进。每阶段都有明确的"过了/没过"判据，没过的阶段不要往下走。
凡是需要回填给我的数据，都写在该阶段的"回填"里。

## 阶段 0 · 环境就绪（约 30 min）

`python -m harness2.cli.check --bench srv` 现在报 4 条 FAIL，逐条清掉：

| FAIL | 处置 |
|---|---|
| `SRV_IMAGE is required` | `export SRV_IMAGE=<你的 vllm-ascend 镜像 tag>`；镜像 tag 只在这一处出现，任务包里从不带 |
| `HARNESS2_SOLVER_MODEL is required` | `export HARNESS2_SOLVER_MODEL=<provider/model>` |
| `OpenCode checkout missing` | 或改用 `--substrate cx` |
| `docker is not on PATH` | 宿主机需 docker + Ascend docker runtime（`docker info` 能看到 runtime 与 device） |

**判据**：`harness2-check --bench srv` 零 FAIL。

## 阶段 1 · 单点连通性（约 1–2 h）

按"炸了最便宜"的顺序，一次只验一件事，不要跳步：

1. **容器能看见卡**
   `docker run --rm --device /dev/davinci0 --device /dev/davinci_manager <image> npu-smi info`
   判据：打印出卡列表。若 `/dev/davinci*` 命名不同，改 `srv_runtime.container_start` 的 glob。

2. **服务能起**
   手工跑一次 `dependencies/srv-bench/base_harness/run_service.sh`。
   判据：`curl 127.0.0.1:8000/health` 返 200。
   **重点核对**：`--enable-prefix-caching` / `--no-enable-prefix-caching` 在你这个 0.23.0 版本是否存在；
   PD 的 KV connector 名（默认 `P2pNcclConnector`，Ascend 上可能不同，用 `cfg["kv_connector"]` 覆盖）。

3. **aisbench 能跑**
   `aisbench --help`，逐个比对 `SRV_AISBENCH_PERF_TEMPLATE` / `SRV_AISBENCH_ACC_TEMPLATE` 里的占位符。
   判据：纯P 那条命令能真跑出 summary。
   **这是最可能一次性改多项的地方**，flag 名按你镜像内的版本改，不要照着模板硬套。

4. **collect() 能解析**
   跑一次 perf，把 aisbench 的 summary JSON 结构与 `srv_runtime._extract_perf` 的键名启发式对照。
   判据：`metrics.json` 里 `throughput_req_s` / `ttft_ms_p90` / `tpot_ms` 三键非 null。

**回填给我**：① `npu-smi info` 输出 ② 实际可用的 PD 启动命令行 ③ aisbench summary JSON 的键名结构

## 阶段 2 · 标定 baseline（约 2–4 h）

先只标 3 个 seed 任务：`cfg-tp-choice` / `msr-pure-p` / `kod-sched-batch`。

- 按 `dependencies/srv-bench/baseline/README.md` 的合并脚本产出 `{scenario: value}` 形态
- 填进各自 `task.json` 的 `baseline`
- `python scripts/validate_srv_tasks.py` 确认该任务的 warning 消失

**判据**：seed 任务的 `baseline` 不再是全 null。
**注意**：类1/类3 在标定前一律 unscored（不是 0），这是设计，不是 bug。

## 阶段 3 · 端到端一次真跑（约 1–2 h）

```
harness2 --bench srv --k 1 --mode parallel
harness2-judge --bench srv --k 1 --mode parallel
harness2-report --bench srv --k 1 --mode parallel
```

**判据**：① 有 run 目录产出 ② judge 出分 ③ 失败项必须是 `unscored` 而不是被记成 0 分
（如果看到"服务没起来"被记成 0 分，那是判分缺陷，回来找我）

## 阶段 4 · 类3 的 PR 参考确认

```
python scripts/pick_perf_prs.py --repo vllm-project/vllm-ascend \
    --since <0.23.0 基线 commit> --py-only --limit 20 --out candidates.json
```

人工确认后，把 PR 编号填进对应 `kod-*/reference/README.md` 的 TODO。
**只选不需要重编译算子的**（`--py-only` 已过滤 `csrc/`），否则 40min 预算内编译不完。

## 阶段 5 · 扩容（前四阶段都过之后）

每类从 1 个 seed 扩到 3 个，重标定，再跑一次 `--k 1`。

---

## 全阶段都要盯的三件事

1. **不许出现假数字**：任何拿不到的指标都必须是 `ok=false` 或 `null`，不允许有"看起来像真的"的填充值。
2. **skip 必须有理由**：`64k1k`/`128k1k` 超 32768 上下文，`reason` 字段必须非空。
3. **判据不下发**：如果 agent 在 run 里"看到了"baseline 数值或阈值，说明泄漏了，立刻停。
