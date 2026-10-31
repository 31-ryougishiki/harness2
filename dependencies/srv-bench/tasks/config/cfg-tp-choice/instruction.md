# 在 8 卡 Ascend 上为 Qwen3-8B 选择张量并行度

## 背景

服务现在固定以 `--tensor-parallel-size 8` 启动：权重被切成 8 份，每卡只存 1/8，但每一层都要做一次跨卡归约。
并行度调小，每卡权重变多、通信量下降，但单卡算力与显存压力上升；并行度调大则相反。哪一边在这台机器上更划算只能实测。
请给出你的判断、支撑它的实测证据，以及能一键复现的脚本。

## 环境（不可变）

| 项 | 值 |
|---|---|
| 模型 | `Qwen/Qwen3-8B`（原生上下文 32768 tokens，**不启用 YaRN**） |
| 镜像 | 由环境变量 `SRV_IMAGE` 注入（vllm-ascend 0.23.0 系），不得写死其他镜像 tag |
| 硬件 | Ascend A2/A3，单机 8 卡 |
| 压测 | 镜像内 `ais_bench` 的 perf 模式 |
| 精度数据集 | `gsm8k_gen_0_shot_cot_chat_prompt.py` |

场景（输入 token, 输出 token）：

- `4k2k` = (4096, 2048)
- `16k1k` = (16384, 1024)
- `64k1k` / `128k1k` 超出模型上下文，不要请求（请求了也只会被跳过）。

## 允许你改动

- 服务启动参数：`--tensor-parallel-size`，以及为让所选并行度跑通/跑满必须配套调整的参数（如 `--max-model-len`、`--gpu-memory-utilization`）
- 你的 `run.sh` / `report.md` / `metrics.json`

## 禁止

- 只调并行度却修改了与并行度无关的开关（会污染对比）
- 改模型权重、tokenizer、量化方式或镜像 tag
- 改 `ais_bench` 的压测配置、数据集，或换用别的压测工具
- 请求 `64k1k` / `128k1k`
- 手写、估算或推测 `metrics.json` 里的数值
- 跳过要求覆盖的场景；确实测不了必须写 `skipped: true` 和原因

## 必须做到

- 至少实测两组不同的张量并行度，其中一组必须是当前默认的 `TP=8`。
- 每个场景都要在同一套服务上跑一次 gsm8k 精度，把结果写进 `accuracy`；不允许只报吞吐不报精度。
- 服务起不来的配置也要记进 `report.md`，并写清报错。

## 必须交付

### 1. `metrics.json`

真实执行产生的测量结果，字段名必须与下面模板一致；测不到就 `null`，不要填估计值。

```json
{
  "task_id": "cfg-tp-choice",
  "ok": true,
  "error": "",
  "image": "<SRV_IMAGE 的实际值>",
  "model": "Qwen/Qwen3-8B",
  "service": {
    "mode": "single",
    "p_cmd": "<实际启动命令>",
    "d_cmd": null,
    "prefix_cache": {"p": "<实际值，没有就写 n/a>", "d": "n/a"}
  },
  "scenarios": {
    "4k2k": {
      "skipped": false,
      "reason": "",
      "throughput_req_s": null,
      "ttft_ms_p90": null,
      "tpot_ms": null,
      "accuracy": null,
      "aisbench_cmd": "<实际执行的命令>",
      "raw": "artifacts/aisbench_4k2k.json"
    },
    "16k1k": {
      "skipped": false,
      "reason": "",
      "throughput_req_s": null,
      "ttft_ms_p90": null,
      "tpot_ms": null,
      "accuracy": null,
      "aisbench_cmd": "<实际执行的命令>",
      "raw": "artifacts/aisbench_16k1k.json"
    }
  }
}
```

字段含义：

- `throughput_req_s`：每秒完成的请求数（req/s）
- `ttft_ms_p90`：首 token 时延 P90（ms）
- `tpot_ms`：每个输出 token 的平均耗时（ms）
- `accuracy`：gsm8k 准确率，用 0~100 的百分数（只写数字，不要带百分号）；没跑精度就填 null
- `aisbench_cmd`：该场景实际执行的完整命令
- `raw`：相对 run 目录的产物路径，文件缺失则该场景无效
- 服务没起来或压测异常退出：置 `ok: false` 并在 `error` 写清原因，**不要伪造数值**
- 被跳过的场景必须 `skipped: true` 且 `reason` 非空

### 2. `run.sh`

一键复现：起服务 → 跑 `4k2k` → 跑 `16k1k` → 跑一次 gsm8k 精度 → 落 `metrics.json`。
脚本里给出的命令必须与实际执行过的一致，不得事后改写或简化。

### 3. `report.md`

写清你的过程与判断依据：试过哪些组合、各自的实测值与失败原因、最终选择及理由。

## 时间预算

3600 秒：从开始动手到三个交付物落盘。超时未完成按未完成处理。
