# 纯 P 视角的 prefill 测量（单实例，output_len=1）

## 背景

这次只关心预填充（prefill）阶段的能力：让服务只算第一个输出 token，时间和吞吐基本由输入侧决定。
做法是用 `ais_bench` 的 perf 模式，并把每次请求的输出长度覆盖为 1。
本任务看的是测量流程是否正确、命令与产物是否可复现，不是数字大小。

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

- 服务启动参数（并行度、显存、并发等）可自选，只要能被 `run.sh` 复现
- `ais_bench` 的输出长度覆盖参数与场景参数
- 你的 `run.sh` / `report.md` / `metrics.json`

## 禁止

- 用默认输出长度冒充覆盖为 1 的测量
- 只写命令不写原始产物路径
- 改模型权重、tokenizer、量化方式或镜像 tag
- 改 `ais_bench` 的压测配置、数据集，或换用别的压测工具
- 请求 `64k1k` / `128k1k`
- 手写、估算或推测 `metrics.json` 里的数值
- 跳过要求覆盖的场景；确实测不了必须写 `skipped: true` 和原因

## 必须做到

- 用单实例方式拉起服务（一个容器内只跑一个 vLLM 进程），模型 `Qwen/Qwen3-8B`。
- 对 `4k2k`、`16k1k` 各跑一次 perf，命令里必须显式把输出长度覆盖为 1（即 `output_len=1`）。
- 实际执行的命令原样写进 `metrics.json` 的 `scenarios[*].aisbench_cmd`，并在 `run.sh` 里给出同一条命令。
- 原始结果落盘到 `artifacts/` 下，`metrics.json` 的 `raw` 指向该文件；文件缺失则该场景无效。

## 必须交付

### 1. `metrics.json`

真实执行产生的测量结果，字段名必须与下面模板一致；测不到就 `null`，不要填估计值。

```json
{
  "task_id": "msr-pure-p",
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

一键复现：起服务 → 跑 `4k2k`（output_len=1）→ 跑 `16k1k`（output_len=1）→ 落 `metrics.json`。
脚本里给出的命令必须与实际执行过的一致，不得事后改写或简化。

### 3. `report.md`

写清你的过程与判断依据：试过哪些组合、各自的实测值与失败原因、最终选择及理由。

## 时间预算

2400 秒：从开始动手到三个交付物落盘。超时未完成按未完成处理。
