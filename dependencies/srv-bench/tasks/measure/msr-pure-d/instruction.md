# PD 分离 1P1D 的 decode 测量（P 开 prefix cache，D 关闭）

## 背景

这次做 PD 分离：同一容器内起两个实例，一个只负责 prefilling（P），一个只负责 decoding（D）。
为了压缩预填充对 decode 的干扰，P 侧开启 prefix cache 并设到 100%，D 侧关闭 prefix cache。
输出长度保持场景默认值，用长输出观察 decode 表现。本任务看的是测量流程与拓扑是否如实记录，不是数字大小。

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

- 两个实例各自的启动参数（端口、并行度、显存、并发等），只要能被 `run.sh` 复现
- 你的 `run.sh` / `report.md` / `metrics.json`

## 禁止

- 用一个实例既当 P 又当 D，或让 P 承担 decode、D 承担 prefill
- 在 D 侧打开 prefix cache（或在 P 侧关掉）
- 改模型权重、tokenizer、量化方式或镜像 tag
- 改 `ais_bench` 的压测配置、数据集，或换用别的压测工具
- 请求 `64k1k` / `128k1k`
- 手写、估算或推测 `metrics.json` 里的数值
- 跳过要求覆盖的场景；确实测不了必须写 `skipped: true` 和原因

## 必须做到

- 同一容器内起两个 vLLM 实例（同机 1P1D），模型 `Qwen/Qwen3-8B`。
- P 实例开启 prefix cache 并设置为 100%；D 实例关闭 prefix cache。
- 对 `4k2k`、`16k1k` 各跑一次 perf，不要覆盖输出长度。
- `metrics.json` 里 `service.mode` 为 `pd`，`p_cmd` / `d_cmd` 是两个实例真实执行过的命令，`prefix_cache` 为 `{"p": "100%", "d": "off"}`。
- `report.md` 说明拓扑、端口分配、P 与 D 各自的参数与分工。

## 必须交付

### 1. `metrics.json`

真实执行产生的测量结果，字段名必须与下面模板一致；测不到就 `null`，不要填估计值。

```json
{
  "task_id": "msr-pure-d",
  "ok": true,
  "error": "",
  "image": "<SRV_IMAGE 的实际值>",
  "model": "Qwen/Qwen3-8B",
  "service": {
    "mode": "pd",
    "p_cmd": "<P 实例实际启动命令>",
    "d_cmd": "<D 实例实际启动命令>",
    "prefix_cache": {"p": "100%", "d": "off"}
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

一键复现：起 P、D 两个实例 → 跑 `4k2k` → 跑 `16k1k` → 落 `metrics.json`。
脚本里给出的命令必须与实际执行过的一致，不得事后改写或简化。

### 3. `report.md`

写清你的过程与判断依据：试过哪些组合、各自的实测值与失败原因、最终选择及理由。

## 时间预算

2400 秒：从开始动手到三个交付物落盘。超时未完成按未完成处理。
