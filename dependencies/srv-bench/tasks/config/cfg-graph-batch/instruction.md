# 图捕获/编译开关与 max-num-seqs 联合寻优

## 背景

服务在小 batch 下可以被图捕获/编译加速，但捕获尺寸、并发序列数和显存三者互相牵制：
`--max-num-seqs` 开太小打不满算力，开太大则要捕获更多尺寸、显存吃紧或捕获命中率下降。
请联合调这几类开关，让两个场景的吞吐尽可能高，同时精度不退化。

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

- 图捕获 / 编译相关开关（例如 `--enforce-eager` 是否开启、编译级别等；具体参数名以镜像内 `vllm serve --help` 为准）
- 并发相关：`--max-num-seqs`，以及为配合它必须调整的 `--max-num-batched-tokens`、图捕获尺寸类参数
- 你的 `run.sh` / `report.md` / `metrics.json`

## 禁止

- 用降低精度的手段换吞吐（如换量化、换 dtype）
- 改模型权重、tokenizer、量化方式或镜像 tag
- 改 `ais_bench` 的压测配置、数据集，或换用别的压测工具
- 请求 `64k1k` / `128k1k`
- 手写、估算或推测 `metrics.json` 里的数值
- 跳过要求覆盖的场景；确实测不了必须写 `skipped: true` 和原因

## 必须做到

- 至少覆盖两组不同组合，其中一组必须是「关闭图捕获」的保守组合，作为对照。
- 每组都要记录启动耗时、是否成功、以及两个场景的实测值；胜出的一组必须能稳定跑完两个场景。
- 每个场景都要在同一套服务上跑一次 gsm8k 精度并写进 `accuracy`。

## 必须交付

### 1. `metrics.json`

真实执行产生的测量结果，字段名必须与下面模板一致；测不到就 `null`，不要填估计值。

```json
{
  "task_id": "cfg-graph-batch",
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
