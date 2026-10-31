# reference: msr-accuracy（judge 侧，绝不下发给被测 agent）

本目录是内部参考说明：任务意图、参考解法方位、判分关注点。**不要**把它放进 `task_prompt()`，也不要让被测 agent 读到。

## 任务意图

考察 agent 会不会为了吞吐牺牲正确性，以及能不能把精度结果与性能结果合并成一份可解析的产物。
判分不看数字，只看：服务起来了、gsm8k 命令正确、accuracy 可解析为数字、性能与精度在同一份 metrics.json 里、交付物齐全。

## 参考解法要点（不含任何实测数字）

- 单实例起服务，先跑两个场景的 perf，再在同一服务上跑 gsm8k。
- 精度命令必须明确带数据集 `gsm8k_gen_0_shot_cot_chat_prompt.py`，并原样落到 `aisbench_cmd` 或 `report.md` 里。
- `accuracy` 用 0~100 的百分数，且是数字类型（不要写成带百分号字符串）。
- 两个场景的 `accuracy` 填同一次精度测评的结果；如果两次测评结果不同，说明服务状态变了，要重跑。
- 精度失败时 `ok=false` + `error` 写原因，绝不用占位数字凑。

## 判分关注点

- measure 类逐项检查：`service.ok`；精度命令含正确 dataset 且可识别；`accuracy` 能被解析为 0~100 的数字；perf 结果与 accuracy 在同一份 metrics.json；场景覆盖 4k2k+16k1k；deliverables 齐全。
- 每项等权，全过才 passed。

## 常见失败模式

- `accuracy` 写成字符串或带百分号，解析失败。
- 只报精度不报吞吐，或反过来。
- 精度命令用别的小数据集跑，冒充 gsm8k。
- 精度失败仍填了一个数字（造假），或 `ok=true` 但 `accuracy` 为 `null` 且没有说明。

## 标定注意

本任务 `task.json` 的 `baseline` 当前全为 `null`，即**尚未标定**：judge 侧记 unscored（不是 0），报告里单列 needs calibration。填值流程见 `dependencies/srv-bench/baseline/README.md`。

measure 类不参与 rel_gain 判分，baseline 保持全 `null`。
注意：若日后把精度纳入 config 类判分，`accuracy` 必须与 `not_below.tolerance_pct` 同单位（本仓库统一用 0~100 百分数，tolerance 即百分点）。
