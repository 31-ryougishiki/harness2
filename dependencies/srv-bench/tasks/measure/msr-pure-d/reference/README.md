# reference: msr-pure-d（judge 侧，绝不下发给被测 agent）

本目录是内部参考说明：任务意图、参考解法方位、判分关注点。**不要**把它放进 `task_prompt()`，也不要让被测 agent 读到。

## 任务意图

考察 agent 能不能按拓扑约束把 PD 分离真的跑起来：同一容器内的 1P1D，P 侧 prefix cache 100%、D 侧关闭。
判分不看数字，只看两条真实命令、拓扑字段、场景覆盖与产物路径是否可核对。

## 参考解法要点（不含任何实测数字）

- 同一容器内起两个 vLLM 实例（不同端口），P 只做 prefill、D 只做 decode，二者通过 PD 传输通道对接。
- P 侧开启 prefix cache 并设置为 100%；D 侧关闭 prefix cache（具体 flag 以镜像内 `vllm serve --help` 为准）。
- 两个场景各跑一次 perf，保持场景默认输出长度，不要覆盖。
- `service.mode` 必须是 `pd`，`p_cmd` 与 `d_cmd` 必须是两个实例真实执行过的、彼此不同的命令。
- `prefix_cache` 写 `{"p": "100%", "d": "off"}`。

## 判分关注点

- measure 类逐项检查：`service.ok`；`service.mode` 为 pd 且 `p_cmd`、`d_cmd` 非空且不同；`prefix_cache.p` 表示 100%、`prefix_cache.d` 表示关闭；`aisbench_cmd` 含正确 model/mode；`raw` 可解析；场景覆盖 4k2k+16k1k；deliverables 齐全。
- 每项等权，全过才 passed。

## 常见失败模式

- 只起了一个实例，把 `d_cmd` 填成 `null` 或复制 `p_cmd`。
- D 侧也开着 prefix cache（`prefix_cache.d` 写成 100%）。
- 拓扑写对但两侧端口/参数自相矛盾，按命令重放起不来。
- 用单实例冒充 PD，report.md 里却声称做了分离。

## 标定注意

本任务 `task.json` 的 `baseline` 当前全为 `null`，即**尚未标定**：judge 侧记 unscored（不是 0），报告里单列 needs calibration。填值流程见 `dependencies/srv-bench/baseline/README.md`。

measure 类不参与 rel_gain 判分，baseline 保持全 `null`。
如需记录参考量级，另存到 `baseline/notes.md`，不要写进 task.json 的 baseline。
