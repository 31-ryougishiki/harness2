# Launch kit (reference)

Two starting points for the service work. **They are yours to change**: rewrite them, split
them, or replace them with something better, as long as the artifacts the assignment names
still come out and every number in them is one you actually observed.

| file | what it is |
|---|---|
| `run_service.sh` | launcher skeleton: one instance, or one prefill plus one decode instance in the same container, then wait for `/health` |
| `aisbench_example.sh` | client skeleton: a perf run for one (input, output) length pair, and the accuracy run |

## What the environment fixes

- Model `Qwen/Qwen3-8B`, native context **32768** tokens. YaRN stays off, so a request pair
  whose input plus output length exceeds 32768 cannot be served: say so instead of running it.
- The image comes from `SRV_IMAGE`. Never write an image tag into a script or a report.
- Hardware is Ascend A2/A3, one node. There is no CUDA path.
- The service must answer `/health` before any client run means anything.

## Habits that keep the record honest

- One log per attempt: the exact command, whether the service came up, and the client call
  you actually ran.
- Change one launch setting at a time and repeat the same client call before comparing.
- When the service does not come up, read the log tail before changing anything else.
- Write the files the assignment names to `deliverables/<filename>`.
