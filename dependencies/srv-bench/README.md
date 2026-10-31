# srv-bench

Task packages for the `srv` cell of harness2: inference-service work on an Ascend host with
vLLM-Ascend, measured through `ais_bench`. `CONTRACT.md` is the interface everything here
is built against; this file only says what is where.

## Layout

    CONTRACT.md              the interface: anchors, schemas, srv_runtime / srv_grade APIs
    tasks/<category>/<id>/   one task package
      task.json              the machine-readable task (schema: CONTRACT.md section 3)
      instruction.md         the brief the agent receives, and nothing else
      verify.py              the package's own offline self-check
      reference/             judge-side notes; NEVER shipped to the agent
    base_harness/            the launch kit the agent starts from and may rewrite
      run_service.sh         launcher skeleton: one instance, or 1P1D in one container
      aisbench_example.sh    client skeleton: perf for one (input, output) pair, and accuracy
      README.md              what the two files do, and what the environment fixes
    baseline/README.md       how the per-task baseline figures get filled in

Categories: `config` (launch-configuration work), `measure` (driving ais_bench and
reporting), `code` (serving-code changes). Directory prefixes follow the category:
`cfg-` / `msr-` / `kod-`.

## The anchors

| item | value |
|---|---|
| model | `Qwen/Qwen3-8B` |
| native context | 32768 tokens, YaRN off |
| image | from `SRV_IMAGE`; no tag is written into a task |
| hardware | Ascend A2/A3, one node |
| scenarios | `4k2k` and `16k1k` measurable; `64k1k` / `128k1k` exceed the window and are skipped with a reason |
| topology | `single` (one instance) or `pd` (one prefill + one decode instance, same container) |
| accuracy set | `gsm8k_gen_0_shot_cot_chat_prompt.py`, tolerance +-0.5% |

## Adding a task

1. `mkdir -p tasks/<category>/<category-prefix>-<name>` and write `task.json` against
   CONTRACT.md section 3, with the revisions in sections 8 and 9 in force: a `measure` task may
   ship `thresholds: {}`, a baseline value may be `null`, one number, or a per-scenario map,
   and a `code` task may spell its scope as `forbidden_paths` or `protected_paths` (the
   union is what applies). Leave every `baseline` value `null` until a real measurement
   exists; a null baseline marks the task unscored, never zero.
2. Write `instruction.md` as the brief only. It must not carry `thresholds`, the baseline
   figures, the contents of `reference/`, or anything from a PR that has not been confirmed.
   The adapter's `task_prompt()` returns this file and nothing else.
3. Add `reference/README.md` with what a human needs to confirm a run by hand.
4. Add `verify.py` that checks the package offline and exits non-zero on failure.
5. Run `python scripts/validate_srv_tasks.py` from the repository root.

## Offline checks

    python scripts/validate_srv_tasks.py          # schema + each package's own verify.py
    python -m harness2.preflight --bench srv --dry-run
    harness2-setup srv --dry-run

None of these need docker, an NPU or `ais_bench`; the container, the service and the client
runs live in `harness2/benchmarks/srv_runtime.py` and only that needs the Ascend host.
