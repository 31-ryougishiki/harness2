# SRV: inference-service work

`srv` is harness2's cell for the work that makes an inference service faster without making
its answers worse. The agent gets a launch kit, a fixed model and a fixed context window, and
either changes how the service is launched, drives the load client and reports what happened,
or changes serving code and re-measures. Everything measured happens on an Ascend host; this
repository only holds the seam, the task packages and the launch kit.

## What is fixed

| item | value |
|---|---|
| model | `Qwen/Qwen3-8B` |
| native context | 32768 tokens; YaRN stays off |
| image | `SRV_IMAGE`, injected by the environment -- never a tag inside a task |
| hardware | Ascend A2/A3, one node |
| scenarios | `4k2k` = (4096, 2048) and `16k1k` = (16384, 1024) measurable; `64k1k` /
  `128k1k` exceed the window and are reported as skipped with a reason |
| topology | `single` (one instance) or `pd` (1P1D, both inside one container; prefill keeps
  the prefix cache, decode turns it off) |
| accuracy set | `gsm8k_gen_0_shot_cot_chat_prompt.py`, tolerance +-0.5% |

`harness2/benchmarks/srv_runtime.py` owns `SRV_SCENARIOS` and `MODEL_CTX_TOKENS`; nothing
else may restate them.

## Three kinds of task

| category | the work |
|---|---|
| `config` | choose launch settings (parallelism, batch, memory fraction, cache) that the device and the 32768-token window actually allow |
| `measure` | build the path that brings the service up and drives the client, and report what ran and what could not |
| `code` | change serving code inside an allowed path, then re-measure with the same client call |

## Commands

    harness2-setup srv                      # validate SRV_IMAGE, create the skeleton
    harness2-setup srv --dry-run            # the same plan, nothing written, no docker
    harness2-check --bench srv              # readiness (needs the Ascend host)
    python -m harness2.preflight --bench srv --dry-run    # offline plan, no container
    harness2 --bench srv --substrate oc --k 1 --split all # the sweep: roll, improve, select
    harness2-judge --bench srv --k 1        # mark the selected runs
    harness2-report --bench srv --k 1       # pass rate and grade per mode
    python scripts/validate_srv_tasks.py    # every task package, offline

The sweep needs `docker`, the Ascend driver, `SRV_IMAGE` and the task packages; the dry-run
and the package validator need none of them, and neither imports docker nor the client.

## What the adapter owns, and what it does not

`harness2/benchmarks/srv.py` supplies the `BenchSpec` (`srv_spec()`), the task prompts
(`srv_prompts()`) and the adapter: task reading, prompt rendering, harness staging,
run-directory bookkeeping, delivery, trajectory, and the marked verdict. It never measures
anything itself and never invents a figure.

- `task_prompt()` returns `instruction.md` with any front-matter stripped. `task.json`
  (thresholds, baseline) never reaches a prompt, and `task_files()` returns `None` so the
  judge-side `reference/` directory is never shipped.
- `base_harness()` returns the reference tree: the always-on body plus `launch_kit/`, copied
  from `dependencies/srv-bench/base_harness/`. A proposed harness replaces the always-on body;
  the kit rides along with every staging because it is not on the editable surface.
- `score()` reads the `metrics.json` that `srv_runtime.collect` wrote and hands it to
  `srv_grade.grade`. A missing file, a missing marker module, a raised exception or a task
  whose baseline is not calibrated all return `None` (unscored), never `0`.
- `validate_tree()` refuses a harness whose always-on body loses the launch-kit path, loses
  the `deliverables/` rule, or names judge-side material (`thresholds`, `reference/`).

### The rollout seam

The adapter starts the substrate itself, the way the LAB adapter does: `SrvAdapter._launch()`
builds the command from the substrate's reference argv (`OpencodeSubstrate.run_argv` through bun
from the pinned checkout, or `CodexSubstrate.exec_argv`), seeds the workspace from the staged
harness, runs it detached with the stream going to `run_dir/opencode_session.jsonl` (or
`codex_session.jsonl`) and the diagnostics to `run_dir/runner.log`, and records the exact
invocation in `run_dir/runner.json`. A non-zero exit is a failed rollout, never a cached one.

The prompt is the harness's own always-on text (read from the STAGED tree, so a proposed harness
is what runs) followed by the task's `instruction.md`. The always-on files are not written to
the workspace: they travel in the prompt, and a second copy on disk would be injected twice.

`srv_runtime` keeps to what CONTRACT.md section 5 lists -- the container, the service, the
client runs and `collect`. Nothing outside that list is asked of it, and `collect` runs after
the rollout returns, writing `metrics.json` into the run directory.

## Judging and calibration

`srv_grade.grade(metrics, task_json)` returns `{"score": 0..1, "passed": bool, "detail": ...}`
for a calibrated task and `None` for an uncalibrated one. `srv.score()` expresses the score on
a 100-point denominator so the shared judge's `frac` is that score, and carries `passed`
through as `all_pass`. An uncalibrated task is reported, not failed: `harness2-report` prints
a NOTE that those rows carry no grade and are left out of the denominator.

`dependencies/srv-bench/baseline/README.md` says how to fill a task's baseline in. Until every
value is real, leave them all `null`; a half-filled baseline is a defect, and the task
packages' own `verify.py` rejects it.

### Contract revisions this cell depends on

- **v1.1 section 8 (D1/D3/D4)**: a `measure` task may ship `thresholds: {}` -- it is never
  compared by number; accuracy is a 0-100 percentage and a tolerance is in percentage points; a
  baseline value is `null`, one number, or a per-scenario map, and grading compares per
  scenario. The adapter passes `task.json` through untouched and interprets none of it.
- **v1.2 section 9 (D7/D8/D9)**: metrics.json's keys are `task_id, ok, error, image, model,
  service, deliverables, scenarios` plus `changed_files` for `code` tasks;
  `deliverables` and `changed_files` are filled by `collect()` by scanning the run
  directory, not self-reported, and a grader that finds them missing returns `None` (unscored)
  rather than blaming the agent; `forbidden_paths` and `protected_paths` are one constraint
  under two spellings and the union applies. `scripts/validate_srv_tasks.py` reads the same key
  set and accepts either path spelling.

## Known gaps

1. `harness2-setup` and `harness2-check` still declare their own `choices=("jb", "lab", "wb")`
   tuples, so `--bench srv` is rejected by argparse before this adapter is reached. One line
   each is the whole fix (`harness2/cli/check.py` and `harness2/cli/setup.py`); both files were
   outside this change's file boundary, so the working entry point is
   `python -m harness2.preflight --bench srv --dry-run` and `make_adapter("srv", ...)`.
   `harness2/install.srv()` exists and is exercised directly for the same reason.
2. The launcher is exercised offline with a stand-in command (argv construction, workspace
   seeding, prompt composition, stream and log files, non-zero exit); the real `bun run` and
   `codex exec` invocations need the pinned checkouts and a reachable model route, so they are
   unverified on this host.
3. `validate_tree()` checks the always-on TEXT, not a rendered prompt: the srv cell has no
   upstream renderer to call (LAB renders through its own checkout). A harness that renders
   differently from its text is caught on the host, at rollout time, not here.
4. No measurement is verified on a host without docker, an NPU and `ais_bench`; on such a host
   the dry-run, discovery, staging, prompt rendering and the package validator are what run.
5. `srv_runtime.py` and `srv_grade.py` are imported lazily everywhere, so this cell imports
   and discovers tasks even when neither module is present.
