# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark dependency and runtime readiness checks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from . import config
from .install import PATCHES, missing_runners
from .substrate.opencode import OpencodeSubstrate


def check(adapter, *, judge=False) -> list[str]:
    missing = missing_runners()
    if missing:
        return [missing]
    errors = []

    def probe(label, cmd, *, cwd=None, remedy="run harness2-setup for this benchmark"):
        try:
            r = subprocess.run(
                [str(c) for c in cmd], cwd=cwd, capture_output=True, text=True, timeout=45
            )
            if r.returncode:
                errors.append(f"{label} failed; {remedy}")
            return r.returncode == 0
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{label}: {type(exc).__name__}; see README.md")
            return False

    for validate in (
        lambda: config.validate_models(judge=judge),
        adapter.tasks,
        OpencodeSubstrate().preflight,
        adapter.preflight,
    ):
        try:
            validate()
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(str(exc))

    bench = adapter.spec.name.split("-", 1)[0]
    if bench == "srv":
        errors.extend(srv_errors(adapter))
        return errors
    name = {"jb": "job-bench-eval", "lab": "harvey-labs", "wb": "workbuddy-bench"}[bench]
    repo = config.DEPENDENCIES / name
    for dependency in (name, "opencode"):
        target = config.DEPENDENCIES / dependency
        if (target / ".git").exists():
            try:
                head = subprocess.check_output(
                    ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
                ).strip()
                if head != config.DEPENDENCY_PINS[dependency]:
                    errors.append(f"{dependency}: checkout differs from the supported pin")
            except subprocess.CalledProcessError:
                errors.append(f"{dependency}: cannot read checkout revision")
        patch = config.THIRD_PARTY / "patches" / PATCHES[dependency]
        probe(
            f"{dependency} patch",
            ["git", "-C", target, "apply", "--reverse", "--check", patch],
        )
    if not (config.OPENCODE_DIR / "node_modules").is_dir():
        errors.append(f"OpenCode dependencies missing; run harness2-setup {bench}")

    if any(
        m.startswith("google-vertex") for m in (config.SOLVER_MODEL_OC, config.IMPROVER_MODEL)
    ) or (judge and (bench == "wb" or config.JUDGE_MODEL.startswith(("claude", "gemini")))):
        probe(
            "Google Application Default Credentials",
            ["gcloud", "auth", "application-default", "print-access-token"],
        )
    if bench == "lab":
        for binary in ("pandoc", "pdftoppm"):
            if not shutil.which(binary):
                errors.append(f"{binary} missing; install pandoc and poppler-utils (README.md)")
        probe(
            "LAB document and judge dependencies",
            [config.LAB_PYTHON, "-c", "import evaluation.scoring; import google.auth, anthropic"],
            cwd=repo,
        )
    elif bench == "jb" and judge:
        probe(
            "JobBench document and judge dependencies",
            [
                config.JB_JUDGE_PYTHON,
                "-c",
                "import pandas, openpyxl, docx, mammoth, pdfplumber, pptx, openai",
            ],
        )
        if not config.JB_JUDGE_API_BASE:
            errors.append(
                "Set GOOGLE_CLOUD_PROJECT or HARNESS2_JUDGE_BASE_URL for JobBench judging"
            )
        if (
            config.JUDGE_MODEL
            and not config.JB_JUDGE_API_KEY
            and not config.JUDGE_MODEL.startswith("gemini")
        ):
            errors.append(
                "A non-Gemini JobBench judge requires HARNESS2_JUDGE_BASE_URL and HARNESS2_JUDGE_API_KEY"
            )
    elif bench == "wb":
        probe("Docker daemon", ["docker", "info", "--format", "{{.ServerVersion}}"])
        if adapter.spec.name == "wb-office":
            probe(
                "WorkBuddy document rendering",
                [config.WB_RENDER_PY, "-c", "import openpyxl, pptx"],
                remedy="install 'harness2[render]' in the main environment",
            )
        probe(
            "WorkBuddy runtime",
            [
                repo / ".venv/bin/python",
                "-c",
                "import harbor; from workbuddy_bench.runner.harness_adapters import HARNESS_ADAPTERS; "
                "assert all(k in HARNESS_ADAPTERS for k in ('opencode', 'oc-harness', 'codex-harness'))",
            ],
            cwd=repo,
        )
        for role, expected in (
            ("solver", config.SOLVER_MODEL_OC.removeprefix("google-vertex/")),
            ("judge", config.JUDGE_MODEL),
        ):
            path = repo / "configs/models/harness2" / f"{role}.yaml"
            try:
                if json.loads(path.read_text())["model"]["name"] != "google/" + expected:
                    errors.append(f"WorkBuddy {role} configuration is stale; rerun setup")
            except (OSError, ValueError, KeyError, TypeError):
                errors.append(f"WorkBuddy {role} configuration missing; rerun setup")
    return errors


def require_ready(adapter, *, judge=False):
    """Raise before a caller spends anything when this benchmark is not ready to run.

    `judge=True` also demands what grading will need: `HARNESS2_JUDGE_MODEL` set, and,
    where a benchmark grades in its own environment, that environment installed and
    pointed at an endpoint. Those are offline presence and import checks — they catch an
    unset or unbuilt judge, not an account that cannot serve the judge model, which is
    still first learned when the judge actually calls it.
    """
    errors = check(adapter, judge=judge)
    if errors:
        raise config.ConfigError(
            "Setup incomplete; nothing has run yet:\n  - " + "\n  - ".join(errors)
        )


def srv_errors(adapter) -> list[str]:
    """The srv cell's own readiness: image, task tree, launch kit, marker module, host tools.

    Offline on purpose: no container is started and nothing is measured. The machine checks
    that only the Ascend host can answer (driver present, image pullable) are the runtime's.
    """
    from .benchmarks import srv as S

    errs = []
    if not S.SRV_DIR.is_dir():
        errs.append(f"srv task tree missing at {S.SRV_DIR}; run harness2-setup srv")
    missing_kit = [n for n in S.KIT_NAMES if not (S.KIT_DIR / n).is_file()]
    if missing_kit:
        errs.append(f"launch kit incomplete under {S.KIT_DIR}: {', '.join(missing_kit)} missing")
    for module in ("srv_runtime", "srv_grade"):
        if not (config.PKG / "benchmarks" / f"{module}.py").is_file():
            errs.append(
                f"harness2/benchmarks/{module}.py is missing; srv cannot run or mark a rollout "
                f"without it (CONTRACT.md sections 5 and 6)"
            )
    if not shutil.which("docker"):
        errs.append("docker is not on PATH; srv runs its service inside a container")
    if os.name != "nt":
        dev = Path("/dev")
        if dev.is_dir() and not any(dev.glob("davinci*")):
            errs.append(
                "no /dev/davinci* device node: the Ascend driver is not exposed to this host"
            )
    try:
        tasks = adapter.tasks()
    except (OSError, RuntimeError, ValueError) as exc:
        errs.append(str(exc))
    else:
        print(f"  srv: {len(tasks)} task(s) discovered", flush=True)
    # adapter.preflight() is not repeated here: check() already ran it once, above.
    return errs


def srv_plan(adapter) -> list[str]:
    """What a real srv run needs on this host, and which of it is present. Reads only."""
    from .benchmarks import srv as S

    try:
        n_tasks = len(adapter.tasks())
    except (OSError, RuntimeError, ValueError):
        n_tasks = 0
    image = config.env(S.IMAGE_ENV, "")
    rows = [
        ("cell", f"{adapter.spec.name}/{adapter.substrate.name} "
                 f"install_mode={adapter.spec.install_mode}"),
        ("image", f"{S.IMAGE_ENV}={image or '(unset)'}"),
        ("model", f"{S.MODEL}, context {S.MODEL_CTX_TOKENS} tokens, YaRN off"),
        ("scenarios", "4k2k/16k1k measurable; 64k1k/128k1k exceed the window and are skipped"),
        ("tasks", f"{n_tasks} discovered under {S.SRV_DIR / 'tasks'}"),
        ("launch kit", f"{', '.join(S.KIT_NAMES)} under {S.KIT_DIR}"),
        ("docker", "on PATH" if shutil.which("docker") else "MISSING"),
        ("rollout", "substrate argv built by the adapter (bun/opencode or codex), run detached"),
    ]
    out = ["srv plan (dry run: nothing is started, no container, no model call)"]
    out += [f"  {key:<11} {value}" for key, value in rows]
    out += [
        "  next        run harness2-check --bench srv on the Ascend host; a real run needs "
        "docker, the driver and SRV_IMAGE",
    ]
    return out


def main(argv: list[str] | None = None) -> int:
    """Readiness as a command: python -m harness2.preflight --bench srv [--dry-run]."""
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m harness2.preflight",
        description="Benchmark dependency and runtime readiness checks.",
    )
    p.add_argument("--bench", required=True, help="jb | wb | lab | srv")
    p.add_argument("--subset", default=None, choices=("office", "web", "code"))
    p.add_argument("--substrate", default="oc", choices=("oc", "cx"))
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan for this cell and start nothing (srv: offline by construction)",
    )
    args = p.parse_args(argv)
    from .cli._common import make_adapter

    adapter = make_adapter(args.bench, args.subset, args.substrate)
    if args.dry_run and args.bench == "srv":
        print("\n".join(srv_plan(adapter)))
        return 0
    errors = check(adapter, judge=True)
    for error in errors:
        print(f"FAIL: {error}")
    if errors:
        print(f"Setup incomplete: {len(errors)} issue(s).")
        return 1
    print(f"Ready: {adapter.spec.name}/{args.substrate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
