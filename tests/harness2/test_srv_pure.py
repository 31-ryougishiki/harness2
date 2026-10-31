"""Pure-function tests for the SRV-Bench runtime executor and grader.

Standard library only (``python -m unittest`` or ``python -m pytest``).  Nothing
here needs an NPU, docker, or ais_bench: the container-facing tests stop at the
first docker call and assert the honest ``ok=False`` path.
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness2.benchmarks import srv_grade, srv_runtime  # noqa: E402

MODULE_DIR = ROOT / "harness2" / "benchmarks"
PERF_CMD = (
    "ais_bench --models vllm_api_stream_chat --datasets synthetic_gen --mode perf"
    " --num-prompts 200 --max-input-len 4096 --max-output-len 2048"
    " --work-dir /tmp/srv-aisbench/4k2k"
)
ACC_CMD = (
    "ais_bench --models vllm_api_stream_chat"
    " --datasets gsm8k_gen_0_shot_cot_chat_prompt --mode accuracy"
    " --num-prompts 200 --work-dir /tmp/srv-aisbench/accuracy"
)
DIFF = (
    "--- a/vllm/ascend/foo.py\n"
    "+++ b/vllm/ascend/foo.py\n"
    "@@ -1 +1 @@\n"
    "-old\n"
    "+new\n"
    "--- /dev/null\n"
    "+++ b/tests/harness2/test_srv_pure.py\n"
)
NULL_BASELINE = {
    "throughput_req_s": None,
    "ttft_ms_p90": None,
    "tpot_ms": None,
    "accuracy": None,
}


def config_spec(**overrides) -> dict:
    spec = {
        "id": "cfg-tp-choice",
        "category": "config",
        "domain": "config",
        "mode": "single",
        "scenarios": ["4k2k"],
        "baseline": {"throughput_req_s": 2.0, **{k: None for k in ("ttft_ms_p90", "tpot_ms",
                                                                  "accuracy")}},
        "thresholds": {"throughput_req_s": {"kind": "rel_gain", "min_pct": 3.0}},
        "deliverables": ["metrics.json"],
    }
    spec.update(overrides)
    return spec


def scenario_block(**overrides) -> dict:
    block = {
        "skipped": False,
        "reason": "",
        "throughput_req_s": 2.1,
        "ttft_ms_p90": 400.0,
        "tpot_ms": 12.0,
        "accuracy": None,
        "aisbench_cmd": PERF_CMD,
        "raw": "artifacts/aisbench_4k2k.json",
    }
    block.update(overrides)
    return block


def config_metrics(**overrides) -> dict:
    metrics = {
        "task_id": "cfg-tp-choice",
        "ok": True,
        "error": "",
        "image": "vllm-ascend:0.23.0-a2",
        "model": "Qwen/Qwen3-8B",
        "service": {
            "ok": True,
            "mode": "single",
            "p_cmd": "python -m vllm.entrypoints.openai.api_server --enable-prefix-caching",
            "d_cmd": None,
            "prefix_cache": {"p": "100%", "d": "off"},
            "error": "",
            "log": "",
        },
        "scenarios": {"4k2k": scenario_block()},
        "deliverables": {"metrics.json": True},
    }
    metrics.update(overrides)
    return metrics


def measure_spec(**overrides) -> dict:
    spec = {
        "id": "msr-perf-run",
        "category": "measure",
        "domain": "measure",
        "mode": "single",
        "scenarios": ["4k2k", "64k1k"],
        "baseline": dict(NULL_BASELINE),
        "thresholds": {},
        "deliverables": ["metrics.json", "run.sh", "report.md"],
    }
    spec.update(overrides)
    return spec


def measure_metrics(**overrides) -> dict:
    metrics = {
        "task_id": "msr-perf-run",
        "ok": True,
        "error": "",
        "image": "vllm-ascend:0.23.0-a2",
        "model": "Qwen/Qwen3-8B",
        "service": {
            "ok": True,
            "mode": "single",
            "p_cmd": "python -m vllm.entrypoints.openai.api_server --enable-prefix-caching",
            "d_cmd": None,
            "prefix_cache": {"p": "100%", "d": "off"},
            "error": "",
            "log": "",
        },
        "scenarios": {
            "4k2k": scenario_block(),
            "64k1k": {
                "skipped": True,
                "reason": "exceeds model context: input_len=65536 + output_len=1024"
                          " = 66560 > MODEL_CTX_TOKENS=32768",
            },
        },
        "deliverables": {"metrics.json": True, "run.sh": True, "report.md": True},
    }
    metrics.update(overrides)
    return metrics


class ScenarioTableTest(unittest.TestCase):
    def test_table_is_the_contract_table(self):
        self.assertEqual(
            srv_runtime.SRV_SCENARIOS,
            {
                "4k2k": (4096, 2048),
                "16k1k": (16384, 1024),
                "64k1k": (65536, 1024),
                "128k1k": (131072, 1024),
            },
        )
        self.assertEqual(srv_runtime.MODEL_CTX_TOKENS, 32768)

    def test_four_scenarios_are_split_by_context(self):
        usable, skipped = srv_runtime.applicable_scenarios(
            ["4k2k", "16k1k", "64k1k", "128k1k"]
        )
        self.assertEqual(usable, ["4k2k", "16k1k"])
        self.assertEqual(sorted(skipped), ["128k1k", "64k1k"])
        for name, reason in skipped.items():
            input_len, output_len = srv_runtime.SRV_SCENARIOS[name]
            self.assertIn(f"input_len={input_len}", reason)
            self.assertIn(f"output_len={output_len}", reason)
            self.assertIn(str(srv_runtime.MODEL_CTX_TOKENS), reason)

    def test_total_equal_to_context_is_measurable(self):
        with mock.patch.dict(
            srv_runtime.SRV_SCENARIOS, {"edge": (30000, 32768 - 30000)}, clear=False
        ):
            usable, skipped = srv_runtime.applicable_scenarios(["edge"])
        self.assertEqual(usable, ["edge"])
        self.assertEqual(skipped, {})

    def test_total_over_context_is_skipped(self):
        with mock.patch.dict(
            srv_runtime.SRV_SCENARIOS, {"over": (30000, 32769 - 30000)}, clear=False
        ):
            usable, skipped = srv_runtime.applicable_scenarios(["over"])
        self.assertEqual(usable, [])
        self.assertIn("over", skipped)

    def test_unknown_and_empty_requests(self):
        usable, skipped = srv_runtime.applicable_scenarios([])
        self.assertEqual((usable, skipped), ([], {}))
        usable, skipped = srv_runtime.applicable_scenarios(["8k1k"])
        self.assertEqual(usable, [])
        self.assertIn("8k1k", skipped)


class ApiSurfaceTest(unittest.TestCase):
    def test_contract_names_and_signatures(self):
        for name in (
            "applicable_scenarios",
            "container_start",
            "container_stop",
            "service_up",
            "aisbench_perf",
            "aisbench_accuracy",
            "collect",
        ):
            self.assertTrue(callable(getattr(srv_runtime, name)), name)
        service_up = inspect.signature(srv_runtime.service_up)
        self.assertEqual(list(service_up.parameters)[:2], ["mode", "scenario"])
        for keyword in ("container", "cfg", "log_dir"):
            self.assertEqual(
                service_up.parameters[keyword].kind,
                inspect.Parameter.KEYWORD_ONLY,
                keyword,
            )
        perf = inspect.signature(srv_runtime.aisbench_perf)
        self.assertEqual(list(perf.parameters)[:1], ["scenario"])
        self.assertEqual(perf.parameters["output_len"].default, None)
        self.assertEqual(
            list(inspect.signature(srv_runtime.collect).parameters),
            ["container", "task_dir", "run_dir", "spec"],
        )

    def test_only_stdlib_imports(self):
        stdlib = set(sys.stdlib_module_names)
        for name in ("srv_runtime.py", "srv_grade.py"):
            roots = _imported_roots(MODULE_DIR / name)
            self.assertTrue(
                roots <= stdlib, f"{name} imports non-stdlib modules {sorted(roots - stdlib)}"
            )
            self.assertNotIn("docker", roots)
            self.assertNotIn("requests", roots)


class GradeConfigTest(unittest.TestCase):
    def test_rel_gain_passing(self):
        result = srv_grade.grade(config_metrics(), config_spec())
        self.assertIsNotNone(result)
        self.assertTrue(result["passed"])
        self.assertEqual(result["score"], 1.0)
        check = result["detail"]["checks"][0]
        self.assertAlmostEqual(check["gain_pct"], 5.0)
        self.assertEqual(result["detail"]["failed"], [])

    def test_rel_gain_failing(self):
        metrics = config_metrics(
            scenarios={"4k2k": scenario_block(throughput_req_s=2.01)}
        )
        result = srv_grade.grade(metrics, config_spec())
        self.assertIsNotNone(result)
        self.assertFalse(result["passed"])
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["detail"]["failed"], ["throughput_req_s@4k2k"])

    def test_equality_with_threshold_passes(self):
        metrics = config_metrics(
            scenarios={"4k2k": scenario_block(throughput_req_s=2.06)}
        )
        result = srv_grade.grade(metrics, config_spec())
        self.assertTrue(result["passed"])

    def test_latency_metric_uses_lower_is_better(self):
        spec = config_spec(
            baseline={"ttft_ms_p90": 500.0},
            thresholds={"ttft_ms_p90": {"kind": "rel_gain", "min_pct": 3.0}},
        )
        metrics = config_metrics(
            scenarios={"4k2k": scenario_block(ttft_ms_p90=450.0)}
        )
        result = srv_grade.grade(metrics, spec)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["detail"]["checks"][0]["direction"], "lower_is_better")

    def test_unmeasured_metric_fails_instead_of_passing_silently(self):
        spec = config_spec(
            baseline={"throughput_req_s": 2.0, "tpot_ms": 10.0},
            thresholds={
                "throughput_req_s": {"kind": "rel_gain", "min_pct": 3.0},
                "tpot_ms": {"kind": "rel_gain", "min_pct": 3.0},
            },
        )
        result = srv_grade.grade(config_metrics(), spec)
        self.assertFalse(result["passed"])
        self.assertIn("tpot_ms@4k2k", result["detail"]["failed"])

    def test_accuracy_is_percentage_points(self):
        # CONTRACT 8/D3: accuracy is 0..100, tolerance_pct is percentage points.
        spec = config_spec(
            baseline={"accuracy": 82.0},
            thresholds={"accuracy": {"kind": "not_below", "tolerance_pct": 0.5}},
        )
        ok = srv_grade.grade(
            config_metrics(scenarios={"4k2k": scenario_block(accuracy=82.5)}), spec
        )
        self.assertTrue(ok["passed"], ok)
        check = ok["detail"]["checks"][0]
        self.assertEqual(check["unit"], "percent")
        self.assertAlmostEqual(check["floor_pct"], 81.5)
        edge = srv_grade.grade(
            config_metrics(scenarios={"4k2k": scenario_block(accuracy=81.5)}), spec
        )
        self.assertTrue(edge["passed"], edge)  # exactly on the floor
        bad = srv_grade.grade(
            config_metrics(scenarios={"4k2k": scenario_block(accuracy=81.4)}), spec
        )
        self.assertFalse(bad["passed"])
        self.assertIn("accuracy@4k2k", bad["detail"]["failed"])

    def test_accuracy_is_not_normalised_from_a_fraction(self):
        # 0.825 must NOT be read as 82.5%: no /100 scaling on either side.
        spec = config_spec(
            baseline={"accuracy": 82.0},
            thresholds={"accuracy": {"kind": "not_below", "tolerance_pct": 0.5}},
        )
        result = srv_grade.grade(
            config_metrics(scenarios={"4k2k": scenario_block(accuracy=0.825)}), spec
        )
        self.assertFalse(result["passed"])

    def test_uncalibrated_baseline_is_unscored(self):
        spec = config_spec(baseline=dict(NULL_BASELINE))
        self.assertIsNone(srv_grade.grade(config_metrics(), spec))

    def test_ok_false_is_unscored(self):
        self.assertIsNone(srv_grade.grade(config_metrics(ok=False), config_spec()))

    def test_all_scenarios_skipped_is_unscored(self):
        metrics = config_metrics(
            scenarios={"64k1k": {"skipped": True, "reason": "exceeds model context"}}
        )
        self.assertIsNone(srv_grade.grade(metrics, config_spec()))

    def test_unreadable_inputs_are_unscored(self):
        self.assertIsNone(srv_grade.grade(None, config_spec()))
        self.assertIsNone(srv_grade.grade(config_metrics(), None))
        self.assertIsNone(srv_grade.grade(config_metrics(scenarios={}), config_spec()))
        self.assertIsNone(
            srv_grade.grade(config_metrics(), config_spec(category="unknown"))
        )


class GradeBaselineFormsTest(unittest.TestCase):
    def test_per_scenario_baselines_are_graded_independently(self):
        # CONTRACT 8/D4: a 4k2k gain must never mask a 16k1k regression.
        spec = config_spec(
            scenarios=["4k2k", "16k1k"],
            baseline={"throughput_req_s": {"4k2k": 2.0, "16k1k": 0.4}},
            thresholds={"throughput_req_s": {"kind": "rel_gain", "min_pct": 3.0}},
        )
        metrics = config_metrics(
            scenarios={
                "4k2k": scenario_block(throughput_req_s=2.2),
                "16k1k": scenario_block(throughput_req_s=0.3),
            }
        )
        result = srv_grade.grade(metrics, spec)
        self.assertFalse(result["passed"])
        self.assertEqual(result["detail"]["failed"], ["throughput_req_s@16k1k"])
        self.assertAlmostEqual(result["score"], 0.5)
        summary = result["detail"]["per_scenario"]
        self.assertTrue(summary["4k2k"]["throughput_req_s"]["ok"])
        self.assertFalse(summary["16k1k"]["throughput_req_s"]["ok"])
        self.assertAlmostEqual(
            summary["4k2k"]["throughput_req_s"]["gain_pct"], 10.0
        )
        self.assertEqual(summary["16k1k"]["throughput_req_s"]["baseline"], 0.4)

    def test_flat_baseline_is_shared_by_every_scenario(self):
        spec = config_spec(
            scenarios=["4k2k", "16k1k"], baseline={"throughput_req_s": 2.0}
        )
        metrics = config_metrics(
            scenarios={
                "4k2k": scenario_block(throughput_req_s=2.5),
                "16k1k": scenario_block(throughput_req_s=1.0),
            }
        )
        result = srv_grade.grade(metrics, spec)
        self.assertEqual(result["detail"]["failed"], ["throughput_req_s@16k1k"])

    def test_skipped_scenario_is_not_comparable(self):
        spec = config_spec(
            scenarios=["4k2k", "64k1k"], baseline={"throughput_req_s": 2.0}
        )
        metrics = config_metrics(
            scenarios={
                "4k2k": scenario_block(throughput_req_s=2.5),
                "64k1k": {"skipped": True, "reason": "exceeds model context"},
            }
        )
        result = srv_grade.grade(metrics, spec)
        self.assertTrue(result["passed"], result)
        self.assertEqual(list(result["detail"]["per_scenario"]), ["4k2k"])

    def test_no_comparable_scenario_is_unscored(self):
        spec = config_spec(
            scenarios=["4k2k", "16k1k"], baseline={"throughput_req_s": None}
        )
        metrics = config_metrics(
            scenarios={
                "4k2k": scenario_block(throughput_req_s=2.5),
                "16k1k": scenario_block(throughput_req_s=1.0),
            }
        )
        self.assertIsNone(srv_grade.grade(metrics, spec))

    def test_scenario_baseline_map_missing_a_scenario(self):
        spec = config_spec(
            scenarios=["4k2k", "16k1k"],
            baseline={"throughput_req_s": {"4k2k": 2.0}},
        )
        metrics = config_metrics(
            scenarios={
                "4k2k": scenario_block(throughput_req_s=2.2),
                "16k1k": scenario_block(throughput_req_s=0.1),
            }
        )
        result = srv_grade.grade(metrics, spec)
        self.assertTrue(result["passed"], result)
        self.assertEqual(list(result["detail"]["per_scenario"]), ["4k2k"])


class GradeCodeTest(unittest.TestCase):
    def test_clean_patch_scores_like_config(self):
        spec = config_spec(
            category="code",
            allowed_paths=["harness2/benchmarks"],
            protected_paths=["tests"],
        )
        metrics = config_metrics(changed_files=["harness2/benchmarks/srv_runtime.py"])
        result = srv_grade.grade(metrics, spec)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["detail"]["violation"], "")

    def test_protected_path_violation_scores_zero(self):
        spec = config_spec(
            category="code",
            allowed_paths=["harness2/benchmarks", "tests"],
            protected_paths=["tests"],
        )
        metrics = config_metrics(changed_files=["tests/harness2/test_srv_pure.py"])
        result = srv_grade.grade(metrics, spec)
        self.assertIsNotNone(result)
        self.assertFalse(result["passed"])
        self.assertEqual(result["score"], 0.0)
        self.assertIn("tests/harness2/test_srv_pure.py", result["detail"]["violation"])

    def test_patch_outside_allowed_paths_scores_zero(self):
        spec = config_spec(category="code", allowed_paths=["vllm/ascend"])
        metrics = config_metrics(changed_files=["vllm/other.py"])
        result = srv_grade.grade(metrics, spec)
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["passed"])

    def test_clean_patch_cannot_score_an_uncalibrated_task(self):
        spec = config_spec(
            category="code",
            baseline=dict(NULL_BASELINE),
            allowed_paths=["vllm/ascend"],
        )
        metrics = config_metrics(changed_files=["vllm/ascend/runtime.py"])
        self.assertIsNone(srv_grade.grade(metrics, spec))

    def test_missing_changed_files_is_unscored(self):
        # CONTRACT 9/D8: no runtime patch report -> unscored, never a violation.
        spec = config_spec(category="code", allowed_paths=["vllm/ascend"])
        result = srv_grade.grade(config_metrics(), spec)
        self.assertIsNone(result)

    def test_forbidden_paths_alone_must_bind(self):
        # CONTRACT 9/D7: C's kod-* tasks declare only forbidden_paths.
        spec = config_spec(category="code", forbidden_paths=["tests", "benchmarks", "scripts"])
        metrics = config_metrics(changed_files=["tests/harness2/test_srv_pure.py"])
        result = srv_grade.grade(metrics, spec)
        self.assertIsNotNone(result)
        self.assertFalse(result["passed"])
        self.assertEqual(result["score"], 0.0)
        self.assertIn("forbidden path", result["detail"]["violation"])
        self.assertEqual(result["detail"]["failed"], ["patch_paths"])

    def test_protected_and_forbidden_paths_are_a_union(self):
        spec = config_spec(
            category="code", protected_paths=["docs"], forbidden_paths=["tests"]
        )
        for path in ("docs/srv.md", "tests/harness2/test_srv_pure.py"):
            metrics = config_metrics(changed_files=[path])
            result = srv_grade.grade(metrics, spec)
            self.assertEqual(result["score"], 0.0, path)
            self.assertIn("forbidden path", result["detail"]["violation"])

    def test_clean_patch_with_forbidden_paths_only(self):
        spec = config_spec(category="code", forbidden_paths=["tests"])
        metrics = config_metrics(changed_files=["vllm/ascend/runtime.py"])
        result = srv_grade.grade(metrics, spec)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["detail"]["violation"], "")


class GradeMeasureTest(unittest.TestCase):
    def assert_only_failed(self, result, name):
        self.assertIsNotNone(result)
        self.assertFalse(result["passed"])
        self.assertEqual(result["detail"]["failed"], [name])
        self.assertAlmostEqual(result["score"], 0.8)
        details = {check["name"]: check for check in result["detail"]["checks"]}
        self.assertEqual(sorted(details), sorted(srv_grade.MEASURE_CHECKS))
        self.assertTrue(details[name]["detail"])

    def test_all_five_checks_pass(self):
        result = srv_grade.grade(measure_metrics(), measure_spec())
        self.assertIsNotNone(result)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["score"], 1.0)
        names = [check["name"] for check in result["detail"]["checks"]]
        self.assertEqual(names, list(srv_grade.MEASURE_CHECKS))

    def test_service_not_up(self):
        metrics = measure_metrics()
        metrics["service"]["ok"] = False
        self.assert_only_failed(
            srv_grade.grade(metrics, measure_spec()), "service_up"
        )

    def test_aisbench_cmd_wrong_mode(self):
        metrics = measure_metrics()
        metrics["scenarios"]["4k2k"]["aisbench_cmd"] = (
            "ais_bench --models vllm_api_stream_chat --datasets synthetic_gen"
            " --mode accuracy"
        )
        self.assert_only_failed(
            srv_grade.grade(metrics, measure_spec()), "aisbench_cmd"
        )

    def test_aisbench_cmd_missing_dataset(self):
        metrics = measure_metrics()
        metrics["scenarios"]["4k2k"]["aisbench_cmd"] = (
            "ais_bench --models vllm_api_stream_chat --datasets random_gen --mode perf"
        )
        self.assert_only_failed(
            srv_grade.grade(metrics, measure_spec()), "aisbench_cmd"
        )

    def test_report_unparseable(self):
        metrics = measure_metrics()
        metrics["scenarios"]["4k2k"]["raw"] = ""
        self.assert_only_failed(
            srv_grade.grade(metrics, measure_spec()), "report_parsed"
        )

    def test_missing_required_scenario(self):
        metrics = measure_metrics()
        del metrics["scenarios"]["64k1k"]
        self.assert_only_failed(
            srv_grade.grade(metrics, measure_spec()), "scenarios_coverage"
        )

    def test_skipped_without_reason(self):
        metrics = measure_metrics()
        metrics["scenarios"]["64k1k"] = {"skipped": True, "reason": ""}
        self.assert_only_failed(
            srv_grade.grade(metrics, measure_spec()), "scenarios_coverage"
        )

    def test_skipping_a_measurable_scenario(self):
        spec = measure_spec(scenarios=["4k2k", "16k1k"])
        metrics = measure_metrics()
        metrics["scenarios"] = {
            "4k2k": {"skipped": True, "reason": "exceeds model context"},
            "16k1k": scenario_block(),
        }
        self.assert_only_failed(srv_grade.grade(metrics, spec), "scenarios_coverage")

    def test_missing_deliverable(self):
        metrics = measure_metrics(
            deliverables={"metrics.json": True, "run.sh": True, "report.md": False}
        )
        self.assert_only_failed(
            srv_grade.grade(metrics, measure_spec()), "deliverables"
        )

    def test_missing_deliverables_key_is_unscored(self):
        # CONTRACT 9/D8: a missing key is a runtime gap, not a missing delivery.
        metrics = measure_metrics()
        del metrics["deliverables"]
        result = srv_grade.grade(metrics, measure_spec())
        self.assertIsNone(result)
        self.assertFalse(result is not None and result["passed"] is False)

    def test_numbers_are_not_compared(self):
        # measure grades process, not numbers: a tiny throughput still passes.
        spec = measure_spec(baseline={"throughput_req_s": 999.0})
        metrics = measure_metrics()
        metrics["scenarios"]["4k2k"]["throughput_req_s"] = 0.01
        result = srv_grade.grade(metrics, spec)
        self.assertTrue(result["passed"], result)

    def test_empty_thresholds_are_legal(self):
        # CONTRACT 8/D1: measure tasks ship thresholds {} and must still grade.
        spec = measure_spec(thresholds={})
        self.assertEqual(spec["thresholds"], {})
        result = srv_grade.grade(measure_metrics(), spec)
        self.assertTrue(result["passed"], result)

    def test_thresholds_are_ignored_for_measure(self):
        # A configured accuracy threshold cannot turn measure into a numeric check.
        spec = measure_spec(
            thresholds={"accuracy": {"kind": "not_below", "tolerance_pct": 0.5}}
        )
        result = srv_grade.grade(measure_metrics(), spec)
        self.assertTrue(result["passed"], result)

    def test_measure_requires_accuracy_cmd_when_declared(self):
        spec = measure_spec(accuracy=True)
        metrics = measure_metrics()
        metrics["accuracy"] = {"ok": True, "accuracy": 82.5, "cmd": ACC_CMD,
                               "raw": "artifacts/aisbench_accuracy.json", "error": ""}
        result = srv_grade.grade(metrics, spec)
        self.assertTrue(result["passed"], result)
        metrics["accuracy"]["cmd"] = "ais_bench --datasets gsm8k_gen_0_shot_cot_chat_prompt"
        result = srv_grade.grade(metrics, spec)
        self.assertIn("aisbench_cmd", result["detail"]["failed"])
        del metrics["accuracy"]
        result = srv_grade.grade(metrics, spec)
        self.assertIn("aisbench_cmd", result["detail"]["failed"])

    def test_measure_ok_false_is_unscored(self):
        self.assertIsNone(srv_grade.grade(measure_metrics(ok=False), measure_spec()))


class GradePurityTest(unittest.TestCase):
    def test_grade_reads_no_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            before = sorted(workdir.iterdir())
            result = srv_grade.grade(measure_metrics(), measure_spec())
            self.assertTrue(result["passed"])
            self.assertEqual(before, sorted(workdir.iterdir()))


class RuntimeFailurePathTest(unittest.TestCase):
    """No NPU/docker here: every container-facing call must fail honestly."""

    def test_service_up_rejects_bad_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = srv_runtime.service_up(
                "mirror", "4k2k", container="srv-dead", cfg={}, log_dir=Path(tmp)
            )
        self.assertFalse(result["ok"])
        self.assertIn("mirror", result["error"])
        self.assertEqual(result["p_cmd"], "")
        self.assertEqual(result["prefix_cache"], {"p": "100%", "d": "off"})

    def test_service_up_rejects_unknown_scenario(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = srv_runtime.service_up(
                "single", "nope", container="srv-dead", cfg={}, log_dir=Path(tmp)
            )
        self.assertFalse(result["ok"])
        self.assertIn("nope", result["error"])

    def test_service_up_rejects_scenario_over_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = srv_runtime.service_up(
                "single", "64k1k", container="srv-dead", cfg={}, log_dir=Path(tmp)
            )
        self.assertFalse(result["ok"])
        self.assertIn("32768", result["error"])
        self.assertIn("applicable_scenarios", result["error"])

    def test_single_mode_leaves_d_cmd_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = srv_runtime.service_up(
                "single", "4k2k", container="srv-dead", cfg={}, log_dir=Path(tmp)
            )
        self.assertIsNone(result["d_cmd"])
        self.assertIn("--enable-prefix-caching", result["p_cmd"])
        self.assertEqual(result["prefix_cache"], {"p": "100%", "d": "off"})

    def test_pd_mode_builds_two_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = srv_runtime.service_up(
                "pd", "16k1k", container="srv-dead", cfg={}, log_dir=Path(tmp)
            )
        self.assertIn("--enable-prefix-caching", result["p_cmd"])
        self.assertIn("kv_producer", result["p_cmd"])
        self.assertIn("--no-enable-prefix-caching", result["d_cmd"])
        self.assertIn("kv_consumer", result["d_cmd"])
        self.assertEqual(result["prefix_cache"], {"p": "100%", "d": "off"})
        self.assertIn("--port 8001", result["d_cmd"])

    def test_max_model_len_follows_scenario(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = srv_runtime.service_up(
                "single", "4k2k", container="srv-dead", cfg={}, log_dir=Path(tmp)
            )
        self.assertIn("--max-model-len 6144", result["p_cmd"])

    def test_aisbench_perf_records_cmd_and_overrides_output_len(self):
        plain = srv_runtime.aisbench_perf("4k2k", container="srv-dead")
        self.assertFalse(plain["ok"])
        self.assertIn("--mode perf", plain["cmd"])
        self.assertIn("--max-output-len 2048", plain["cmd"])
        self.assertIsNone(plain["throughput_req_s"])
        pure_p = srv_runtime.aisbench_perf("4k2k", container="srv-dead", output_len=1)
        self.assertIn("--max-output-len 1", pure_p["cmd"])
        self.assertFalse(pure_p["ok"])

    def test_aisbench_perf_rejects_unknown_scenario(self):
        result = srv_runtime.aisbench_perf("nope", container="srv-dead")
        self.assertFalse(result["ok"])
        self.assertIn("nope", result["error"])
        self.assertEqual(result["cmd"], "")

    def test_aisbench_accuracy_records_dataset_and_mode(self):
        result = srv_runtime.aisbench_accuracy(container="srv-dead")
        self.assertFalse(result["ok"])
        self.assertIn("gsm8k_gen_0_shot_cot_chat_prompt", result["cmd"])
        self.assertIn("--mode accuracy", result["cmd"])
        self.assertIsNone(result["accuracy"])

    def test_health_wait_times_out_without_hanging(self):
        with self.assertRaises(srv_runtime.SrvError) as caught:
            srv_runtime._wait_healthy(
                "srv-dead",
                [{"tag": "p", "port": 1, "rc": "/tmp/srv/4k2k.p.rc"}],
                host="127.0.0.1",
                timeout=0.5,
                interval=0.05,
            )
        self.assertIn("/health not ready", str(caught.exception))
        self.assertLessEqual(srv_runtime.MAX_HEALTH_TIMEOUT_S, 600.0)

    def test_container_stop_is_quiet(self):
        self.assertIsNone(srv_runtime.container_stop("srv-does-not-exist"))

    def test_collect_fills_deliverables_by_scanning_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            spec = config_spec(scenarios=["4k2k"])
            srv_runtime.collect("srv-dead", Path(tmp), run_dir, spec)
            written = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(written["deliverables"], {"metrics.json": True})
        self.assertNotIn("changed_files", written)

    def test_collect_fills_changed_files_for_code_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            (run_dir / "deliverables").mkdir(parents=True)
            (run_dir / "deliverables" / "patch.diff").write_text(DIFF, encoding="utf-8")
            spec = config_spec(category="code", forbidden_paths=["tests"])
            srv_runtime.collect("srv-dead", Path(tmp), run_dir, spec)
            written = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(
            written["changed_files"],
            ["vllm/ascend/foo.py", "tests/harness2/test_srv_pure.py"],
        )

    def test_collect_writes_metrics_json_on_service_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            spec = config_spec(scenarios=["4k2k"])
            metrics = srv_runtime.collect("srv-dead", Path(tmp), run_dir, spec)
            self.assertFalse(metrics["ok"])
            self.assertIn("service did not start", metrics["error"])
            written = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertFalse(written["ok"])
            self.assertTrue(written["scenarios"]["4k2k"]["skipped"])
            self.assertTrue(written["scenarios"]["4k2k"]["reason"])
            self.assertIsNone(srv_grade.grade(written, spec))


class DeliveryScanTest(unittest.TestCase):
    def test_scan_walks_run_dir_and_deliverables_subdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "deliverables").mkdir()
            (run / "deliverables" / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            (run / "deliverables" / "report.md").write_text("# r\n", encoding="utf-8")
            found = srv_runtime._scan_deliverables(
                {"deliverables": ["metrics.json", "run.sh", "report.md"]}, run
            )
        self.assertEqual(
            found, {"metrics.json": False, "run.sh": True, "report.md": True}
        )

    def test_changed_files_are_parsed_from_the_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "deliverables").mkdir()
            (run / "deliverables" / "patch.diff").write_text(DIFF, encoding="utf-8")
            files = srv_runtime._changed_files_from_patch({}, run)
        self.assertEqual(
            files, ["vllm/ascend/foo.py", "tests/harness2/test_srv_pure.py"]
        )

    def test_patch_parsing_ignores_dev_null_and_duplicates(self):
        parsed = srv_runtime._parse_patch_paths(
            "--- /dev/null\n+++ b/a/b.py\n--- a/a/b.py\n+++ b/a/b.py\n"
        )
        self.assertEqual(parsed, ["a/b.py"])

    def test_no_patch_file_reports_no_changed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(srv_runtime._changed_files_from_patch({}, Path(tmp)), [])


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


if __name__ == "__main__":
    unittest.main()
