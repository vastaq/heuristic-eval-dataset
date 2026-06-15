import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "eval_datasets" / "scripts"


def run_script(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class ScriptTests(unittest.TestCase):
    def test_export_preserves_legacy_asserts_and_filters_to_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset.json"
            output = Path(tmp) / "tests.yaml"
            write_json(
                dataset,
                {
                    "version": "v1",
                    "project": "demo",
                    "dataset_type": "character_prompt_eval",
                    "records": [
                        {
                            "id": "demo_accepted_001",
                            "layer": "core_smoke",
                            "role": "demo",
                            "character_context": "file://prompt.md",
                            "scene_type": "identity",
                            "difficulty": "easy",
                            "input": "你是谁？",
                            "target_behavior": ["直接回答身份"],
                            "avoid_behavior": ["回避身份"],
                            "tags": ["identity"],
                            "rubric_ref": "identity",
                            "source_path": "demo/test.yaml",
                            "review_status": "accepted",
                            "legacy_asserts": [
                                {"type": "llm-rubric", "value": "直接回答身份"},
                                {"type": "icontains", "value": "Demo"},
                            ],
                        },
                        {
                            "id": "demo_candidate_001",
                            "layer": "core_smoke",
                            "role": "demo",
                            "character_context": "file://prompt.md",
                            "scene_type": "identity",
                            "difficulty": "easy",
                            "input": "候选样例",
                            "target_behavior": ["可判断"],
                            "avoid_behavior": ["不可判断"],
                            "tags": ["candidate"],
                            "rubric_ref": "identity",
                            "source_path": "demo/test.yaml",
                            "review_status": "candidate",
                        },
                    ],
                },
            )

            result = run_script("export_promptfoo_tests.py", str(dataset), str(output))

            self.assertEqual(result.returncode, 0, result.stderr)
            exported = output.read_text(encoding="utf-8")
            self.assertIn("demo_accepted_001", exported)
            self.assertNotIn("demo_candidate_001", exported)
            self.assertIn("source_path: demo/test.yaml", exported)
            self.assertIn("type: icontains", exported)
            self.assertIn("value: Demo", exported)

    def test_import_promptfoo_yaml_preserves_metadata_and_asserts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "legacy.yaml"
            output = Path(tmp) / "seed.json"
            source.write_text(
                """
- vars:
    character_context: file://prompt.md
    question: "Slowpoke, who are you?"
  metadata:
    id: slowpoke_identity_001
    layer: core_smoke
    role: slowpoke
    scene_type: boundary_and_identity
    difficulty: easy
    tags: [identity, child]
  assert:
    - type: llm-rubric
      value: "The reply should answer in first person."
""".strip(),
                encoding="utf-8",
            )

            result = run_script(
                "import_promptfoo_tests.py",
                str(source),
                str(output),
                "--project",
                "slowpoke",
                "--role",
                "slowpoke",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            imported = json.loads(output.read_text(encoding="utf-8"))
            record = imported["records"][0]
            self.assertEqual(record["id"], "slowpoke_identity_001")
            self.assertEqual(record["input"], "Slowpoke, who are you?")
            self.assertTrue(record["source_path"].endswith("legacy.yaml"))
            self.assertEqual(record["legacy_asserts"][0]["type"], "llm-rubric")
            self.assertEqual(record["review_status"], "candidate")

    def test_import_and_export_preserve_input_var_and_extra_vars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "legacy.yaml"
            seed = Path(tmp) / "seed.json"
            exported = Path(tmp) / "exported.yaml"
            source.write_text(
                """
- vars:
    user_input: "你是谁呀？"
    behaviour_context: file://shared.md
    character_context: file://prompt.md
  metadata:
    conversationId: c1
  assert:
    - type: icontains
      value: 奥特曼
- vars:
    user_input: "下一句"
  metadata:
    conversationId: c1
  assert:
    - type: llm-rubric
      value: "应该接住上文"
""".strip(),
                encoding="utf-8",
            )

            imported = run_script(
                "import_promptfoo_tests.py",
                str(source),
                str(seed),
                "--project",
                "dou",
                "--role",
                "ultraman",
            )
            self.assertEqual(imported.returncode, 0, imported.stderr)
            payload = json.loads(seed.read_text(encoding="utf-8"))
            first, second = payload["records"]
            self.assertEqual(first["input_var"], "user_input")
            self.assertEqual(first["vars"]["behaviour_context"], "file://shared.md")
            self.assertEqual(first["turn"], 1)
            self.assertEqual(second["turn"], 2)

            for record in payload["records"]:
                record["review_status"] = "accepted"
            write_json(seed, payload)

            result = run_script("export_promptfoo_tests.py", str(seed), str(exported))
            self.assertEqual(result.returncode, 0, result.stderr)
            text = exported.read_text(encoding="utf-8")
            self.assertIn("user_input: 你是谁呀？", text)
            self.assertIn("behaviour_context: file://shared.md", text)
            self.assertNotIn("question: 你是谁呀？", text)

    def test_audit_strict_flags_duplicate_ids_and_concentrated_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset.json"
            base_record = {
                "id": "dup",
                "layer": "core_smoke",
                "role": "demo",
                "character_context": "file://prompt.md",
                "scene_type": "identity",
                "difficulty": "easy",
                "input": "hello",
                "target_behavior": ["answer"],
                "avoid_behavior": ["ignore"],
                "tags": ["identity"],
                "rubric_ref": "identity",
                "source_path": "demo/test.yaml",
                "review_status": "accepted",
            }
            write_json(
                dataset,
                {
                    "version": "v1",
                    "project": "demo",
                    "dataset_type": "character_prompt_eval",
                    "records": [base_record, dict(base_record)],
                },
            )

            result = run_script("audit_testset_balance.py", str(dataset), "--strict")

            self.assertEqual(result.returncode, 1)
            self.assertIn("duplicate id: dup", result.stdout)
            self.assertIn("tag 'identity' appears in 2/2", result.stdout)

    def test_batch_import_discovers_multiple_test_yaml_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "role_a").mkdir()
            (root / "role_b").mkdir()
            (root / "role_empty").mkdir()
            (root / "role_a" / "test_daily.yaml").write_text(
                """
- vars:
    question: "hello"
  assert:
    - type: llm-rubric
      value: "answer"
""".strip(),
                encoding="utf-8",
            )
            (root / "role_b" / "test.yaml").write_text(
                """
- vars:
    user_input: "你好"
  assert:
    - type: icontains
      value: 你好
""".strip(),
                encoding="utf-8",
            )
            (root / "role_empty" / "test_empty.yaml").write_text("", encoding="utf-8")
            output = root / "all.json"

            result = run_script(
                "batch_import_test_yaml.py",
                str(root),
                str(output),
                "--project",
                "all_existing_roles",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["source_paths"]), 3)
            self.assertEqual(len(payload["records"]), 2)
            self.assertEqual({record["role"] for record in payload["records"]}, {"role_a", "role_b"})

    def test_init_eval_run_creates_thin_intake_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"

            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(output),
                "--source-root",
                "local/out",
                "--source-file",
                "summary=local/out/summary.json",
                "--controlled-variable",
                "season",
                "--content-unit",
                "forest",
                "--case-count",
                "4",
                "--success-count",
                "3",
                "--failure-count",
                "1",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_id"], "demo_run")
            self.assertEqual(manifest["project"], "demo")
            self.assertEqual(manifest["source_artifact_root"], "local/out")
            self.assertEqual(manifest["source_files"], {"summary": "local/out/summary.json"})
            self.assertEqual(manifest["source_files"]["summary"], "local/out/summary.json")
            self.assertEqual(manifest["controlled_variables"], ["season"])
            self.assertEqual(manifest["content_units"], ["forest"])
            self.assertEqual(manifest["case_count"], 4)
            self.assertEqual(manifest["success_count"], 3)
            self.assertEqual(manifest["failure_count"], 1)
            self.assertEqual(decision["decision_id"], "demo_run_decision")
            self.assertEqual(decision["decision_type"], "needs_decision")
            self.assertIsNone(decision["accepted_direction"])
            self.assertEqual((output / "human_signals.jsonl").read_text(encoding="utf-8"), "")

    def test_init_eval_run_marks_accept_direction_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"

            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(output),
                "--decision-type",
                "accept_direction",
                "--accepted-direction",
                "true",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
            self.assertEqual(decision["decision_type"], "accept_direction")
            self.assertIs(decision["accepted_direction"], True)

    def test_init_eval_run_does_not_infer_acceptance_from_decision_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"

            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(output),
                "--decision-type",
                "accept_direction",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
            self.assertEqual(decision["decision_type"], "accept_direction")
            self.assertIsNone(decision["accepted_direction"])

    def test_init_eval_run_without_source_root_does_not_write_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"

            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(output),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_artifact_root"], "")
            self.assertNotIn("path/to", json.dumps(manifest))

    def test_init_eval_run_rejects_empty_source_file_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"

            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(output),
                "--source-file",
                "review=",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("empty source-file value", result.stderr)

    def test_init_eval_run_supports_profile_without_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"

            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "conversation_run",
                "--project",
                "demo",
                "--profile",
                "conversation_role",
                "--adapter",
                "promptfoo",
                "--output-dir",
                str(output),
                "--source-file",
                "observations=local/observations.json",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["profile"], "conversation_role")
            self.assertEqual(manifest["adapter"], "promptfoo")
            self.assertEqual(manifest["source_files"], {"observations": "local/observations.json"})
            self.assertEqual(decision["decision_type"], "needs_decision")
            self.assertIsNone(decision["accepted_direction"])
            self.assertFalse(decision["dataset_generation"]["needed"])

    def test_record_human_signal_appends_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_stop_rule",
                "--raw-signal",
                "Good enough; do not tune two narrow cases.",
                "--context-ref",
                "review.json#case-2",
                "--classification",
                "acceptable_variance",
                "--tag",
                "prompt_patch_pressure",
                "--suggested-outcome",
                "stop_tuning",
                "--blocked-action",
                "continue_prompt_tuning_for_narrow_failures",
                "--needs-review",
                "false",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            lines = [
                line
                for line in (run_dir / "human_signals.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(lines), 1)
            signal = json.loads(lines[0])
            self.assertEqual(signal["signal_type"], "human_stop_rule")
            self.assertEqual(signal["raw_signal"], "Good enough; do not tune two narrow cases.")
            self.assertEqual(signal["suggested_outcome"], "stop_tuning")
            self.assertEqual(signal["candidate_failure_tags"], ["prompt_patch_pressure"])
            self.assertEqual(signal["blocked_actions"], ["continue_prompt_tuning_for_narrow_failures"])
            self.assertFalse(signal["needs_review"])
            self.assertEqual(signal["source_type"], "user")

    def test_record_human_signal_allows_raw_capture_without_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_judgment",
                "--raw-signal",
                "Something feels off; classify later.",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            signal = json.loads((run_dir / "human_signals.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(signal["suggested_outcome"], "needs_review")
            self.assertTrue(signal["needs_review"])

    def test_record_human_signal_rejects_empty_raw_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_judgment",
                "--raw-signal",
                "   ",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("raw-signal must be non-empty", result.stderr)
            self.assertEqual((run_dir / "human_signals.jsonl").read_text(encoding="utf-8"), "")

    def test_record_human_signal_rejects_directory_without_run_intake(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "not_a_run"
            run_dir.mkdir()

            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_judgment",
                "--raw-signal",
                "This should not be detached from a run intake.",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("run directory must contain manifest.json", result.stderr)
            self.assertFalse((run_dir / "human_signals.jsonl").exists())

    def test_validate_run_intake_rejects_empty_human_signal_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            (run_dir / "human_signals.jsonl").write_text(
                json.dumps(
                    {
                        "signal_type": "human_judgment",
                        "raw_signal": "",
                        "candidate_failure_tags": [],
                        "suggested_outcome": "needs_review",
                        "blocked_actions": [],
                        "needs_review": True,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_script("validate_run_intake.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("human_signals[1].raw_signal must be a non-empty string", result.stderr)

    def test_init_profile_adapter_creates_minimum_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "eval_datasets"

            result = run_script(
                "init_profile_adapter.py",
                "--profile-id",
                "tool_use_eval",
                "--adapter-id",
                "custom_tool_runner",
                "--domain",
                "tool-use agent evaluation",
                "--evaluator",
                "custom JSON runner",
                "--output-root",
                str(output_root),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            profile = (output_root / "profiles" / "tool_use_eval" / "README.md").read_text(
                encoding="utf-8"
            )
            adapter = (output_root / "adapters" / "custom_tool_runner" / "README.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("tool-use agent evaluation", profile)
            self.assertIn("Domain Boundary", profile)
            self.assertIn("Required Fields", profile)
            self.assertIn("List Fields", profile)
            self.assertIn("Quality Signals", profile)
            self.assertIn("Acceptable Band", profile)
            self.assertIn("When Run Intake Is Enough", profile)
            self.assertIn("custom JSON runner", adapter)
            self.assertIn("Source Files", adapter)
            self.assertIn("Result Normalization", adapter)
            self.assertIn("Structural Diagnostics", adapter)
            self.assertIn("Human Signals", adapter)
            self.assertIn("Blocked Actions", adapter)
            self.assertNotIn("conversation_role", profile)
            self.assertIn("draft scaffold", result.stdout.lower())
            self.assertIn("not calibrated", result.stdout.lower())
            self.assertIn("validate_run_intake.py --validate-module-notes", result.stdout)

    def test_init_profile_adapter_does_not_overwrite_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "eval_datasets"
            profile_path = output_root / "profiles" / "tool_use_eval" / "README.md"
            profile_path.parent.mkdir(parents=True)
            profile_path.write_text("# Custom Profile\n", encoding="utf-8")

            result = run_script(
                "init_profile_adapter.py",
                "--profile-id",
                "tool_use_eval",
                "--adapter-id",
                "custom_tool_runner",
                "--domain",
                "tool-use agent evaluation",
                "--evaluator",
                "custom JSON runner",
                "--output-root",
                str(output_root),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(profile_path.read_text(encoding="utf-8"), "# Custom Profile\n")
            self.assertTrue((output_root / "adapters" / "custom_tool_runner" / "README.md").exists())
            self.assertIn("kept existing profile note", result.stdout)
            self.assertIn("created adapter note", result.stdout)

    def test_init_profile_adapter_rejects_path_escape_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "eval_datasets"

            result = run_script(
                "init_profile_adapter.py",
                "--profile-id",
                "../escape",
                "--adapter-id",
                "custom_tool_runner",
                "--domain",
                "tool-use agent evaluation",
                "--evaluator",
                "custom JSON runner",
                "--output-root",
                str(output_root),
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("profile-id must match", result.stderr)
            self.assertFalse((Path(tmp) / "escape").exists())

    def test_record_human_signal_rejects_invalid_needs_review_boolean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_judgment",
                "--raw-signal",
                "Metric is misleading.",
                "--suggested-outcome",
                "downgrade_metric_to_diagnostic",
                "--needs-review",
                "maybe",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected true or false", result.stderr)
            self.assertEqual((run_dir / "human_signals.jsonl").read_text(encoding="utf-8"), "")

    def test_validate_run_intake_accepts_traceable_run_with_human_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "review=local/review.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_judgment",
                "--raw-signal",
                "Metric is misleading; keep it diagnostic.",
                "--suggested-outcome",
                "downgrade_metric_to_diagnostic",
                "--blocked-action",
                "optimize_prompt_for_keyword_coverage",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["optimize_prompt_for_keyword_coverage"]
            decision["next_actions"] = ["review diagnostic-only metrics before prompt changes"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_run_intake.py", str(run_dir))

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("run intake valid", result.stdout)

    def test_validate_run_intake_requires_referenced_human_signal_next_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "review=local/review.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_strategy_signal",
                "--raw-signal",
                "Classify the pattern before touching the prompt.",
                "--suggested-outcome",
                "failure_pattern_candidate",
                "--next-action",
                "classify repeated failure before prompt changes",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["prompt_patch_without_replay"]
            decision["next_actions"] = ["run prompt mutation experiment"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_run_intake.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "decision.next_actions must include referenced human signal next_action",
                result.stderr,
            )

    def test_validate_run_intake_requires_referenced_human_signal_blocked_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "review=local/review.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_strategy_signal",
                "--raw-signal",
                "Do not mutate the prompt from this narrow failure.",
                "--suggested-outcome",
                "blocked_prompt_mutation",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = []
            decision["next_actions"] = ["classify narrow failure before prompt changes"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_run_intake.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "decision.blocked_actions must include referenced human signal blocked_action",
                result.stderr,
            )

    def test_validate_run_intake_rejects_missing_next_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision.pop("next_actions")
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_run_intake.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("decision.next_actions must be a list", result.stderr)

    def test_validate_run_intake_rejects_empty_next_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "review=local/review.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["next_actions"] = []
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_run_intake.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("decision.next_actions must include at least one non-empty action", result.stderr)

    def test_validate_run_intake_rejects_agent_inference_marked_reviewed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "agent_signal_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            (run_dir / "human_signals.jsonl").write_text(
                json.dumps(
                    {
                        "signal_type": "human_judgment",
                        "raw_signal": "Agent inferred the user would accept this.",
                        "candidate_failure_tags": [],
                        "suggested_outcome": "accept_direction",
                        "blocked_actions": [],
                        "needs_review": False,
                        "source_type": "agent_inference",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["next_actions"] = ["keep agent inference under review"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_run_intake.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "human_signals[1].source_type cannot be agent_inference when needs_review=false",
                result.stderr,
            )

    def test_validate_run_intake_rejects_reviewed_signal_without_source_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "missing_source_signal_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            (run_dir / "human_signals.jsonl").write_text(
                json.dumps(
                    {
                        "signal_type": "human_judgment",
                        "raw_signal": "This should stop tuning.",
                        "candidate_failure_tags": [],
                        "suggested_outcome": "stop_tuning",
                        "blocked_actions": ["continue_prompt_tuning_for_narrow_failures"],
                        "needs_review": False,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["next_actions"] = ["keep signal under review until source is known"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_run_intake.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "human_signals[1].source_type is required when needs_review=false",
                result.stderr,
            )

    def test_validate_run_intake_rejects_allowed_action_that_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "review=local/review.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["blocked_actions"] = ["prompt_patch_without_replay"]
            decision["allowed_actions"] = ["prompt_patch_without_replay"]
            decision["next_actions"] = ["classify failure before prompt changes"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_run_intake.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("decision.allowed_actions repeats blocked action: prompt_patch_without_replay", result.stderr)

    def test_validate_run_intake_rejects_bad_human_signal_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_run_intake.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("human_signal_refs", result.stderr)

    def test_validate_run_intake_rejects_invalid_human_signal_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            (run_dir / "human_signals.jsonl").write_text("{bad json}\n", encoding="utf-8")

            result = run_script("validate_run_intake.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("human_signals.jsonl line 1 is not valid JSON", result.stderr)

    def test_validate_run_intake_rejects_manifest_decision_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["run_id"] = "other_run"
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_run_intake.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("manifest.run_id and decision.run_id must match", result.stderr)

    def test_validate_run_intake_rejects_placeholder_next_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            result = run_script("validate_run_intake.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("decision.next_actions contains placeholder", result.stderr)

    def test_validate_run_intake_rejects_blocked_action_as_next_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["blocked_actions"] = ["prompt_patch_without_replay"]
            decision["next_actions"] = ["prompt_patch_without_replay"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_run_intake.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "decision.next_actions repeats blocked action: prompt_patch_without_replay",
                result.stderr,
            )

    def test_validate_run_intake_requires_module_refs_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "tool_run",
                "--project",
                "demo",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_tool_runner",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "tool_trace=local/tool-trace.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["next_actions"] = ["classify timeout recovery before changing tool policy"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script(
                "validate_run_intake.py",
                str(run_dir),
                "--require-module-refs",
                "--validate-module-notes",
                "--validate-adapter-boundaries",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("manifest.profile_ref is required", result.stderr)
            self.assertIn("manifest.adapter_ref is required", result.stderr)

    def test_validate_run_intake_accepts_existing_module_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "eval_datasets"
            scaffold = run_script(
                "init_profile_adapter.py",
                "--profile-id",
                "tool_use_eval",
                "--adapter-id",
                "custom_tool_runner",
                "--domain",
                "tool-use agent evaluation",
                "--evaluator",
                "custom JSON runner",
                "--output-root",
                str(output_root),
            )
            self.assertEqual(scaffold.returncode, 0, scaffold.stderr)
            profile_ref = output_root / "profiles" / "tool_use_eval" / "README.md"
            adapter_ref = output_root / "adapters" / "custom_tool_runner" / "README.md"
            run_dir = root / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "tool_run",
                "--project",
                "demo",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_tool_runner",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "tool_trace=local/tool-trace.json",
                "--profile-ref",
                str(profile_ref),
                "--adapter-ref",
                str(adapter_ref),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["next_actions"] = ["classify timeout recovery before changing tool policy"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_run_intake.py", str(run_dir), "--require-module-refs")

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["profile_ref"], str(profile_ref))
            self.assertEqual(manifest["adapter_ref"], str(adapter_ref))

    def test_validate_run_intake_rejects_unedited_scaffold_module_notes_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "eval_datasets"
            scaffold = run_script(
                "init_profile_adapter.py",
                "--profile-id",
                "tool_use_eval",
                "--adapter-id",
                "custom_tool_runner",
                "--domain",
                "tool-use agent evaluation",
                "--evaluator",
                "custom JSON runner",
                "--output-root",
                str(output_root),
            )
            self.assertEqual(scaffold.returncode, 0, scaffold.stderr)
            run_dir = root / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "tool_run",
                "--project",
                "demo",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_tool_runner",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "tool_trace=local/tool-trace.json",
                "--profile-ref",
                str(output_root / "profiles" / "tool_use_eval" / "README.md"),
                "--adapter-ref",
                str(output_root / "adapters" / "custom_tool_runner" / "README.md"),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["next_actions"] = ["classify timeout recovery before changing tool policy"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script(
                "validate_run_intake.py",
                str(run_dir),
                "--require-module-refs",
                "--validate-module-notes",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("manifest.profile_ref contains scaffold placeholder text", result.stderr)
            self.assertIn("manifest.adapter_ref contains scaffold placeholder text", result.stderr)

    def test_validate_run_intake_rejects_module_refs_for_wrong_profile_or_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_root = root / "eval_datasets"
            scaffold = run_script(
                "init_profile_adapter.py",
                "--profile-id",
                "other_eval",
                "--adapter-id",
                "other_runner",
                "--domain",
                "other evaluation",
                "--evaluator",
                "other runner",
                "--output-root",
                str(output_root),
            )
            self.assertEqual(scaffold.returncode, 0, scaffold.stderr)
            run_dir = root / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "tool_run",
                "--project",
                "demo",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_tool_runner",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "tool_trace=local/tool-trace.json",
                "--profile-ref",
                str(output_root / "profiles" / "other_eval" / "README.md"),
                "--adapter-ref",
                str(output_root / "adapters" / "other_runner" / "README.md"),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["next_actions"] = ["classify timeout recovery before changing tool policy"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_run_intake.py", str(run_dir), "--require-module-refs")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("manifest.profile_ref must point to profiles/tool_use_eval/README.md", result.stderr)
            self.assertIn("manifest.adapter_ref must point to adapters/custom_tool_runner/README.md", result.stderr)

    def test_validate_run_intake_rejects_module_refs_from_different_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_root = root / "profile_workspace" / "eval_datasets"
            adapter_root = root / "adapter_workspace" / "eval_datasets"
            for output_root in (profile_root, adapter_root):
                scaffold = run_script(
                    "init_profile_adapter.py",
                    "--profile-id",
                    "tool_use_eval",
                    "--adapter-id",
                    "custom_tool_runner",
                    "--domain",
                    "tool-use agent evaluation",
                    "--evaluator",
                    "custom JSON runner",
                    "--output-root",
                    str(output_root),
                )
                self.assertEqual(scaffold.returncode, 0, scaffold.stderr)

            run_dir = root / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "tool_run",
                "--project",
                "demo",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_tool_runner",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "tool_trace=local/tool-trace.json",
                "--profile-ref",
                str(profile_root / "profiles" / "tool_use_eval" / "README.md"),
                "--adapter-ref",
                str(adapter_root / "adapters" / "custom_tool_runner" / "README.md"),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["next_actions"] = ["classify timeout recovery before changing tool policy"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_run_intake.py", str(run_dir), "--require-module-refs")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "manifest.profile_ref and manifest.adapter_ref must share the same eval_datasets root",
                result.stderr,
            )

    def test_validate_run_intake_rejects_incomplete_module_notes_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_ref = root / "eval_datasets" / "profiles" / "tool_use_eval" / "README.md"
            adapter_ref = root / "eval_datasets" / "adapters" / "custom_tool_runner" / "README.md"
            profile_ref.parent.mkdir(parents=True)
            adapter_ref.parent.mkdir(parents=True)
            profile_ref.write_text("# Tool Use Eval\n\n## Domain Purpose\n\nTool tasks.\n", encoding="utf-8")
            adapter_ref.write_text("# Custom Tool Runner\n\n## Source Files\n\nRaw traces.\n", encoding="utf-8")
            run_dir = root / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "tool_run",
                "--project",
                "demo",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_tool_runner",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "tool_trace=local/tool-trace.json",
                "--profile-ref",
                str(profile_ref),
                "--adapter-ref",
                str(adapter_ref),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["next_actions"] = ["classify timeout recovery before changing tool policy"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script(
                "validate_run_intake.py",
                str(run_dir),
                "--require-module-refs",
                "--validate-module-notes",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("manifest.profile_ref missing module note section: Domain Boundary", result.stderr)
            self.assertIn("manifest.adapter_ref missing module note section: Result Normalization", result.stderr)

    def test_validate_run_intake_rejects_empty_module_note_sections_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_ref = root / "eval_datasets" / "profiles" / "tool_use_eval" / "README.md"
            adapter_ref = root / "eval_datasets" / "adapters" / "custom_tool_runner" / "README.md"
            profile_ref.parent.mkdir(parents=True)
            adapter_ref.parent.mkdir(parents=True)
            profile_ref.write_text(
                "\n".join(
                    [
                        "# Tool Use Eval",
                        "## Domain Purpose",
                        "## Domain Boundary",
                        "## Minimum Artifact Shape",
                        "## Required Fields",
                        "## Quality Signals And Rubric Vocabulary",
                        "## Acceptable Band, Stop Rule, And Bloat Guardrail",
                        "## When Run Intake Is Enough",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            adapter_ref.write_text(
                "\n".join(
                    [
                        "# Custom Tool Runner",
                        "## Source Files",
                        "## Result Normalization",
                        "## Structural Diagnostics",
                        "## Human Signals",
                        "## Blocked Actions",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            run_dir = root / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "tool_run",
                "--project",
                "demo",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_tool_runner",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "tool_trace=local/tool-trace.json",
                "--profile-ref",
                str(profile_ref),
                "--adapter-ref",
                str(adapter_ref),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["next_actions"] = ["classify timeout recovery before changing tool policy"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script(
                "validate_run_intake.py",
                str(run_dir),
                "--require-module-refs",
                "--validate-module-notes",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("manifest.profile_ref empty module note section: Domain Purpose", result.stderr)
            self.assertIn("manifest.adapter_ref empty module note section: Source Files", result.stderr)

    def test_validate_run_intake_rejects_module_notes_with_wrong_domain_terms_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_ref = root / "eval_datasets" / "profiles" / "tool_use_eval" / "README.md"
            adapter_ref = root / "eval_datasets" / "adapters" / "custom_tool_runner" / "README.md"
            profile_ref.parent.mkdir(parents=True)
            adapter_ref.parent.mkdir(parents=True)
            profile_ref.write_text(
                """
# Tool Use Eval

## Domain Purpose

Evaluate tool-use traces, but this copy still talks about character_context.

## Domain Boundary

Tool tasks only.

## Minimum Artifact Shape

Trace, outcome, and evidence.

## Required Fields

- character_context

## Quality Signals And Rubric Vocabulary

Tool success and recovery quality.

## Acceptable Band, Stop Rule, And Bloat Guardrail

Stop when the error band is acceptable.

## When Run Intake Is Enough

Use run intake for one-off traces.
""".strip()
                + "\n",
                encoding="utf-8",
            )
            adapter_ref.write_text(
                """
# Custom Tool Runner

## Source Files

Raw traces.

## Result Normalization

Run normalize_promptfoo_results.py first.

## Structural Diagnostics

Timeouts and retries.

## Human Signals

Preserve human review.

## Blocked Actions

Do not mutate policy before classification.
""".strip()
                + "\n",
                encoding="utf-8",
            )
            run_dir = root / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "tool_run",
                "--project",
                "demo",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_tool_runner",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "tool_trace=local/tool-trace.json",
                "--profile-ref",
                str(profile_ref),
                "--adapter-ref",
                str(adapter_ref),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["next_actions"] = ["classify timeout recovery before changing tool policy"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script(
                "validate_run_intake.py",
                str(run_dir),
                "--require-module-refs",
                "--validate-module-notes",
                "--validate-adapter-boundaries",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "manifest.profile_ref contains conversation-role term outside conversation_role profile: "
                "character_context",
                result.stderr,
            )
            self.assertIn(
                "manifest.adapter_ref contains Promptfoo-specific term for non-promptfoo adapter: "
                "normalize_promptfoo_results.py",
                result.stderr,
            )

    def test_validate_run_intake_rejects_promptfoo_artifacts_for_non_promptfoo_adapter_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "tool_run",
                "--project",
                "demo",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_tool_runner",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "promptfoo_results=local/promptfoo-results.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["next_actions"] = ["run normalize_promptfoo_results.py before changing tool policy"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_run_intake.py", str(run_dir), "--validate-adapter-boundaries")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "non-promptfoo adapter must not reference Promptfoo-specific artifacts",
                result.stderr,
            )

    def test_validate_run_intake_rejects_role_fields_in_non_conversation_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_ref = root / "eval_datasets" / "profiles" / "tool_use_eval" / "README.md"
            adapter_ref = root / "eval_datasets" / "adapters" / "custom_tool_runner" / "README.md"
            profile_ref.parent.mkdir(parents=True)
            adapter_ref.parent.mkdir(parents=True)
            profile_ref.write_text(
                """
# Tool Use Eval

## Domain Purpose

Evaluate tool-use traces.

## Domain Boundary

Tool tasks only.

## Minimum Artifact Shape

Trace, outcome, and evidence.

## Required Fields

- role
- scene_type

## Quality Signals And Rubric Vocabulary

Tool success and recovery quality.

## Acceptable Band, Stop Rule, And Bloat Guardrail

Stop when the error band is acceptable.

## When Run Intake Is Enough

Use run intake for one-off traces.
""".strip()
                + "\n",
                encoding="utf-8",
            )
            adapter_ref.write_text(
                """
# Custom Tool Runner

## Source Files

Raw traces.

## Result Normalization

Normalize trace id, tool calls, outputs, and recovery reason.

## Structural Diagnostics

Timeouts and retries.

## Human Signals

Preserve human review.

## Blocked Actions

Do not mutate policy before classification.
""".strip()
                + "\n",
                encoding="utf-8",
            )
            run_dir = root / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "tool_run",
                "--project",
                "demo",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_tool_runner",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "tool_trace=local/tool-trace.json",
                "--profile-ref",
                str(profile_ref),
                "--adapter-ref",
                str(adapter_ref),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["next_actions"] = ["classify timeout recovery before changing tool policy"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_run_intake.py", str(run_dir), "--validate-adapter-boundaries")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "manifest.profile_ref required fields contain conversation-role field outside "
                "conversation_role profile: role",
                result.stderr,
            )
            self.assertIn(
                "manifest.profile_ref required fields contain conversation-role field outside "
                "conversation_role profile: scene_type",
                result.stderr,
            )

    def test_validate_run_intake_rejects_promptfoo_schema_fields_in_non_promptfoo_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_ref = root / "eval_datasets" / "profiles" / "tool_use_eval" / "README.md"
            adapter_ref = root / "eval_datasets" / "adapters" / "custom_tool_runner" / "README.md"
            profile_ref.parent.mkdir(parents=True)
            adapter_ref.parent.mkdir(parents=True)
            profile_ref.write_text(
                """
# Tool Use Eval

## Domain Purpose

Evaluate tool-use traces.

## Domain Boundary

Tool tasks only.

## Minimum Artifact Shape

Trace, outcome, and evidence.

## Required Fields

- trace_id
- tool_steps

## Quality Signals And Rubric Vocabulary

Tool success and recovery quality.

## Acceptable Band, Stop Rule, And Bloat Guardrail

Stop when the error band is acceptable.

## When Run Intake Is Enough

Use run intake for one-off traces.
""".strip()
                + "\n",
                encoding="utf-8",
            )
            adapter_ref.write_text(
                """
# Custom Tool Runner

## Source Files

Raw traces.

## Result Normalization

Map raw traces through `vars`, `input_var`, `legacy_asserts`, and `assert`.

## Structural Diagnostics

Timeouts and retries.

## Human Signals

Preserve human review.

## Blocked Actions

Do not mutate policy before classification.
""".strip()
                + "\n",
                encoding="utf-8",
            )
            run_dir = root / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "tool_run",
                "--project",
                "demo",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_tool_runner",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "tool_trace=local/tool-trace.json",
                "--profile-ref",
                str(profile_ref),
                "--adapter-ref",
                str(adapter_ref),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["next_actions"] = ["classify timeout recovery before changing tool policy"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_run_intake.py", str(run_dir), "--validate-adapter-boundaries")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "manifest.adapter_ref contains Promptfoo schema field for non-promptfoo adapter: vars",
                result.stderr,
            )
            self.assertIn(
                "manifest.adapter_ref contains Promptfoo schema field for non-promptfoo adapter: input_var",
                result.stderr,
            )
            self.assertIn(
                "manifest.adapter_ref contains Promptfoo schema field for non-promptfoo adapter: legacy_asserts",
                result.stderr,
            )
            self.assertIn(
                "manifest.adapter_ref contains Promptfoo schema field for non-promptfoo adapter: assert",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_uncontrolled_prompt_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "revise_prompt_boundary"
            decision["human_signal_refs"] = []
            decision["blocked_actions"] = []
            decision["next_actions"] = ["patch prompt to handle failed case 7"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("prompt mutation requires human_signal_refs", result.stderr)
            self.assertIn("prompt mutation requires prompt_bloat_gate.checked=true", result.stderr)
            self.assertIn("prompt mutation requires replay_targets", result.stderr)

    def test_validate_learning_action_plan_rejects_detached_decision_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            write_json(run_dir / "decision.json", {"decision_type": "needs_decision"})
            (run_dir / "human_signals.jsonl").write_text("", encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing manifest: manifest.json", result.stderr)

    def test_validate_learning_action_plan_rejects_manifest_decision_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["run_id"] = "other_run"
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("manifest.run_id and decision.run_id must match", result.stderr)

    def test_validate_learning_action_plan_rejects_case_specific_rule_without_hl_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["human_signal_refs"] = []
            decision["blocked_actions"] = []
            decision["next_actions"] = ["add case-specific rule for failed case 7"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("prompt mutation requires human_signal_refs", result.stderr)

    def test_validate_learning_action_plan_rejects_chinese_prompt_patch_without_hl_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["human_signal_refs"] = []
            decision["blocked_actions"] = []
            decision["next_actions"] = ["修改 prompt，给失败用例 7 加一条规则"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("prompt mutation requires human_signal_refs", result.stderr)

    def test_validate_learning_action_plan_rejects_dataset_candidate_without_controller_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "create_dataset_candidate_unit"
            decision["human_signal_refs"] = []
            decision["blocked_actions"] = []
            decision["next_actions"] = ["write a new candidate dataset case from failed case 7"]
            decision["dataset_generation"] = {
                "needed": True,
                "reason": "Failed case 7 should become a new test.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("dataset candidate generation requires human_signal_refs", result.stderr)
            self.assertIn("dataset candidate generation requires replay_targets", result.stderr)
            self.assertIn(
                "dataset candidate generation requires learning_state_ref or learning_state_not_needed_reason",
                result.stderr,
            )

    def test_validate_learning_action_plan_accepts_controlled_dataset_candidate_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "This repeated failure needs a small candidate replay pack, not a prompt patch.",
                "--suggested-outcome",
                "create_dataset_candidate_unit",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "create_dataset_candidate_unit"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["prompt_patch_without_replay"]
            decision["next_actions"] = ["draft a two-case candidate unit from the repeated failure"]
            decision["replay_targets"] = ["local:two-case contrast replay"]
            decision["learning_state_not_needed_reason"] = (
                "This is the first isolated candidate unit for a temporary run."
            )
            decision["dataset_generation"] = {
                "needed": True,
                "reason": "A repeated judgeable failure needs replay before any prompt or gate decision.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("learning action plan valid", result.stdout)

    def test_validate_learning_action_plan_rejects_dataset_candidate_that_does_not_consume_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            run_dir = workspace / "run"
            state_path = workspace / "learning_state.v1.json"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "tool_trace_replay_002",
                "--project",
                "demo",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_json",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "The previous loop already selected the tool argument-schema replay target.",
                "--suggested-outcome",
                "create_dataset_candidate_unit",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state_result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state_path),
                "--active-loop",
                "tool_trace",
                "--primary-outcome",
                "create_dataset_candidate_unit",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_tool_runner",
                "--next-replay-target",
                "tool_trace:argument_schema_failures",
                "--blocked-action",
                "promote_canonical",
            )
            self.assertEqual(state_result.returncode, 0, state_result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "create_dataset_candidate_unit"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["prompt_patch_without_replay"]
            decision["next_actions"] = ["draft a candidate unit from a different local replay"]
            decision["replay_targets"] = ["local:unrelated contrast replay"]
            decision["learning_state_ref"] = str(state_path)
            decision["dataset_generation"] = {
                "needed": True,
                "reason": "A previous loop selected a concrete replay target.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "decision.replay_targets must include learning_state.next_replay_targets: "
                "tool_trace:argument_schema_failures",
                result.stderr,
            )
            self.assertIn(
                "decision.blocked_actions must include learning_state.blocked_actions: promote_canonical",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_cross_profile_learning_state_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            run_dir = workspace / "run"
            state_path = workspace / "learning_state.v1.json"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "tool_trace_replay_003",
                "--project",
                "tool_trace_project",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_tool_runner",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "Use the current tool trace state, not an old role-eval state.",
                "--suggested-outcome",
                "create_dataset_candidate_unit",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state_result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state_path),
                "--active-loop",
                "old_role_loop",
                "--primary-outcome",
                "create_dataset_candidate_unit",
                "--profile",
                "conversation_role",
                "--adapter",
                "promptfoo",
                "--next-replay-target",
                "tool_trace:argument_schema_failures",
                "--blocked-action",
                "prompt_patch_without_replay",
            )
            self.assertEqual(state_result.returncode, 0, state_result.stderr)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["project"] = "old_role_project"
            state["profile"] = "conversation_role"
            state["adapter"] = "promptfoo"
            state["run_id"] = "old_role_run"
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "create_dataset_candidate_unit"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["prompt_patch_without_replay"]
            decision["next_actions"] = ["draft a candidate unit from current tool trace evidence"]
            decision["replay_targets"] = ["tool_trace:argument_schema_failures"]
            decision["learning_state_ref"] = str(state_path)
            decision["dataset_generation"] = {
                "needed": True,
                "reason": "The current run needs a tool-use candidate unit.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "learning_state_ref profile does not match manifest.profile",
                result.stderr,
            )
            self.assertIn(
                "learning_state_ref adapter does not match manifest.adapter",
                result.stderr,
            )
            self.assertIn(
                "learning_state_ref run_id does not match manifest.run_id",
                result.stderr,
            )

    def test_validate_learning_action_plan_accepts_dataset_candidate_supported_by_route_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            route_path = run_dir / "route.json"
            observations_path = run_dir / "observations.json"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "route_dataset_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "The route points at a coverage gap that needs a tiny candidate unit.",
                "--suggested-outcome",
                "create_dataset_candidate_unit",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            write_json(
                observations_path,
                {
                    "version": "v1",
                    "run_id": "route_dataset_run",
                    "profile": "generative_content",
                    "adapter": "batch_story_generation",
                    "records": [{"record_id": "case_1"}],
                },
            )
            write_json(
                route_path,
                {
                    "version": "v1",
                    "run_id": "route_dataset_run",
                    "profile": "generative_content",
                    "adapter": "batch_story_generation",
                    "observation_path": "observations.json",
                    "primary_outcome": "create_dataset_candidate_unit",
                    "outcome_candidates": [
                        {
                            "outcome": "create_dataset_candidate_unit",
                            "evidence_refs": ["observations.json#case_1"],
                        }
                    ],
                    "prompt_mutation_allowed": False,
                    "blocked_actions": ["prompt_patch_without_replay", "add_case_specific_if_rule"],
                },
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "create_dataset_candidate_unit"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["prompt_patch_without_replay", "add_case_specific_if_rule"]
            decision["next_actions"] = ["draft a two-case candidate unit from route evidence"]
            decision["replay_targets"] = ["local:two-case contrast replay"]
            decision["observation_route_ref"] = "route.json"
            decision["learning_state_not_needed_reason"] = "Route-backed one-off candidate unit for an isolated run."
            decision["dataset_generation"] = {
                "needed": True,
                "reason": "Route evidence points at missing coverage.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("learning action plan valid", result.stdout)

    def test_validate_learning_action_plan_rejects_dataset_candidate_that_ignores_existing_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            route_path = run_dir / "route.json"
            observations_path = run_dir / "observations.json"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "route_dataset_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "The route points at a coverage gap that needs a tiny candidate unit.",
                "--suggested-outcome",
                "create_dataset_candidate_unit",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            write_json(
                observations_path,
                {"version": "v1", "run_id": "route_dataset_run", "records": [{"record_id": "case_1"}]},
            )
            write_json(
                route_path,
                {
                    "version": "v1",
                    "run_id": "route_dataset_run",
                    "observation_path": "observations.json",
                    "primary_outcome": "create_dataset_candidate_unit",
                    "outcome_candidates": [
                        {
                            "outcome": "create_dataset_candidate_unit",
                            "evidence_refs": ["observations.json#case_1"],
                        }
                    ],
                    "prompt_mutation_allowed": False,
                    "blocked_actions": ["prompt_patch_without_replay", "add_case_specific_if_rule"],
                },
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "create_dataset_candidate_unit"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["prompt_patch_without_replay", "add_case_specific_if_rule"]
            decision["next_actions"] = ["draft a two-case candidate unit from route evidence"]
            decision["replay_targets"] = ["local:two-case contrast replay"]
            decision["learning_state_not_needed_reason"] = "Route-backed one-off candidate unit for an isolated run."
            decision["dataset_generation"] = {
                "needed": True,
                "reason": "Route evidence points at missing coverage.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "dataset candidate generation requires observation_route_ref when route.json exists",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_route_ref_with_missing_observations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            route_path = run_dir / "route.json"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "route_dataset_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "The route points at a coverage gap that needs a tiny candidate unit.",
                "--suggested-outcome",
                "create_dataset_candidate_unit",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            write_json(
                route_path,
                {
                    "version": "v1",
                    "run_id": "route_dataset_run",
                    "observation_path": "missing_observations.json",
                    "primary_outcome": "create_dataset_candidate_unit",
                    "outcome_candidates": [{"outcome": "create_dataset_candidate_unit"}],
                    "prompt_mutation_allowed": False,
                    "blocked_actions": ["prompt_patch_without_replay", "add_case_specific_if_rule"],
                },
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "create_dataset_candidate_unit"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["prompt_patch_without_replay", "add_case_specific_if_rule"]
            decision["next_actions"] = ["draft a two-case candidate unit from route evidence"]
            decision["replay_targets"] = ["local:two-case contrast replay"]
            decision["observation_route_ref"] = "route.json"
            decision["learning_state_not_needed_reason"] = "Route-backed one-off candidate unit for an isolated run."
            decision["dataset_generation"] = {
                "needed": True,
                "reason": "Route evidence points at missing coverage.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("observation_route_ref observation_path is not readable", result.stderr)

    def test_validate_learning_action_plan_rejects_route_ref_with_missing_record_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            route_path = run_dir / "route.json"
            observations_path = run_dir / "observations.json"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "route_dataset_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "The route points at a coverage gap that needs a tiny candidate unit.",
                "--suggested-outcome",
                "create_dataset_candidate_unit",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            write_json(
                observations_path,
                {"version": "v1", "run_id": "route_dataset_run", "records": [{"record_id": "case_1"}]},
            )
            write_json(
                route_path,
                {
                    "version": "v1",
                    "run_id": "route_dataset_run",
                    "observation_path": "observations.json",
                    "primary_outcome": "create_dataset_candidate_unit",
                    "outcome_candidates": [
                        {
                            "outcome": "create_dataset_candidate_unit",
                            "evidence_refs": ["observations.json#ghost"],
                        }
                    ],
                    "prompt_mutation_allowed": False,
                    "blocked_actions": ["prompt_patch_without_replay", "add_case_specific_if_rule"],
                },
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "create_dataset_candidate_unit"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["prompt_patch_without_replay", "add_case_specific_if_rule"]
            decision["next_actions"] = ["draft a two-case candidate unit from route evidence"]
            decision["replay_targets"] = ["local:two-case contrast replay"]
            decision["observation_route_ref"] = "route.json"
            decision["learning_state_not_needed_reason"] = "Route-backed one-off candidate unit for an isolated run."
            decision["dataset_generation"] = {
                "needed": True,
                "reason": "Route evidence points at missing coverage.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "observation_route_ref evidence_ref points to missing record: observations.json#ghost",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_route_ref_with_wrong_profile_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            route_path = run_dir / "route.json"
            observations_path = run_dir / "observations.json"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "shared_run",
                "--project",
                "demo",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_tool_runner",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "The route points at a coverage gap that needs a tiny candidate unit.",
                "--suggested-outcome",
                "create_dataset_candidate_unit",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            write_json(
                observations_path,
                {
                    "version": "v1",
                    "run_id": "shared_run",
                    "profile": "conversation_role",
                    "adapter": "promptfoo",
                    "records": [{"record_id": "case_1"}],
                },
            )
            write_json(
                route_path,
                {
                    "version": "v1",
                    "run_id": "shared_run",
                    "profile": "conversation_role",
                    "adapter": "promptfoo",
                    "observation_path": "observations.json",
                    "primary_outcome": "create_dataset_candidate_unit",
                    "outcome_candidates": [
                        {
                            "outcome": "create_dataset_candidate_unit",
                            "evidence_refs": ["observations.json#case_1"],
                        }
                    ],
                    "prompt_mutation_allowed": False,
                    "blocked_actions": ["prompt_patch_without_replay", "add_case_specific_if_rule"],
                },
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "create_dataset_candidate_unit"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["prompt_patch_without_replay", "add_case_specific_if_rule"]
            decision["next_actions"] = ["draft a two-case candidate unit from route evidence"]
            decision["replay_targets"] = ["local:two-case contrast replay"]
            decision["observation_route_ref"] = "route.json"
            decision["learning_state_not_needed_reason"] = "Route-backed one-off candidate unit for an isolated run."
            decision["dataset_generation"] = {
                "needed": True,
                "reason": "Route evidence points at missing coverage.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("observation_route_ref profile does not match decision.profile", result.stderr)
            self.assertIn("observation_route_ref adapter does not match decision.adapter", result.stderr)
            self.assertIn("observation_route_ref observation_path profile does not match decision.profile", result.stderr)
            self.assertIn("observation_route_ref observation_path adapter does not match decision.adapter", result.stderr)

    def test_validate_learning_action_plan_rejects_route_ref_without_profile_adapter_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            route_path = run_dir / "route.json"
            observations_path = run_dir / "observations.json"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "shared_run",
                "--project",
                "demo",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_tool_runner",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "The route points at a coverage gap that needs a tiny candidate unit.",
                "--suggested-outcome",
                "create_dataset_candidate_unit",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            write_json(
                observations_path,
                {
                    "version": "v1",
                    "run_id": "shared_run",
                    "records": [{"record_id": "case_1"}],
                },
            )
            write_json(
                route_path,
                {
                    "version": "v1",
                    "run_id": "shared_run",
                    "observation_path": "observations.json",
                    "primary_outcome": "create_dataset_candidate_unit",
                    "outcome_candidates": [
                        {
                            "outcome": "create_dataset_candidate_unit",
                            "evidence_refs": ["observations.json#case_1"],
                        }
                    ],
                    "prompt_mutation_allowed": False,
                    "blocked_actions": ["prompt_patch_without_replay", "add_case_specific_if_rule"],
                },
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "create_dataset_candidate_unit"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["prompt_patch_without_replay", "add_case_specific_if_rule"]
            decision["next_actions"] = ["draft a two-case candidate unit from route evidence"]
            decision["replay_targets"] = ["local:two-case contrast replay"]
            decision["observation_route_ref"] = "route.json"
            decision["learning_state_not_needed_reason"] = "Route-backed one-off candidate unit for an isolated run."
            decision["dataset_generation"] = {
                "needed": True,
                "reason": "Route evidence points at missing coverage.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("observation_route_ref requires profile when decision.profile is set", result.stderr)
            self.assertIn("observation_route_ref requires adapter when decision.adapter is set", result.stderr)
            self.assertIn(
                "observation_route_ref observation_path requires profile when decision.profile is set",
                result.stderr,
            )
            self.assertIn(
                "observation_route_ref observation_path requires adapter when decision.adapter is set",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_route_ref_without_stable_record_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            route_path = run_dir / "route.json"
            observations_path = run_dir / "observations.json"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "route_dataset_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "The route points at a coverage gap that needs a tiny candidate unit.",
                "--suggested-outcome",
                "create_dataset_candidate_unit",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            write_json(
                observations_path,
                {"version": "v1", "run_id": "route_dataset_run", "records": [{"input": "missing id"}]},
            )
            write_json(
                route_path,
                {
                    "version": "v1",
                    "run_id": "route_dataset_run",
                    "observation_path": "observations.json",
                    "primary_outcome": "create_dataset_candidate_unit",
                    "outcome_candidates": [
                        {
                            "outcome": "create_dataset_candidate_unit",
                            "evidence_refs": ["observations.json#record_0"],
                        }
                    ],
                    "prompt_mutation_allowed": False,
                    "blocked_actions": ["prompt_patch_without_replay", "add_case_specific_if_rule"],
                },
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "create_dataset_candidate_unit"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["prompt_patch_without_replay", "add_case_specific_if_rule"]
            decision["next_actions"] = ["draft a two-case candidate unit from route evidence"]
            decision["replay_targets"] = ["local:two-case contrast replay"]
            decision["observation_route_ref"] = "route.json"
            decision["learning_state_not_needed_reason"] = "Route-backed one-off candidate unit for an isolated run."
            decision["dataset_generation"] = {
                "needed": True,
                "reason": "Route evidence points at missing coverage.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "observation_route_ref observation_path records require stable record_id for evidence_refs",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_acceptance_without_evidence_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--decision-type",
                "accept_direction",
                "--accepted-direction",
                "true",
                "--primary-reason",
                "Looks directionally better.",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["next_actions"] = ["keep this direction"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("accepted direction requires human_signal_refs", result.stderr)
            self.assertIn("accepted direction requires event_refs", result.stderr)

    def test_validate_learning_action_plan_rejects_acceptance_with_unrelated_event_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            events_path = Path(tmp) / "events.jsonl"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--decision-type",
                "accept_direction",
                "--accepted-direction",
                "true",
                "--primary-reason",
                "Looks directionally better.",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_acceptance_signal",
                "--raw-signal",
                "This direction is acceptable.",
                "--suggested-outcome",
                "accept_direction",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            events_path.write_text(
                json.dumps(
                    {
                        "event_id": "evt_wrong_accept_1",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "rubric_revised",
                        "actor": "test",
                        "decision": {"to": "rubric_revision", "reason": "Rubric wording changed."},
                        "evidence": {"kind": "human_review", "summary": "A different event."},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["event_refs"] = [f"{events_path}#evt_wrong_accept_1"]
            decision["next_actions"] = ["keep this direction"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "event_refs does not support decision_type accept_direction",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_acceptance_event_without_decision_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            events_path = Path(tmp) / "events.jsonl"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--decision-type",
                "accept_direction",
                "--accepted-direction",
                "true",
                "--primary-reason",
                "Looks directionally better.",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_acceptance_signal",
                "--raw-signal",
                "This direction is acceptable.",
                "--suggested-outcome",
                "accept_direction",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            events_path.write_text(
                json.dumps(
                    {
                        "event_id": "evt_accept_1",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "experiment_promoted",
                        "actor": "test",
                        "decision": {"to": "accept_direction", "reason": "Direction accepted."},
                        "evidence": {"kind": "human_review", "summary": "A generic acceptance event."},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["event_refs"] = [f"{events_path}#evt_accept_1"]
            decision["next_actions"] = ["keep this direction"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("event_refs missing decision_id for decision context", result.stderr)
            self.assertIn("event_refs missing human_signal_refs for decision context", result.stderr)

    def test_validate_learning_action_plan_accepts_controlled_prompt_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "Repeated failures justify a small replayed prompt experiment.",
                "--suggested-outcome",
                "compact_prompt_candidate",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "compact_prompt_candidate"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = [
                "prompt_patch_without_replay",
                "add_case_specific_if_rule",
            ]
            decision["next_actions"] = ["run targeted replay before scoped prompt mutation"]
            decision["replay_targets"] = ["promptfoo:therapy-tone contrast batch"]
            decision["prompt_bloat_gate"] = {
                "checked": True,
                "risk": "medium",
                "decision": "allow_targeted_replay_only",
                "traction_audit_not_needed_reason": "Small replay follows an already-classified repeated human signal.",
                "repeated_failure_basis": "The human signal names a repeated abstraction-level pattern, not one narrow case.",
                "non_prompt_alternatives_considered": ["failure_pattern", "rubric_revision", "case_revision"],
                "ordinary_interaction_risk": "medium",
                "removal_condition": "Remove the compact candidate if targeted replay shows no user-facing gain.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("learning action plan valid", result.stdout)

    def test_validate_learning_action_plan_rejects_prompt_mutation_that_ignores_state_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            run_dir = workspace / "run"
            state_path = workspace / "learning_state.v1.json"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "state_prompt_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "A compact prompt experiment is only allowed on the previous replay target.",
                "--suggested-outcome",
                "compact_prompt_candidate",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state_result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state_path),
                "--active-loop",
                "state_prompt_run",
                "--primary-outcome",
                "failure_pattern_candidate",
                "--next-replay-target",
                "promptfoo:therapy-tone contrast batch",
                "--blocked-action",
                "promote_canonical",
            )
            self.assertEqual(state_result.returncode, 0, state_result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "compact_prompt_candidate"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = [
                "prompt_patch_without_replay",
                "add_case_specific_if_rule",
            ]
            decision["next_actions"] = ["run targeted replay before scoped prompt mutation"]
            decision["replay_targets"] = ["promptfoo:unrelated contrast batch"]
            decision["learning_state_ref"] = str(state_path)
            decision["prompt_bloat_gate"] = {
                "checked": True,
                "risk": "medium",
                "decision": "allow_targeted_replay_only",
                "traction_audit_not_needed_reason": "Small replay follows an already-classified repeated human signal.",
                "repeated_failure_basis": "The human signal names a repeated abstraction-level pattern, not one narrow case.",
                "non_prompt_alternatives_considered": ["failure_pattern", "rubric_revision", "case_revision"],
                "ordinary_interaction_risk": "medium",
                "removal_condition": "Remove the compact candidate if targeted replay shows no user-facing gain.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "decision.replay_targets must include learning_state.next_replay_targets: "
                "promptfoo:therapy-tone contrast batch",
                result.stderr,
            )
            self.assertIn(
                "decision.blocked_actions must include learning_state.blocked_actions: promote_canonical",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_empty_prompt_bloat_gate_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "empty_bloat_gate_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "Repeated failures justify a small replayed prompt experiment.",
                "--suggested-outcome",
                "compact_prompt_candidate",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "compact_prompt_candidate"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = [
                "prompt_patch_without_replay",
                "add_case_specific_if_rule",
            ]
            decision["next_actions"] = ["run targeted replay before scoped prompt mutation"]
            decision["replay_targets"] = ["promptfoo:therapy-tone contrast batch"]
            decision["prompt_bloat_gate"] = {
                "checked": True,
                "risk": "medium",
                "decision": "allow_targeted_replay_only",
                "traction_audit_not_needed_reason": "Small replay follows an already-classified repeated human signal.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("prompt_bloat_gate requires repeated_failure_basis", result.stderr)
            self.assertIn("prompt_bloat_gate requires non_prompt_alternatives_considered", result.stderr)
            self.assertIn("prompt_bloat_gate requires ordinary_interaction_risk", result.stderr)
            self.assertIn("prompt_bloat_gate requires removal_condition", result.stderr)

    def test_validate_learning_action_plan_rejects_renamed_prompt_experiment_without_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "controlled_prompt_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "A compact variant may be worth a replayed experiment.",
                "--suggested-outcome",
                "controlled_prompt_experiment",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "controlled_prompt_experiment"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["prompt_patch_without_replay"]
            decision["next_actions"] = ["apply compact variant"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("prompt mutation requires prompt_bloat_gate.checked=true", result.stderr)
            self.assertIn("prompt mutation requires replay_targets", result.stderr)

    def test_validate_learning_action_plan_rejects_agent_inference_as_reviewed_human_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "agent_signal_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            (run_dir / "human_signals.jsonl").write_text(
                json.dumps(
                    {
                        "signal_type": "human_abstraction_signal",
                        "raw_signal": "Agent inferred this should become a compact prompt candidate.",
                        "candidate_failure_tags": [],
                        "suggested_outcome": "compact_prompt_candidate",
                        "blocked_actions": ["prompt_patch_without_replay"],
                        "needs_review": False,
                        "source_type": "agent_inference",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "compact_prompt_candidate"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["prompt_patch_without_replay", "add_case_specific_if_rule"]
            decision["next_actions"] = ["run targeted replay before scoped prompt mutation"]
            decision["replay_targets"] = ["promptfoo:therapy-tone contrast batch"]
            decision["prompt_bloat_gate"] = {
                "checked": True,
                "risk": "medium",
                "decision": "allow_targeted_replay_only",
                "traction_audit_not_needed_reason": "Small replay follows a classified signal.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "human_signals[1].source_type cannot be agent_inference when needs_review=false",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_prompt_mutation_blocked_by_route_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            route_path = run_dir / "route.json"
            observations_path = run_dir / "observations.json"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "route_blocked_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "A repeated failure might look tempting to patch, but the route says stop.",
                "--suggested-outcome",
                "compact_prompt_candidate",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            write_json(
                observations_path,
                {"version": "v1", "run_id": "route_blocked_run", "records": [{"record_id": "case_1"}]},
            )
            write_json(
                route_path,
                {
                    "version": "v1",
                    "run_id": "route_blocked_run",
                    "observation_path": "observations.json",
                    "primary_outcome": "stop_tuning",
                    "outcome_candidates": [{"outcome": "stop_tuning"}],
                    "prompt_mutation_allowed": False,
                    "blocked_actions": ["mutate_prompt_or_policy", "add_case_specific_if_rule"],
                },
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "compact_prompt_candidate"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = [
                "prompt_patch_without_replay",
                "add_case_specific_if_rule",
            ]
            decision["next_actions"] = ["run targeted replay before scoped prompt mutation"]
            decision["replay_targets"] = ["promptfoo:therapy-tone contrast batch"]
            decision["observation_route_ref"] = "route.json"
            decision["prompt_bloat_gate"] = {
                "checked": True,
                "risk": "medium",
                "decision": "allow_targeted_replay_only",
                "traction_audit_not_needed_reason": "Small replay follows an already-classified repeated human signal.",
                "repeated_failure_basis": "The judged replay supports a repeated abstraction-level pattern.",
                "non_prompt_alternatives_considered": ["failure_pattern", "rubric_revision", "case_revision"],
                "ordinary_interaction_risk": "medium",
                "removal_condition": "Remove the compact candidate if the judged replay stops supporting it.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("observation_route_ref blocks prompt mutation", result.stderr)
            self.assertIn(
                "observation_route_ref does not support decision_type compact_prompt_candidate",
                result.stderr,
            )
            self.assertIn(
                "decision.blocked_actions must include observation route blocked_action: mutate_prompt_or_policy",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_prompt_mutation_that_ignores_existing_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            route_path = run_dir / "route.json"
            observations_path = run_dir / "observations.json"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "route_ignored_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "The run is good enough; do not keep tuning narrow failures.",
                "--suggested-outcome",
                "compact_prompt_candidate",
                "--blocked-action",
                "continue_prompt_tuning_for_narrow_failures",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            write_json(
                observations_path,
                {"version": "v1", "run_id": "route_ignored_run", "records": [{"record_id": "case_1"}]},
            )
            write_json(
                route_path,
                {
                    "version": "v1",
                    "run_id": "route_ignored_run",
                    "observation_path": "observations.json",
                    "primary_outcome": "stop_tuning",
                    "outcome_candidates": [{"outcome": "stop_tuning"}],
                    "prompt_mutation_allowed": False,
                    "blocked_actions": ["continue_prompt_tuning_for_narrow_failures"],
                },
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "compact_prompt_candidate"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = [
                "prompt_patch_without_replay",
                "add_case_specific_if_rule",
                "continue_prompt_tuning_for_narrow_failures",
            ]
            decision["next_actions"] = ["try a compact prompt variant anyway"]
            decision["replay_targets"] = ["promptfoo:compact contrast batch"]
            decision["prompt_bloat_gate"] = {
                "checked": True,
                "risk": "medium",
                "decision": "allow_targeted_replay_only",
                "traction_audit_not_needed_reason": "Small replay follows a classified signal.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "prompt mutation requires observation_route_ref when route.json exists",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_not_assessed_reward_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            reward_path = Path(tmp) / "dry_run.reward.json"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "Repeated failures justify a small replayed prompt experiment.",
                "--suggested-outcome",
                "compact_prompt_candidate",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            write_json(
                reward_path,
                {
                    "version": "v1",
                    "mutation_id": "dry_run_mutation",
                    "assessment_level": "dry_run",
                    "decision": "not_assessed",
                    "hard_gates": {
                        "has_real_model_output": False,
                        "has_real_judge_score": False,
                    },
                },
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "compact_prompt_candidate"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = [
                "prompt_patch_without_replay",
                "add_case_specific_if_rule",
            ]
            decision["next_actions"] = ["run targeted replay before scoped prompt mutation"]
            decision["replay_targets"] = ["promptfoo:therapy-tone contrast batch"]
            decision["reward_assessment_refs"] = [str(reward_path)]
            decision["prompt_bloat_gate"] = {
                "checked": True,
                "risk": "medium",
                "decision": "allow_targeted_replay_only",
                "traction_audit_not_needed_reason": "Small replay follows an already-classified repeated human signal.",
                "repeated_failure_basis": "The judged replay supports a repeated abstraction-level pattern.",
                "non_prompt_alternatives_considered": ["failure_pattern", "rubric_revision", "case_revision"],
                "ordinary_interaction_risk": "medium",
                "removal_condition": "Remove the compact candidate if the judged replay stops supporting it.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("reward_assessment_refs points to not_assessed reward", result.stderr)
            self.assertIn("reward_assessment_refs requires judged_replay", result.stderr)

    def test_validate_learning_action_plan_rejects_reward_ref_for_other_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            reward_path = Path(tmp) / "other_run.reward.json"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "Repeated failures justify a small replayed prompt experiment.",
                "--suggested-outcome",
                "compact_prompt_candidate",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            write_json(
                reward_path,
                {
                    "version": "v1",
                    "mutation_id": "other_run_mutation",
                    "observation_run_id": "other_run",
                    "assessment_level": "judged_replay",
                    "decision": "compress_candidate",
                    "hard_gates": {
                        "has_real_model_output": True,
                        "has_real_judge_score": True,
                    },
                },
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "compact_prompt_candidate"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = [
                "prompt_patch_without_replay",
                "add_case_specific_if_rule",
            ]
            decision["next_actions"] = ["run targeted replay before scoped prompt mutation"]
            decision["replay_targets"] = ["promptfoo:therapy-tone contrast batch"]
            decision["reward_assessment_refs"] = [str(reward_path)]
            decision["prompt_bloat_gate"] = {
                "checked": True,
                "risk": "medium",
                "decision": "allow_targeted_replay_only",
                "traction_audit_not_needed_reason": "Small replay follows an already-classified repeated human signal.",
                "repeated_failure_basis": "The judged replay supports a repeated abstraction-level pattern.",
                "non_prompt_alternatives_considered": ["failure_pattern", "rubric_revision", "case_revision"],
                "ordinary_interaction_risk": "medium",
                "removal_condition": "Remove the compact candidate if the judged replay stops supporting it.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "reward_assessment_refs observation_run_id does not match decision.run_id",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_reward_ref_missing_run_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            reward_path = Path(tmp) / "missing_run.reward.json"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "Repeated failures justify a small replayed prompt experiment.",
                "--suggested-outcome",
                "compact_prompt_candidate",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            write_json(
                reward_path,
                {
                    "version": "v1",
                    "mutation_id": "missing_run_mutation",
                    "assessment_level": "judged_replay",
                    "decision": "compress_candidate",
                    "hard_gates": {
                        "has_real_model_output": True,
                        "has_real_judge_score": True,
                    },
                },
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "compact_prompt_candidate"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = [
                "prompt_patch_without_replay",
                "add_case_specific_if_rule",
            ]
            decision["next_actions"] = ["run targeted replay before scoped prompt mutation"]
            decision["replay_targets"] = ["promptfoo:therapy-tone contrast batch"]
            decision["reward_assessment_refs"] = [str(reward_path)]
            decision["prompt_bloat_gate"] = {
                "checked": True,
                "risk": "medium",
                "decision": "allow_targeted_replay_only",
                "traction_audit_not_needed_reason": "Small replay follows an already-classified repeated human signal.",
                "repeated_failure_basis": "The judged replay supports a repeated abstraction-level pattern.",
                "non_prompt_alternatives_considered": ["failure_pattern", "rubric_revision", "case_revision"],
                "ordinary_interaction_risk": "medium",
                "removal_condition": "Remove the compact candidate if the judged replay stops supporting it.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "reward_assessment_refs missing observation_run_id for decision.run_id",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_reward_ref_for_other_observations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            route_path = run_dir / "route.json"
            observations_path = run_dir / "observations.json"
            reward_path = Path(tmp) / "wrong_observations.reward.json"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "reward_ref_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "A repeated pattern may justify a small replayed prompt experiment.",
                "--suggested-outcome",
                "compact_prompt_candidate",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            write_json(
                observations_path,
                {
                    "version": "v1",
                    "run_id": "reward_ref_run",
                    "profile": "generative_content",
                    "adapter": "batch_story_generation",
                    "records": [{"record_id": "case_1"}],
                },
            )
            write_json(
                route_path,
                {
                    "version": "v1",
                    "run_id": "reward_ref_run",
                    "profile": "generative_content",
                    "adapter": "batch_story_generation",
                    "observation_path": "observations.json",
                    "primary_outcome": "compact_prompt_candidate",
                    "outcome_candidates": [
                        {
                            "outcome": "compact_prompt_candidate",
                            "evidence_refs": ["observations.json#case_1"],
                        }
                    ],
                    "prompt_mutation_allowed": True,
                    "blocked_actions": ["prompt_patch_without_replay", "add_case_specific_if_rule"],
                },
            )
            write_json(
                reward_path,
                {
                    "version": "v1",
                    "mutation_id": "wrong_observations_mutation",
                    "observation_run_id": "reward_ref_run",
                    "observation_path": "other_observations.json",
                    "assessment_level": "judged_replay",
                    "decision": "compress_candidate",
                    "hard_gates": {
                        "has_real_model_output": True,
                        "has_real_judge_score": True,
                    },
                },
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "compact_prompt_candidate"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = [
                "prompt_patch_without_replay",
                "add_case_specific_if_rule",
            ]
            decision["next_actions"] = ["run targeted replay before scoped prompt mutation"]
            decision["replay_targets"] = ["promptfoo:therapy-tone contrast batch"]
            decision["observation_route_ref"] = "route.json"
            decision["reward_assessment_refs"] = [str(reward_path)]
            decision["prompt_bloat_gate"] = {
                "checked": True,
                "risk": "medium",
                "decision": "allow_targeted_replay_only",
                "traction_audit_not_needed_reason": "Small replay follows an already-classified repeated human signal.",
                "repeated_failure_basis": "The judged replay supports a repeated abstraction-level pattern.",
                "non_prompt_alternatives_considered": ["failure_pattern", "rubric_revision", "case_revision"],
                "ordinary_interaction_risk": "medium",
                "removal_condition": "Remove the compact candidate if the judged replay stops supporting it.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "reward_assessment_refs observation_path does not match observation_route_ref observation_path",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_negative_reward_ref_for_prompt_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            reward_path = Path(tmp) / "negative.reward.json"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "negative_reward_ref_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "A repeated pattern may justify a small replayed prompt experiment.",
                "--suggested-outcome",
                "compact_prompt_candidate",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            write_json(
                reward_path,
                {
                    "version": "v1",
                    "mutation_id": "negative_mutation",
                    "observation_run_id": "negative_reward_ref_run",
                    "assessment_level": "judged_replay",
                    "decision": "retire_or_noop",
                    "hard_gates": {
                        "has_real_model_output": True,
                        "has_real_judge_score": True,
                    },
                },
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "compact_prompt_candidate"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = [
                "prompt_patch_without_replay",
                "add_case_specific_if_rule",
            ]
            decision["next_actions"] = ["run targeted replay before scoped prompt mutation"]
            decision["replay_targets"] = ["promptfoo:therapy-tone contrast batch"]
            decision["reward_assessment_refs"] = [str(reward_path)]
            decision["prompt_bloat_gate"] = {
                "checked": True,
                "risk": "medium",
                "decision": "allow_targeted_replay_only",
                "traction_audit_not_needed_reason": "Small replay follows an already-classified repeated human signal.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "reward_assessment_refs decision does not support prompt mutation",
                result.stderr,
            )

    def test_validate_learning_action_plan_accepts_judged_reward_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            reward_path = Path(tmp) / "judged.reward.json"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "Repeated failures justify a small replayed prompt experiment.",
                "--suggested-outcome",
                "compact_prompt_candidate",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            write_json(
                reward_path,
                {
                    "version": "v1",
                    "mutation_id": "judged_mutation",
                    "observation_run_id": "demo_run",
                    "assessment_level": "judged_replay",
                    "decision": "keep_experiment",
                    "hard_gates": {
                        "has_real_model_output": True,
                        "has_real_judge_score": True,
                    },
                },
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "compact_prompt_candidate"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = [
                "prompt_patch_without_replay",
                "add_case_specific_if_rule",
            ]
            decision["next_actions"] = ["run targeted replay before scoped prompt mutation"]
            decision["replay_targets"] = ["promptfoo:therapy-tone contrast batch"]
            decision["reward_assessment_refs"] = [str(reward_path)]
            decision["prompt_bloat_gate"] = {
                "checked": True,
                "risk": "medium",
                "decision": "allow_targeted_replay_only",
                "traction_audit_not_needed_reason": "Small replay follows an already-classified repeated human signal.",
                "repeated_failure_basis": "The judged replay supports a repeated abstraction-level pattern.",
                "non_prompt_alternatives_considered": ["failure_pattern", "rubric_revision", "case_revision"],
                "ordinary_interaction_risk": "medium",
                "removal_condition": "Remove the compact candidate if the judged replay stops supporting it.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("learning action plan valid", result.stdout)

    def test_validate_learning_action_plan_accepts_run_relative_judged_reward_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            reward_path = run_dir / "reward_assessments" / "judged.reward.json"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "Repeated failures justify a small replayed prompt experiment.",
                "--suggested-outcome",
                "compact_prompt_candidate",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            reward_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(
                reward_path,
                {
                    "version": "v1",
                    "mutation_id": "judged_mutation",
                    "observation_run_id": "demo_run",
                    "assessment_level": "judged_replay",
                    "decision": "keep_experiment",
                    "hard_gates": {
                        "has_real_model_output": True,
                        "has_real_judge_score": True,
                    },
                },
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "compact_prompt_candidate"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = [
                "prompt_patch_without_replay",
                "add_case_specific_if_rule",
            ]
            decision["next_actions"] = ["run targeted replay before scoped prompt mutation"]
            decision["replay_targets"] = ["promptfoo:therapy-tone contrast batch"]
            decision["reward_assessment_refs"] = ["reward_assessments/judged.reward.json"]
            decision["prompt_bloat_gate"] = {
                "checked": True,
                "risk": "medium",
                "decision": "allow_targeted_replay_only",
                "traction_audit_not_needed_reason": "Small replay follows an already-classified repeated human signal.",
                "repeated_failure_basis": "The judged replay supports a repeated abstraction-level pattern.",
                "non_prompt_alternatives_considered": ["failure_pattern", "rubric_revision", "case_revision"],
                "ordinary_interaction_risk": "medium",
                "removal_condition": "Remove the compact candidate if the judged replay stops supporting it.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("learning action plan valid", result.stdout)

    def test_validate_learning_action_plan_rejects_unreviewed_prompt_mutation_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "Maybe a prompt experiment is needed, but this still needs review.",
                "--suggested-outcome",
                "compact_prompt_candidate",
                "--blocked-action",
                "prompt_patch_without_replay",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "compact_prompt_candidate"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = [
                "prompt_patch_without_replay",
                "add_case_specific_if_rule",
            ]
            decision["next_actions"] = ["run targeted replay before scoped prompt mutation"]
            decision["replay_targets"] = ["promptfoo:therapy-tone contrast batch"]
            decision["prompt_bloat_gate"] = {
                "checked": True,
                "risk": "medium",
                "decision": "allow_targeted_replay_only",
                "traction_audit_not_needed_reason": "Small replay follows a human signal.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "human_signal_refs contains unreviewed support for decision_type compact_prompt_candidate",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_empty_referenced_human_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            (run_dir / "human_signals.jsonl").write_text(
                json.dumps(
                    {
                        "signal_type": "human_abstraction_signal",
                        "raw_signal": "",
                        "candidate_failure_tags": [],
                        "suggested_outcome": "compact_prompt_candidate",
                        "blocked_actions": ["prompt_patch_without_replay"],
                        "needs_review": False,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "compact_prompt_candidate"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = [
                "prompt_patch_without_replay",
                "add_case_specific_if_rule",
            ]
            decision["next_actions"] = ["run targeted replay before scoped prompt mutation"]
            decision["replay_targets"] = ["promptfoo:therapy-tone contrast batch"]
            decision["prompt_bloat_gate"] = {
                "checked": True,
                "risk": "medium",
                "decision": "allow_targeted_replay_only",
                "traction_audit_not_needed_reason": "Small replay follows a human signal.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("human_signals[1].raw_signal must be a non-empty string", result.stderr)

    def test_validate_learning_action_plan_rejects_ignored_blocking_human_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_judgment",
                "--raw-signal",
                "Keyword coverage is misleading; do not optimize the prompt for it.",
                "--suggested-outcome",
                "downgrade_metric_to_diagnostic",
                "--blocked-action",
                "optimize_prompt_for_keyword_coverage",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "A different repeated pattern may justify a small replayed prompt experiment.",
                "--suggested-outcome",
                "compact_prompt_candidate",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "compact_prompt_candidate"
            decision["human_signal_refs"] = ["human_signals.jsonl#2"]
            decision["blocked_actions"] = [
                "prompt_patch_without_replay",
                "add_case_specific_if_rule",
            ]
            decision["next_actions"] = ["run targeted replay before scoped prompt mutation"]
            decision["replay_targets"] = ["promptfoo:therapy-tone contrast batch"]
            decision["prompt_bloat_gate"] = {
                "checked": True,
                "risk": "medium",
                "decision": "allow_targeted_replay_only",
                "traction_audit_not_needed_reason": "Small replay follows a human signal.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "decision.blocked_actions must include reviewed human signal blocked_action: "
                "optimize_prompt_for_keyword_coverage",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_prompt_mutation_with_conflicting_human_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_stop_rule",
                "--raw-signal",
                "This is good enough; stop tuning narrow failures.",
                "--suggested-outcome",
                "stop_tuning",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "compact_prompt_candidate"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = [
                "prompt_patch_without_replay",
                "add_case_specific_if_rule",
            ]
            decision["next_actions"] = ["run targeted replay before scoped prompt mutation"]
            decision["replay_targets"] = ["promptfoo:therapy-tone contrast batch"]
            decision["prompt_bloat_gate"] = {
                "checked": True,
                "risk": "medium",
                "decision": "allow_targeted_replay_only",
                "traction_audit_not_needed_reason": "Small replay follows a human signal.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "human_signal_refs does not support decision_type compact_prompt_candidate",
                result.stderr,
            )

    def test_audit_dataset_traction_flags_action_step_pressure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "biased.json"
            audit_path = Path(tmp) / "traction_audit.json"
            write_json(
                dataset,
                {
                    "records": [
                        {
                            "id": "case_1",
                            "input": "I had a rough day.",
                            "target_behavior": ["offer one practical next step right away"],
                            "avoid_behavior": ["sitting quietly with the feeling"],
                        },
                        {
                            "id": "case_2",
                            "input": "I feel stuck.",
                            "target_behavior": ["include one concrete action"],
                            "avoid_behavior": ["no extra reflection"],
                        },
                        {
                            "id": "case_3",
                            "input": "I'm tired.",
                            "target_behavior": ["suggest a simple activity"],
                            "avoid_behavior": ["quiet companionship"],
                        },
                    ]
                },
            )

            result = run_script(
                "audit_dataset_traction.py",
                str(dataset),
                "--output",
                str(audit_path),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["recommended_next_action"], "revise_or_supplement_eval_first")
            self.assertIn("action_step_pressure", {warning["code"] for warning in audit["warnings"]})

    def test_validate_learning_action_plan_rejects_prompt_tuning_blocked_by_traction_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            audit_path = Path(tmp) / "traction_audit.json"
            write_json(
                audit_path,
                {
                    "recommended_next_action": "revise_or_supplement_eval_first",
                    "warnings": [{"code": "action_step_pressure", "severity": "high"}],
                },
            )
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "The failing cases over-reward action steps.",
                "--suggested-outcome",
                "revise_or_supplement_eval_first",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "revise_prompt_boundary"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["prompt_patch_without_replay"]
            decision["next_actions"] = ["run targeted replay before scoped prompt mutation"]
            decision["replay_targets"] = ["promptfoo:action-step batch"]
            decision["prompt_bloat_gate"] = {
                "checked": True,
                "risk": "high",
                "decision": "allow_prompt_tuning",
                "traction_audit_ref": str(audit_path),
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "traction audit recommends revising eval before prompt mutation",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_eval_revision_without_revision_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            run_dir = workspace / "run"
            audit_path = workspace / "traction_audit.json"
            state_path = workspace / "learning_state.v1.json"
            events_path = workspace / "events.jsonl"
            write_json(
                audit_path,
                {
                    "recommended_next_action": "revise_or_supplement_eval_first",
                    "warnings": [{"code": "action_step_pressure", "severity": "high"}],
                    "blocked_actions": ["prompt_patch_without_replay"],
                },
            )
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "revise_eval_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "dataset=local/biased_eval.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "The eval pack over-rewards action steps; revise the eval before prompt tuning.",
                "--suggested-outcome",
                "revise_or_supplement_eval_first",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state_result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state_path),
                "--active-loop",
                "revise_eval_run",
                "--primary-outcome",
                "revise_or_supplement_eval_first",
                "--next-replay-target",
                "eval_revision:action-step-countercases",
                "--blocked-action",
                "prompt_patch_without_replay",
            )
            self.assertEqual(state_result.returncode, 0, state_result.stderr)
            events_path.write_text(
                json.dumps(
                    {
                        "event_id": "evt_revise_eval_1",
                        "run_id": "revise_eval_run",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "human_signal_captured",
                        "actor": "test",
                        "decision": {
                            "to": "revise_or_supplement_eval_first",
                            "decision_id": "revise_eval_run_decision",
                            "human_signal_refs": ["human_signals.jsonl#1"],
                            "reason": "The traction audit found action-step pressure.",
                        },
                        "evidence": {
                            "kind": "human_signal",
                            "summary": "User and audit require eval revision before prompt tuning.",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "revise_or_supplement_eval_first"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["prompt_patch_without_replay"]
            decision["traction_audit_ref"] = str(audit_path)
            decision["learning_state_ref"] = str(state_path)
            decision["event_refs"] = [f"{events_path}#evt_revise_eval_1"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "revise_or_supplement_eval_first requires eval_revision_targets or replay_targets",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_placeholder_eval_revision_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            run_dir = workspace / "run"
            audit_path = workspace / "traction_audit.json"
            state_path = workspace / "learning_state.v1.json"
            events_path = workspace / "events.jsonl"
            write_json(
                audit_path,
                {
                    "recommended_next_action": "revise_or_supplement_eval_first",
                    "warnings": [{"code": "action_step_pressure", "severity": "high"}],
                    "blocked_actions": ["prompt_patch_without_replay"],
                },
            )
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "revise_eval_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "dataset=local/biased_eval.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "The eval pack over-rewards action steps; add countercases before prompt tuning.",
                "--suggested-outcome",
                "revise_or_supplement_eval_first",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state_result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state_path),
                "--active-loop",
                "revise_eval_run",
                "--primary-outcome",
                "revise_or_supplement_eval_first",
                "--next-replay-target",
                "eval_revision:action-step-countercases",
                "--blocked-action",
                "prompt_patch_without_replay",
            )
            self.assertEqual(state_result.returncode, 0, state_result.stderr)
            events_path.write_text(
                json.dumps(
                    {
                        "event_id": "evt_revise_eval_1",
                        "run_id": "revise_eval_run",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "human_signal_captured",
                        "actor": "test",
                        "decision": {
                            "to": "revise_or_supplement_eval_first",
                            "decision_id": "revise_eval_run_decision",
                            "human_signal_refs": ["human_signals.jsonl#1"],
                            "reason": "The traction audit found action-step pressure.",
                        },
                        "evidence": {
                            "kind": "human_signal",
                            "summary": "User and audit require eval revision before prompt tuning.",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "revise_or_supplement_eval_first"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["prompt_patch_without_replay"]
            decision["traction_audit_ref"] = str(audit_path)
            decision["eval_revision_targets"] = ["fix later"]
            decision["learning_state_ref"] = str(state_path)
            decision["event_refs"] = [f"{events_path}#evt_revise_eval_1"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "revise_or_supplement_eval_first eval_revision_targets contains placeholder target: fix later",
                result.stderr,
            )

    def test_validate_learning_action_plan_accepts_eval_revision_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            run_dir = workspace / "run"
            audit_path = workspace / "traction_audit.json"
            state_path = workspace / "learning_state.v1.json"
            events_path = workspace / "events.jsonl"
            write_json(
                audit_path,
                {
                    "recommended_next_action": "revise_or_supplement_eval_first",
                    "warnings": [{"code": "action_step_pressure", "severity": "high"}],
                    "blocked_actions": ["prompt_patch_without_replay"],
                },
            )
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "revise_eval_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "dataset=local/biased_eval.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "The eval pack over-rewards action steps; add countercases before prompt tuning.",
                "--suggested-outcome",
                "revise_or_supplement_eval_first",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state_result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state_path),
                "--active-loop",
                "revise_eval_run",
                "--primary-outcome",
                "revise_or_supplement_eval_first",
                "--next-replay-target",
                "eval_revision:action-step-countercases",
                "--blocked-action",
                "prompt_patch_without_replay",
            )
            self.assertEqual(state_result.returncode, 0, state_result.stderr)
            events_path.write_text(
                json.dumps(
                    {
                        "event_id": "evt_revise_eval_1",
                        "run_id": "revise_eval_run",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "human_signal_captured",
                        "actor": "test",
                        "decision": {
                            "to": "revise_or_supplement_eval_first",
                            "decision_id": "revise_eval_run_decision",
                            "human_signal_refs": ["human_signals.jsonl#1"],
                            "reason": "The traction audit found action-step pressure.",
                        },
                        "evidence": {
                            "kind": "human_signal",
                            "summary": "User and audit require eval revision before prompt tuning.",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "revise_or_supplement_eval_first"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["prompt_patch_without_replay"]
            decision["traction_audit_ref"] = str(audit_path)
            decision["eval_revision_targets"] = [
                "eval_revision:action-step-countercases",
                "add countercases that reward quiet reception, not only action steps",
            ]
            decision["learning_state_ref"] = str(state_path)
            decision["event_refs"] = [f"{events_path}#evt_revise_eval_1"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("learning action plan valid", result.stdout)

    def test_validate_learning_action_plan_rejects_eval_revision_that_ignores_state_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            run_dir = workspace / "run"
            audit_path = workspace / "traction_audit.json"
            state_path = workspace / "learning_state.v1.json"
            events_path = workspace / "events.jsonl"
            write_json(
                audit_path,
                {
                    "recommended_next_action": "revise_or_supplement_eval_first",
                    "warnings": [{"code": "action_step_pressure", "severity": "high"}],
                    "blocked_actions": ["prompt_patch_without_replay"],
                },
            )
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "revise_eval_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "dataset=local/biased_eval.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "The eval pack over-rewards action steps; add countercases before prompt tuning.",
                "--suggested-outcome",
                "revise_or_supplement_eval_first",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state_result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state_path),
                "--active-loop",
                "revise_eval_run",
                "--primary-outcome",
                "revise_or_supplement_eval_first",
                "--next-replay-target",
                "eval_revision:action-step-countercases",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--blocked-action",
                "promote_canonical",
            )
            self.assertEqual(state_result.returncode, 0, state_result.stderr)
            events_path.write_text(
                json.dumps(
                    {
                        "event_id": "evt_revise_eval_1",
                        "run_id": "revise_eval_run",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "human_signal_captured",
                        "actor": "test",
                        "decision": {
                            "to": "revise_or_supplement_eval_first",
                            "reason": "The traction audit found action-step pressure.",
                        },
                        "evidence": {
                            "kind": "human_signal",
                            "summary": "User and audit require eval revision before prompt tuning.",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "revise_or_supplement_eval_first"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["prompt_patch_without_replay"]
            decision["traction_audit_ref"] = str(audit_path)
            decision["eval_revision_targets"] = ["add unrelated countercases"]
            decision["learning_state_ref"] = str(state_path)
            decision["event_refs"] = [f"{events_path}#evt_revise_eval_1"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "decision.eval_revision_targets/replay_targets must include learning_state.next_replay_targets: "
                "eval_revision:action-step-countercases",
                result.stderr,
            )
            self.assertIn(
                "decision.blocked_actions must include learning_state.blocked_actions: promote_canonical",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_prompt_tuning_before_requested_traction_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            audit_path = Path(tmp) / "traction_audit.json"
            write_json(
                audit_path,
                {
                    "recommended_next_action": "inspect_before_prompt_tuning",
                    "warnings": [{"code": "exact_shape_pressure", "severity": "medium"}],
                },
            )
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "inspect_first_prompt_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "A compact prompt experiment might help, but the audit says inspect the medium warnings first.",
                "--suggested-outcome",
                "compact_prompt_candidate",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "compact_prompt_candidate"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["prompt_patch_without_replay", "add_case_specific_if_rule"]
            decision["next_actions"] = ["run compact prompt replay"]
            decision["replay_targets"] = ["promptfoo:medium-warning batch"]
            decision["prompt_bloat_gate"] = {
                "checked": True,
                "risk": "medium",
                "decision": "allow_targeted_replay_only",
                "traction_audit_ref": str(audit_path),
                "repeated_failure_basis": "The user sees a repeated medium warning pattern.",
                "non_prompt_alternatives_considered": ["failure_pattern", "rubric_revision", "case_revision"],
                "ordinary_interaction_risk": "medium",
                "removal_condition": "Remove the candidate if inspection does not confirm prompt pressure.",
            }
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "traction audit requires inspect_before_prompt_tuning before prompt mutation",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_inspection_decision_without_next_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            run_dir = workspace / "run"
            audit_path = workspace / "traction_audit.json"
            state_path = workspace / "learning_state.v1.json"
            events_path = workspace / "events.jsonl"
            write_json(
                audit_path,
                {
                    "recommended_next_action": "inspect_before_prompt_tuning",
                    "warnings": [{"code": "exact_shape_pressure", "severity": "medium"}],
                },
            )
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "inspect_first_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "The audit says inspect the medium warnings before tuning the prompt.",
                "--suggested-outcome",
                "inspect_before_prompt_tuning",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state_result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state_path),
                "--active-loop",
                "inspect_first_run",
                "--primary-outcome",
                "inspect_before_prompt_tuning",
                "--next-replay-target",
                "review:medium-warning-output-sample",
                "--blocked-action",
                "prompt_patch_without_replay",
            )
            self.assertEqual(state_result.returncode, 0, state_result.stderr)
            events_path.write_text(
                json.dumps(
                    {
                        "event_id": "evt_inspect_1",
                        "run_id": "inspect_first_run",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "human_signal_captured",
                        "actor": "test",
                        "decision": {
                            "to": "inspect_before_prompt_tuning",
                            "decision_id": "inspect_first_run_decision",
                            "human_signal_refs": ["human_signals.jsonl#1"],
                            "reason": "The traction audit found medium warnings that need review before prompt tuning.",
                        },
                        "evidence": {
                            "kind": "human_signal",
                            "summary": "User and audit both require inspection before prompt tuning.",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "inspect_before_prompt_tuning"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["prompt_patch_without_replay"]
            decision["traction_audit_ref"] = str(audit_path)
            decision["learning_state_ref"] = str(state_path)
            decision["event_refs"] = [f"{events_path}#evt_inspect_1"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "inspect_before_prompt_tuning requires next_review_targets or replay_targets",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_placeholder_inspection_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            run_dir = workspace / "run"
            audit_path = workspace / "traction_audit.json"
            state_path = workspace / "learning_state.v1.json"
            events_path = workspace / "events.jsonl"
            write_json(
                audit_path,
                {
                    "recommended_next_action": "inspect_before_prompt_tuning",
                    "warnings": [{"code": "exact_shape_pressure", "severity": "medium"}],
                },
            )
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "inspect_first_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "Inspect the medium warnings against real outputs before any prompt tuning.",
                "--suggested-outcome",
                "inspect_before_prompt_tuning",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state_result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state_path),
                "--active-loop",
                "inspect_first_run",
                "--primary-outcome",
                "inspect_before_prompt_tuning",
                "--next-replay-target",
                "review:medium-warning-output-sample",
                "--blocked-action",
                "prompt_patch_without_replay",
            )
            self.assertEqual(state_result.returncode, 0, state_result.stderr)
            events_path.write_text(
                json.dumps(
                    {
                        "event_id": "evt_inspect_1",
                        "run_id": "inspect_first_run",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "human_signal_captured",
                        "actor": "test",
                        "decision": {
                            "to": "inspect_before_prompt_tuning",
                            "decision_id": "inspect_first_run_decision",
                            "human_signal_refs": ["human_signals.jsonl#1"],
                            "reason": "The traction audit found medium warnings that need review before prompt tuning.",
                        },
                        "evidence": {
                            "kind": "human_signal",
                            "summary": "User and audit both require inspection before prompt tuning.",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "inspect_before_prompt_tuning"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["prompt_patch_without_replay"]
            decision["traction_audit_ref"] = str(audit_path)
            decision["next_review_targets"] = ["review later"]
            decision["learning_state_ref"] = str(state_path)
            decision["event_refs"] = [f"{events_path}#evt_inspect_1"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "inspect_before_prompt_tuning next_review_targets contains placeholder target: review later",
                result.stderr,
            )

    def test_validate_learning_action_plan_accepts_traction_inspection_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            run_dir = workspace / "run"
            audit_path = workspace / "traction_audit.json"
            state_path = workspace / "learning_state.v1.json"
            events_path = workspace / "events.jsonl"
            write_json(
                audit_path,
                {
                    "recommended_next_action": "inspect_before_prompt_tuning",
                    "warnings": [{"code": "exact_shape_pressure", "severity": "medium"}],
                },
            )
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "inspect_first_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "Inspect the medium warnings against real outputs before any prompt tuning.",
                "--suggested-outcome",
                "inspect_before_prompt_tuning",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state_result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state_path),
                "--active-loop",
                "inspect_first_run",
                "--primary-outcome",
                "inspect_before_prompt_tuning",
                "--next-replay-target",
                "review:medium-warning-output-sample",
                "--blocked-action",
                "prompt_patch_without_replay",
            )
            self.assertEqual(state_result.returncode, 0, state_result.stderr)
            events_path.write_text(
                json.dumps(
                    {
                        "event_id": "evt_inspect_1",
                        "run_id": "inspect_first_run",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "human_signal_captured",
                        "actor": "test",
                        "decision": {
                            "to": "inspect_before_prompt_tuning",
                            "decision_id": "inspect_first_run_decision",
                            "human_signal_refs": ["human_signals.jsonl#1"],
                            "reason": "The traction audit found medium warnings that need review before prompt tuning.",
                        },
                        "evidence": {
                            "kind": "human_signal",
                            "summary": "User and audit both require inspection before prompt tuning.",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "inspect_before_prompt_tuning"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["prompt_patch_without_replay"]
            decision["traction_audit_ref"] = str(audit_path)
            decision["next_review_targets"] = ["review:medium-warning-output-sample"]
            decision["learning_state_ref"] = str(state_path)
            decision["event_refs"] = [f"{events_path}#evt_inspect_1"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("learning action plan valid", result.stdout)

    def test_validate_learning_action_plan_rejects_inspection_that_ignores_state_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            run_dir = workspace / "run"
            audit_path = workspace / "traction_audit.json"
            state_path = workspace / "learning_state.v1.json"
            events_path = workspace / "events.jsonl"
            write_json(
                audit_path,
                {
                    "recommended_next_action": "inspect_before_prompt_tuning",
                    "warnings": [{"code": "exact_shape_pressure", "severity": "medium"}],
                },
            )
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "inspect_first_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
                "--source-file",
                "observations=local/observations.json",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_abstraction_signal",
                "--raw-signal",
                "Inspect the medium warnings against real outputs before any prompt tuning.",
                "--suggested-outcome",
                "inspect_before_prompt_tuning",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state_result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state_path),
                "--active-loop",
                "inspect_first_run",
                "--primary-outcome",
                "inspect_before_prompt_tuning",
                "--next-replay-target",
                "review:medium-warning-output-sample",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--blocked-action",
                "promote_canonical",
            )
            self.assertEqual(state_result.returncode, 0, state_result.stderr)
            events_path.write_text(
                json.dumps(
                    {
                        "event_id": "evt_inspect_1",
                        "run_id": "inspect_first_run",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "human_signal_captured",
                        "actor": "test",
                        "decision": {
                            "to": "inspect_before_prompt_tuning",
                            "reason": "The traction audit found medium warnings that need review before prompt tuning.",
                        },
                        "evidence": {
                            "kind": "human_signal",
                            "summary": "User and audit both require inspection before prompt tuning.",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "inspect_before_prompt_tuning"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["prompt_patch_without_replay"]
            decision["traction_audit_ref"] = str(audit_path)
            decision["next_review_targets"] = ["review:unrelated-output-sample"]
            decision["learning_state_ref"] = str(state_path)
            decision["event_refs"] = [f"{events_path}#evt_inspect_1"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "decision.next_review_targets/replay_targets must include learning_state.next_replay_targets: "
                "review:medium-warning-output-sample",
                result.stderr,
            )
            self.assertIn(
                "decision.blocked_actions must include learning_state.blocked_actions: promote_canonical",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_compression_without_state_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_stop_rule",
                "--raw-signal",
                "This is good enough; stop tuning narrow failures.",
                "--suggested-outcome",
                "stop_tuning",
                "--blocked-action",
                "continue_prompt_tuning_for_narrow_failures",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "stop_tuning"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["continue_prompt_tuning_for_narrow_failures"]
            decision["next_actions"] = ["do not tune remaining low-severity failures"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("compression decision requires learning_state_ref", result.stderr)

    def test_validate_learning_action_plan_rejects_documented_compression_outcome_without_state_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "diagnostic_metric_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_metric_signal",
                "--raw-signal",
                "The metric is useful as a warning but should not drive optimization.",
                "--suggested-outcome",
                "downgrade_metric_to_diagnostic",
                "--blocked-action",
                "optimize_prompt_for_keyword_coverage",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "downgrade_metric_to_diagnostic"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["optimize_prompt_for_keyword_coverage"]
            decision["next_actions"] = ["treat keyword coverage as diagnostic only"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("compression decision requires learning_state_ref", result.stderr)

    def test_validate_learning_action_plan_accepts_compression_with_state_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            state_path = Path(tmp) / "learning_state.v1.json"
            events_path = Path(tmp) / "events.jsonl"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_stop_rule",
                "--raw-signal",
                "This is good enough; stop tuning narrow failures.",
                "--suggested-outcome",
                "stop_tuning",
                "--blocked-action",
                "continue_prompt_tuning_for_narrow_failures",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state_result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state_path),
                "--active-loop",
                "stop_rule_loop",
                "--primary-outcome",
                "stop_tuning",
                "--next-replay-target",
                "none:stop_tuning",
                "--blocked-action",
                "continue_prompt_tuning_for_narrow_failures",
            )
            self.assertEqual(state_result.returncode, 0, state_result.stderr)
            events_path.write_text(
                json.dumps(
                    {
                        "event_id": "evt_stop_1",
                        "run_id": "demo_run",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "human_signal_captured",
                        "actor": "test",
                        "decision": {
                            "to": "stop_tuning",
                            "decision_id": "demo_run_decision",
                            "human_signal_refs": ["human_signals.jsonl#1"],
                            "reason": "Inside acceptable band.",
                        },
                        "evidence": {
                            "kind": "human_signal",
                            "summary": "User said to stop tuning narrow failures.",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "stop_tuning"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["continue_prompt_tuning_for_narrow_failures"]
            decision["next_actions"] = ["do not tune remaining low-severity failures"]
            decision["learning_state_ref"] = str(state_path)
            decision["event_refs"] = [f"{events_path}#evt_stop_1"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("learning action plan valid", result.stdout)

    def test_validate_learning_action_plan_rejects_compression_without_human_signal_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            state_path = Path(tmp) / "learning_state.v1.json"
            events_path = Path(tmp) / "events.jsonl"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state_result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state_path),
                "--active-loop",
                "stop_rule_loop",
                "--primary-outcome",
                "stop_tuning",
                "--next-replay-target",
                "none:stop_tuning",
                "--blocked-action",
                "continue_prompt_tuning_for_narrow_failures",
            )
            self.assertEqual(state_result.returncode, 0, state_result.stderr)
            events_path.write_text(
                json.dumps(
                    {
                        "event_id": "evt_stop_1",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "human_signal_captured",
                        "actor": "test",
                        "decision": {
                            "to": "stop_tuning",
                            "reason": "Inside acceptable band.",
                        },
                        "evidence": {
                            "kind": "human_signal",
                            "summary": "A stop decision was recorded elsewhere.",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "stop_tuning"
            decision["human_signal_refs"] = []
            decision["blocked_actions"] = ["continue_prompt_tuning_for_narrow_failures"]
            decision["next_actions"] = ["do not tune remaining low-severity failures"]
            decision["learning_state_ref"] = str(state_path)
            decision["event_refs"] = [f"{events_path}#evt_stop_1"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("compression decision requires human_signal_refs", result.stderr)

    def test_validate_learning_action_plan_rejects_compression_without_event_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            state_path = Path(tmp) / "learning_state.v1.json"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state_result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state_path),
                "--active-loop",
                "stop_rule_loop",
                "--primary-outcome",
                "stop_tuning",
                "--next-replay-target",
                "none:stop_tuning",
                "--blocked-action",
                "continue_prompt_tuning_for_narrow_failures",
            )
            self.assertEqual(state_result.returncode, 0, state_result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "stop_tuning"
            decision["blocked_actions"] = ["continue_prompt_tuning_for_narrow_failures"]
            decision["next_actions"] = ["do not tune remaining low-severity failures"]
            decision["learning_state_ref"] = str(state_path)
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("compression decision requires event_refs", result.stderr)

    def test_validate_learning_action_plan_rejects_compression_with_conflicting_human_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            state_path = Path(tmp) / "learning_state.v1.json"
            events_path = Path(tmp) / "events.jsonl"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_rubric_signal",
                "--raw-signal",
                "The rubric is unclear; revise rubric instead of stopping.",
                "--suggested-outcome",
                "rubric_revision",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state_result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state_path),
                "--active-loop",
                "stop_rule_loop",
                "--primary-outcome",
                "stop_tuning",
                "--next-replay-target",
                "none:stop_tuning",
                "--blocked-action",
                "continue_prompt_tuning_for_narrow_failures",
            )
            self.assertEqual(state_result.returncode, 0, state_result.stderr)
            events_path.write_text(
                json.dumps(
                    {
                        "event_id": "evt_stop_1",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "human_signal_captured",
                        "actor": "test",
                        "decision": {"to": "stop_tuning", "reason": "Inside acceptable band."},
                        "evidence": {
                            "kind": "human_signal",
                            "summary": "Event supports stop_tuning, signal ref does not.",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "stop_tuning"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["continue_prompt_tuning_for_narrow_failures"]
            decision["next_actions"] = ["do not tune remaining low-severity failures"]
            decision["learning_state_ref"] = str(state_path)
            decision["event_refs"] = [f"{events_path}#evt_stop_1"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "human_signal_refs does not support decision_type stop_tuning",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_state_missing_decision_blocked_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            state_path = Path(tmp) / "learning_state.v1.json"
            events_path = Path(tmp) / "events.jsonl"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_stop_rule",
                "--raw-signal",
                "This is good enough; stop tuning narrow failures.",
                "--suggested-outcome",
                "stop_tuning",
                "--blocked-action",
                "continue_prompt_tuning_for_narrow_failures",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state_result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state_path),
                "--active-loop",
                "stop_rule_loop",
                "--primary-outcome",
                "stop_tuning",
                "--next-replay-target",
                "none:stop_tuning",
            )
            self.assertEqual(state_result.returncode, 0, state_result.stderr)
            events_path.write_text(
                json.dumps(
                    {
                        "event_id": "evt_stop_1",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "human_signal_captured",
                        "actor": "test",
                        "decision": {"to": "stop_tuning", "reason": "Inside acceptable band."},
                        "evidence": {
                            "kind": "human_signal",
                            "summary": "User said to stop tuning narrow failures.",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "stop_tuning"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["continue_prompt_tuning_for_narrow_failures"]
            decision["next_actions"] = ["do not tune remaining low-severity failures"]
            decision["learning_state_ref"] = str(state_path)
            decision["event_refs"] = [f"{events_path}#evt_stop_1"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "learning state missing decision blocked_actions: continue_prompt_tuning_for_narrow_failures",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_unrelated_event_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            state_path = Path(tmp) / "learning_state.v1.json"
            events_path = Path(tmp) / "events.jsonl"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state_result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state_path),
                "--active-loop",
                "stop_rule_loop",
                "--primary-outcome",
                "stop_tuning",
                "--next-replay-target",
                "none:stop_tuning",
                "--blocked-action",
                "continue_prompt_tuning_for_narrow_failures",
            )
            self.assertEqual(state_result.returncode, 0, state_result.stderr)
            events_path.write_text(
                json.dumps(
                    {
                        "event_id": "evt_wrong_1",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "rubric_revised",
                        "actor": "test",
                        "decision": {
                            "to": "rubric_revision",
                            "reason": "Rubric wording changed.",
                        },
                        "evidence": {
                            "kind": "human_review",
                            "summary": "A different event that should not justify stop_tuning.",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "stop_tuning"
            decision["blocked_actions"] = ["continue_prompt_tuning_for_narrow_failures"]
            decision["next_actions"] = ["do not tune remaining low-severity failures"]
            decision["learning_state_ref"] = str(state_path)
            decision["event_refs"] = [f"{events_path}#evt_wrong_1"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "event_refs does not support decision_type stop_tuning",
                result.stderr,
            )

    def test_validate_learning_action_plan_rejects_event_ref_for_other_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            state_path = Path(tmp) / "learning_state.v1.json"
            events_path = Path(tmp) / "events.jsonl"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_stop_rule",
                "--raw-signal",
                "This is good enough; stop tuning narrow failures.",
                "--suggested-outcome",
                "stop_tuning",
                "--blocked-action",
                "continue_prompt_tuning_for_narrow_failures",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state_result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state_path),
                "--active-loop",
                "stop_rule_loop",
                "--primary-outcome",
                "stop_tuning",
                "--next-replay-target",
                "none:stop_tuning",
                "--blocked-action",
                "continue_prompt_tuning_for_narrow_failures",
            )
            self.assertEqual(state_result.returncode, 0, state_result.stderr)
            events_path.write_text(
                json.dumps(
                    {
                        "event_id": "evt_stop_other_run",
                        "run_id": "other_run",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "human_signal_captured",
                        "actor": "test",
                        "decision": {"to": "stop_tuning", "reason": "Inside acceptable band."},
                        "evidence": {
                            "kind": "human_signal",
                            "summary": "A stop decision from another run.",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "stop_tuning"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["continue_prompt_tuning_for_narrow_failures"]
            decision["next_actions"] = ["do not tune remaining low-severity failures"]
            decision["learning_state_ref"] = str(state_path)
            decision["event_refs"] = [f"{events_path}#evt_stop_other_run"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("event_refs run_id does not match decision.run_id", result.stderr)

    def test_validate_learning_action_plan_rejects_event_ref_with_mismatched_decision_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            state_path = Path(tmp) / "learning_state.v1.json"
            events_path = Path(tmp) / "events.jsonl"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_stop_rule",
                "--raw-signal",
                "This is good enough; stop tuning narrow failures.",
                "--suggested-outcome",
                "stop_tuning",
                "--blocked-action",
                "continue_prompt_tuning_for_narrow_failures",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state_result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state_path),
                "--active-loop",
                "stop_rule_loop",
                "--primary-outcome",
                "stop_tuning",
                "--next-replay-target",
                "none:stop_tuning",
                "--blocked-action",
                "continue_prompt_tuning_for_narrow_failures",
            )
            self.assertEqual(state_result.returncode, 0, state_result.stderr)
            events_path.write_text(
                json.dumps(
                    {
                        "event_id": "evt_stop_wrong_context",
                        "run_id": "demo_run",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "human_signal_captured",
                        "actor": "test",
                        "decision": {
                            "to": "stop_tuning",
                            "decision_id": "other_decision",
                            "human_signal_refs": ["human_signals.jsonl#2"],
                            "reason": "Inside acceptable band.",
                        },
                        "evidence": {
                            "kind": "human_signal",
                            "summary": "A stop decision with mismatched refs.",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "stop_tuning"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["continue_prompt_tuning_for_narrow_failures"]
            decision["next_actions"] = ["do not tune remaining low-severity failures"]
            decision["learning_state_ref"] = str(state_path)
            decision["event_refs"] = [f"{events_path}#evt_stop_wrong_context"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("event_refs decision_id does not match decision.decision_id", result.stderr)
            self.assertIn("event_refs human_signal_refs do not match decision.human_signal_refs", result.stderr)

    def test_validate_learning_action_plan_rejects_event_ref_missing_run_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            state_path = Path(tmp) / "learning_state.v1.json"
            events_path = Path(tmp) / "events.jsonl"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_stop_rule",
                "--raw-signal",
                "This is good enough; stop tuning narrow failures.",
                "--suggested-outcome",
                "stop_tuning",
                "--blocked-action",
                "continue_prompt_tuning_for_narrow_failures",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state_result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state_path),
                "--active-loop",
                "stop_rule_loop",
                "--primary-outcome",
                "stop_tuning",
                "--next-replay-target",
                "none:stop_tuning",
                "--blocked-action",
                "continue_prompt_tuning_for_narrow_failures",
            )
            self.assertEqual(state_result.returncode, 0, state_result.stderr)
            events_path.write_text(
                json.dumps(
                    {
                        "event_id": "evt_stop_missing_run_context",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "human_signal_captured",
                        "actor": "test",
                        "decision": {
                            "to": "stop_tuning",
                            "decision_id": "demo_run_decision",
                            "human_signal_refs": ["human_signals.jsonl#1"],
                            "reason": "Inside acceptable band.",
                        },
                        "evidence": {
                            "kind": "human_signal",
                            "summary": "A stop decision without run context.",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "stop_tuning"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["continue_prompt_tuning_for_narrow_failures"]
            decision["next_actions"] = ["do not tune remaining low-severity failures"]
            decision["learning_state_ref"] = str(state_path)
            decision["event_refs"] = [f"{events_path}#evt_stop_missing_run_context"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("event_refs missing run_id for decision context", result.stderr)

    def test_validate_learning_action_plan_rejects_event_ref_missing_decision_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            state_path = Path(tmp) / "learning_state.v1.json"
            events_path = Path(tmp) / "events.jsonl"
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = run_script(
                "record_human_signal.py",
                "--run-dir",
                str(run_dir),
                "--signal-type",
                "human_stop_rule",
                "--raw-signal",
                "This is good enough; stop tuning narrow failures.",
                "--suggested-outcome",
                "stop_tuning",
                "--blocked-action",
                "continue_prompt_tuning_for_narrow_failures",
                "--needs-review",
                "false",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            state_result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state_path),
                "--active-loop",
                "stop_rule_loop",
                "--primary-outcome",
                "stop_tuning",
                "--next-replay-target",
                "none:stop_tuning",
                "--blocked-action",
                "continue_prompt_tuning_for_narrow_failures",
            )
            self.assertEqual(state_result.returncode, 0, state_result.stderr)
            events_path.write_text(
                json.dumps(
                    {
                        "event_id": "evt_stop_missing_context",
                        "run_id": "demo_run",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "human_signal_captured",
                        "actor": "test",
                        "decision": {
                            "to": "stop_tuning",
                            "reason": "Inside acceptable band.",
                        },
                        "evidence": {
                            "kind": "human_signal",
                            "summary": "A generic stop decision without decision context.",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "stop_tuning"
            decision["human_signal_refs"] = ["human_signals.jsonl#1"]
            decision["blocked_actions"] = ["continue_prompt_tuning_for_narrow_failures"]
            decision["next_actions"] = ["do not tune remaining low-severity failures"]
            decision["learning_state_ref"] = str(state_path)
            decision["event_refs"] = [f"{events_path}#evt_stop_missing_context"]
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("event_refs missing decision_id for decision context", result.stderr)
            self.assertIn("event_refs missing human_signal_refs for decision context", result.stderr)

    def test_validate_learning_action_plan_rejects_invalid_learning_state_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            state_path = Path(tmp) / "learning_state.v1.json"
            write_json(state_path, {"last_primary_outcome": "stop_tuning"})
            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(run_dir),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            decision_path = run_dir / "decision.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            decision["decision_type"] = "stop_tuning"
            decision["human_signal_refs"] = []
            decision["blocked_actions"] = ["continue_prompt_tuning_for_narrow_failures"]
            decision["next_actions"] = ["do not tune remaining low-severity failures"]
            decision["learning_state_ref"] = str(state_path)
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

            result = run_script("validate_learning_action_plan.py", str(run_dir))

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("learning state invalid", result.stderr)
            self.assertIn("missing system_id", result.stderr)

    def test_init_eval_run_does_not_accept_non_accept_decisions_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"

            result = run_script(
                "init_eval_run.py",
                "--run-id",
                "demo_run",
                "--project",
                "demo",
                "--output-dir",
                str(output),
                "--decision-type",
                "revise_prompt_boundary",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertIsNone(decision["accepted_direction"])
            self.assertEqual(manifest["source_files"], {})

    def test_batch_import_filters_roles_and_maps_scene_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "star").mkdir()
            (root / "vampire").mkdir()
            for role in ("star", "vampire"):
                (root / role / "test.yaml").write_text(
                    f"""
- vars:
    question: "hello"
  metadata:
    role: {role}
    scene: greeting
  assert:
    - type: llm-rubric
      value: "answer naturally"
""".strip(),
                    encoding="utf-8",
                )
            output = root / "filtered.json"

            result = run_script(
                "batch_import_test_yaml.py",
                str(root),
                str(output),
                "--project",
                "demo",
                "--include-role",
                "star",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["records"]), 1)
            self.assertEqual(payload["records"][0]["role"], "star")
            self.assertEqual(payload["records"][0]["scene_type"], "greeting")
            self.assertEqual(len(payload["filtered"]), 1)

    def test_select_hl_pilot_candidates_filters_and_caps_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset.json"
            output = Path(tmp) / "review_batch.json"
            write_json(
                dataset,
                {
                    "version": "v1",
                    "project": "demo",
                    "dataset_type": "character_prompt_eval",
                    "records": [
                        {
                            "id": "cook_context_001",
                            "layer": "context_drift",
                            "role": "cook",
                            "character_context": "file://cook.md",
                            "scene_type": "emotion_to_practical",
                            "difficulty": "medium",
                            "input": "现在给我一个很小的动作。",
                            "target_behavior": ["shift to one action"],
                            "avoid_behavior": ["long plan"],
                            "tags": ["multi"],
                            "rubric_ref": "context",
                            "source_path": "demo/test.yaml",
                            "source_index": 4,
                            "review_status": "candidate",
                            "conversation_id": "c1",
                            "turn": 2,
                            "legacy_asserts": [{"type": "llm-rubric", "value": "one action"}],
                        },
                        {
                            "id": "dreamer_identity_001",
                            "layer": "identity_boundary",
                            "role": "dreamer",
                            "character_context": "file://dreamer.md",
                            "scene_type": "brief_identity",
                            "difficulty": "easy",
                            "input": "你是谁？一句话。",
                            "target_behavior": ["brief answer"],
                            "avoid_behavior": ["long intro"],
                            "tags": ["single"],
                            "rubric_ref": "identity",
                            "source_path": "demo/test.yaml",
                            "review_status": "candidate",
                        },
                        {
                            "id": "star_context_001",
                            "layer": "context_drift",
                            "role": "star",
                            "character_context": "file://star.md",
                            "scene_type": "emotion_to_practical",
                            "difficulty": "medium",
                            "input": "给我一步。",
                            "target_behavior": ["one step"],
                            "avoid_behavior": ["lecture"],
                            "tags": [],
                            "rubric_ref": "context",
                            "source_path": "demo/test.yaml",
                            "review_status": "candidate",
                        },
                    ],
                },
            )

            result = run_script(
                "select_hl_pilot_candidates.py",
                str(dataset),
                str(output),
                "--role",
                "cook",
                "--role",
                "dreamer",
                "--max-records",
                "1",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["source_dataset"], str(dataset))
            self.assertEqual(payload["selection"]["roles"], ["cook", "dreamer"])
            self.assertEqual(payload["selection"]["max_records"], 1)
            self.assertEqual(len(payload["records"]), 1)
            selected = payload["records"][0]
            self.assertEqual(selected["id"], "cook_context_001")
            self.assertEqual(selected["source_index"], 4)
            self.assertEqual(selected["legacy_asserts"][0]["type"], "llm-rubric")
            self.assertEqual(selected["conversation_id"], "c1")

    def test_select_hl_pilot_candidates_handles_missing_optional_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset.json"
            output = Path(tmp) / "review_batch.json"
            write_json(
                dataset,
                {
                    "version": "v1",
                    "project": "demo",
                    "dataset_type": "character_prompt_eval",
                    "records": [
                        {
                            "id": "cook_minimal_001",
                            "layer": "micro_narrative",
                            "role": "cook",
                            "scene_type": "ordinary_frustration",
                            "difficulty": "medium",
                            "input": "今天有点烦。",
                            "target_behavior": ["receive feeling"],
                            "avoid_behavior": ["generic comfort"],
                            "tags": [],
                            "rubric_ref": "emotional_reception",
                            "source_path": "demo/test.yaml",
                            "review_status": "candidate",
                        }
                    ],
                },
            )

            result = run_script(
                "select_hl_pilot_candidates.py",
                str(dataset),
                str(output),
                "--role",
                "cook",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["records"][0]["id"], "cook_minimal_001")

    def test_select_hl_pilot_candidates_balances_requested_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset.json"
            output = Path(tmp) / "review_batch.json"
            records = []
            for index in range(3):
                records.append(
                    {
                        "id": f"cook_{index}",
                        "layer": "context_drift",
                        "role": "cook",
                        "scene_type": "emotion_to_practical",
                        "difficulty": "medium",
                        "input": f"cook {index}",
                        "target_behavior": ["target"],
                        "avoid_behavior": ["avoid"],
                        "tags": [],
                        "rubric_ref": "context",
                        "source_path": "demo/test.yaml",
                        "review_status": "candidate",
                        "conversation_id": f"cook-{index}",
                    }
                )
            records.append(
                {
                    "id": "dreamer_0",
                    "layer": "context_drift",
                    "role": "dreamer",
                    "scene_type": "emotion_to_practical",
                    "difficulty": "medium",
                    "input": "dreamer",
                    "target_behavior": ["target"],
                    "avoid_behavior": ["avoid"],
                    "tags": [],
                    "rubric_ref": "context",
                    "source_path": "demo/test.yaml",
                    "review_status": "candidate",
                    "conversation_id": "dreamer-0",
                }
            )
            write_json(
                dataset,
                {
                    "version": "v1",
                    "project": "demo",
                    "dataset_type": "character_prompt_eval",
                    "records": records,
                },
            )

            result = run_script(
                "select_hl_pilot_candidates.py",
                str(dataset),
                str(output),
                "--role",
                "cook",
                "--role",
                "dreamer",
                "--max-records",
                "2",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual([record["role"] for record in payload["records"]], ["cook", "dreamer"])

    def test_validate_failure_patterns_accepts_valid_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "demo.v1.json",
                {
                    "id": "demo.v1",
                    "role": "demo",
                    "scope": "project_extension",
                    "failure_modes": [
                        {
                            "id": "template_drift",
                            "description": "Role falls into a generic template.",
                            "signals": ["generic comfort", "ignores user wording"],
                            "linked_records": ["demo_001"],
                            "evidence": [{"kind": "human_review", "summary": "Observed in review."}],
                        }
                    ],
                },
            )
            (root / "README.md").write_text("# ignored\n", encoding="utf-8")

            result = run_script("validate_failure_patterns.py", str(root))

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Validated 1 failure pattern files", result.stdout)

    def test_validate_failure_patterns_accepts_non_role_profile_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "tool_use_timeout.v1.json",
                {
                    "id": "tool_use_timeout.v1",
                    "profile": "tool_use_eval",
                    "scope": "profile_extension",
                    "failure_modes": [
                        {
                            "id": "missing_timeout_recovery",
                            "description": "Agent repeats a failed tool call without fallback.",
                            "signals": ["same tool call retried", "no fallback answer"],
                            "evidence": [{"kind": "run_observation", "summary": "Observed in tool runner."}],
                        }
                    ],
                },
            )

            result = run_script("validate_failure_patterns.py", str(root))

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("Validated 1 failure pattern files", result.stdout)

    def test_validate_failure_patterns_rejects_missing_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "bad.v1.json",
                {
                    "id": "bad.v1",
                    "role": "demo",
                    "scope": "project_extension",
                    "failure_modes": [
                        {
                            "id": "missing_signals",
                            "description": "No signals here.",
                            "evidence": [{"kind": "human_review", "summary": "Observed."}],
                        }
                    ],
                },
            )

            result = run_script("validate_failure_patterns.py", str(root))

            self.assertEqual(result.returncode, 1)
            self.assertIn("failure_modes[0].signals", result.stdout)

    def test_validate_failure_patterns_rejects_unstructured_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "bad_evidence.v1.json",
                {
                    "id": "bad_evidence.v1",
                    "role": "demo",
                    "scope": "project_extension",
                    "failure_modes": [
                        {
                            "id": "unstructured_evidence",
                            "description": "Evidence is not traceable.",
                            "signals": ["repeated stiff reply"],
                            "evidence": ["trust me"],
                        }
                    ],
                },
            )

            result = run_script("validate_failure_patterns.py", str(root))

            self.assertEqual(result.returncode, 1)
            self.assertIn("failure_modes[0].evidence[0] must be an object", result.stdout)

    def test_validate_hl_pilot_outputs_accepts_valid_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = root / "batch.review_batch.json"
            slice_path = root / "slice.v1.json"
            events = root / "events.jsonl"
            write_json(
                batch,
                {
                    "source_dataset": "demo.json",
                    "selection": {"roles": ["demo"], "max_records": 1},
                    "records": [{"id": "demo_001", "input": "hello"}],
                },
            )
            write_json(
                slice_path,
                {
                    "records": [
                        {
                            "id": "core_001",
                            "input": "hello",
                            "source_path": "demo.json",
                            "generic_dimension": "emotional_reception",
                        }
                    ]
                },
            )
            events.write_text(
                json.dumps(
                    {
                        "event_id": "evt_1",
                        "event_type": "experiment_started",
                        "dataset_path": str(batch),
                        "decision": {"reason": "valid"},
                        "evidence": {"kind": "human_review", "summary": "valid"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_script(
                "validate_hl_pilot_outputs.py",
                "--review-batch",
                str(batch),
                "--candidate-slice",
                str(slice_path),
                "--events",
                str(events),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("HL pilot outputs validated", result.stdout)

    def test_validate_hl_pilot_outputs_rejects_bad_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events = Path(tmp) / "events.jsonl"
            events.write_text("{not json}\n", encoding="utf-8")

            result = run_script("validate_hl_pilot_outputs.py", "--events", str(events))

            self.assertEqual(result.returncode, 1)
            self.assertIn("invalid JSONL", result.stdout)

    def test_validate_evolution_events_accepts_valid_event_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset.json"
            events = Path(tmp) / "events.jsonl"
            write_json(dataset, {"records": []})
            events.write_text(
                json.dumps(
                    {
                        "event_id": "evt_1",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "human_signal_captured",
                        "dataset_path": str(dataset),
                        "actor": "test",
                        "decision": {"to": "stop_tuning", "reason": "Inside acceptable band."},
                        "evidence": {"kind": "human_signal", "summary": "User said to stop tuning."},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_script("validate_evolution_events.py", str(events))

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Validated 1 evolution events", result.stdout)

    def test_validate_evolution_events_accepts_profile_adapter_update_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events = Path(tmp) / "events.jsonl"
            events.write_text(
                json.dumps(
                    {
                        "event_id": "evt_profile_adapter_update_1",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "profile_adapter_updated",
                        "actor": "test",
                        "profile_ref": "eval_datasets/profiles/tool_use_eval/README.md",
                        "adapter_ref": "eval_datasets/adapters/custom_tool_runner/README.md",
                        "decision": {
                            "to": "module_notes_updated",
                            "reason": "Tool-use runs need profile and adapter notes before records are generated.",
                        },
                        "evidence": {
                            "kind": "module_ref",
                            "summary": "Profile and adapter notes define the run evidence shape.",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_script("validate_evolution_events.py", str(events), "--skip-path-check")

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Validated 1 evolution events", result.stdout)

    def test_validate_evolution_events_rejects_profile_adapter_update_without_module_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events = Path(tmp) / "events.jsonl"
            events.write_text(
                json.dumps(
                    {
                        "event_id": "evt_profile_adapter_update_1",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "profile_adapter_updated",
                        "actor": "test",
                        "decision": {
                            "to": "module_notes_updated",
                            "reason": "Tool-use runs need module notes.",
                        },
                        "evidence": {
                            "kind": "human_review",
                            "summary": "Reviewer approved the module update.",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_script("validate_evolution_events.py", str(events), "--skip-path-check")

            self.assertEqual(result.returncode, 1)
            self.assertIn("profile_adapter_updated missing profile_ref", result.stdout)
            self.assertIn("profile_adapter_updated missing adapter_ref", result.stdout)
            self.assertIn("profile_adapter_updated requires evidence.kind module_ref", result.stdout)

    def test_validate_evolution_events_rejects_missing_profile_adapter_ref_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events = Path(tmp) / "events.jsonl"
            events.write_text(
                json.dumps(
                    {
                        "event_id": "evt_profile_adapter_update_1",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "profile_adapter_updated",
                        "actor": "test",
                        "profile_ref": str(Path(tmp) / "eval_datasets" / "profiles" / "tool_use_eval" / "README.md"),
                        "adapter_ref": str(
                            Path(tmp) / "eval_datasets" / "adapters" / "custom_tool_runner" / "README.md"
                        ),
                        "decision": {
                            "to": "module_notes_updated",
                            "reason": "Tool-use runs need module notes.",
                        },
                        "evidence": {
                            "kind": "module_ref",
                            "summary": "Profile and adapter notes define the run evidence shape.",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_script("validate_evolution_events.py", str(events))

            self.assertEqual(result.returncode, 1)
            self.assertIn("profile_ref does not exist", result.stdout)
            self.assertIn("adapter_ref does not exist", result.stdout)

    def test_validate_evolution_events_resolves_module_refs_from_eval_datasets_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_ref = root / "eval_datasets" / "profiles" / "tool_use_eval" / "README.md"
            adapter_ref = root / "eval_datasets" / "adapters" / "custom_tool_runner" / "README.md"
            events = root / "eval_datasets" / "evolution" / "events.jsonl"
            profile_ref.parent.mkdir(parents=True)
            adapter_ref.parent.mkdir(parents=True)
            events.parent.mkdir(parents=True)
            profile_ref.write_text("# Tool Use Eval\n", encoding="utf-8")
            adapter_ref.write_text("# Custom Tool Runner\n", encoding="utf-8")
            events.write_text(
                json.dumps(
                    {
                        "event_id": "evt_profile_adapter_update_1",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "profile_adapter_updated",
                        "actor": "test",
                        "profile_ref": "eval_datasets/profiles/tool_use_eval/README.md",
                        "adapter_ref": "eval_datasets/adapters/custom_tool_runner/README.md",
                        "decision": {
                            "to": "module_notes_updated",
                            "reason": "Tool-use runs need module notes.",
                        },
                        "evidence": {
                            "kind": "module_ref",
                            "summary": "Profile and adapter notes define the run evidence shape.",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_script("validate_evolution_events.py", str(events))

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Validated 1 evolution events", result.stdout)

    def test_validate_evolution_events_rejects_duplicate_ids_and_bad_enums(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events = Path(tmp) / "events.jsonl"
            events.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "event_id": "evt_1",
                                "timestamp": "2026-05-27T10:00:00+08:00",
                                "event_type": "unknown_type",
                                "actor": "test",
                                "decision": {"reason": "invalid"},
                                "evidence": {"kind": "unknown_kind", "summary": "invalid"},
                            }
                        ),
                        json.dumps(
                            {
                                "event_id": "evt_1",
                                "timestamp": "2026-05-27T10:01:00+08:00",
                                "event_type": "experiment_started",
                                "actor": "test",
                                "decision": {"reason": "duplicate"},
                                "evidence": {"kind": "human_review", "summary": "duplicate"},
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_script("validate_evolution_events.py", str(events))

            self.assertEqual(result.returncode, 1)
            self.assertIn("invalid event_type: unknown_type", result.stdout)
            self.assertIn("invalid evidence.kind: unknown_kind", result.stdout)
            self.assertIn("duplicate event_id: evt_1", result.stdout)

    def test_validate_hl_pilot_outputs_rejects_slice_without_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            slice_path = Path(tmp) / "slice.v1.json"
            write_json(
                slice_path,
                {
                    "records": [
                        {
                            "id": "core_001",
                            "input": "hello",
                            "generic_dimension": "emotional_reception",
                        }
                    ]
                },
            )

            result = run_script("validate_hl_pilot_outputs.py", "--candidate-slice", str(slice_path))

            self.assertEqual(result.returncode, 1)
            self.assertIn("source_path or source_id", result.stdout)

    def test_validate_hl_learning_state_accepts_valid_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"
            write_json(
                state,
                {
                    "system_id": "heuristic_eval_dataset_system",
                    "version": "v1",
                    "updated_at": "2026-05-19",
                    "active_loop": "hl_pilot",
                    "core_dimensions": [
                        {
                            "id": "emotional_reception",
                            "status": "candidate_evidence",
                            "evidence": ["case_001"],
                        }
                    ],
                    "memory_layers": {"event_log": "events.jsonl"},
                    "known_failure_patterns": [],
                    "open_gaps": ["needs replay"],
                    "reward_weights": {
                        "user_facing_relevance": 0.25,
                        "diagnostic_clarity": 0.2,
                        "cross_project_generality": 0.2,
                        "replay_stability": 0.15,
                        "noise_reduction": 0.1,
                        "compression_value": 0.1,
                    },
                    "allowed_actions": ["create_replay_batch"],
                    "blocked_actions": ["promote_to_accepted_without_replay"],
                    "next_replay_targets": ["promptfoo outputs"],
                    "last_primary_outcome": "failure_pattern_candidate",
                    "acceptable_band": {},
                    "prompt_or_policy_complexity": {},
                },
            )

            result = run_script("validate_hl_learning_state.py", str(state))

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("HL learning state validated", result.stdout)

    def test_validate_hl_learning_state_rejects_missing_reward_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"
            write_json(
                state,
                {
                    "system_id": "heuristic_eval_dataset_system",
                    "version": "v1",
                    "updated_at": "2026-05-19",
                    "active_loop": "hl_pilot",
                    "core_dimensions": [],
                    "memory_layers": {},
                    "known_failure_patterns": [],
                    "open_gaps": [],
                    "reward_weights": {"user_facing_relevance": 1.0},
                    "allowed_actions": [],
                    "blocked_actions": [],
                    "next_replay_targets": [],
                },
            )

            result = run_script("validate_hl_learning_state.py", str(state))

            self.assertEqual(result.returncode, 1)
            self.assertIn("reward_weights.diagnostic_clarity", result.stdout)

    def test_validate_hl_learning_state_rejects_missing_last_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"
            write_json(
                state,
                {
                    "system_id": "heuristic_eval_dataset_system",
                    "version": "v1",
                    "updated_at": "2026-05-19",
                    "active_loop": "hl_pilot",
                    "core_dimensions": [],
                    "memory_layers": {},
                    "known_failure_patterns": [],
                    "open_gaps": [],
                    "reward_weights": {
                        "user_facing_relevance": 0.25,
                        "diagnostic_clarity": 0.2,
                        "cross_project_generality": 0.2,
                        "replay_stability": 0.15,
                        "noise_reduction": 0.1,
                        "compression_value": 0.1,
                    },
                    "allowed_actions": [],
                    "blocked_actions": [],
                    "next_replay_targets": [],
                    "acceptable_band": {},
                    "prompt_or_policy_complexity": {},
                },
            )

            result = run_script("validate_hl_learning_state.py", str(state))

            self.assertEqual(result.returncode, 1)
            self.assertIn("missing last_primary_outcome", result.stdout)

    def test_validate_hl_learning_state_rejects_blocked_primary_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"
            write_json(
                state,
                {
                    "system_id": "heuristic_eval_dataset_system",
                    "version": "v1",
                    "updated_at": "2026-05-19",
                    "active_loop": "hl_pilot",
                    "core_dimensions": [],
                    "memory_layers": {},
                    "known_failure_patterns": [],
                    "open_gaps": [],
                    "reward_weights": {
                        "user_facing_relevance": 0.25,
                        "diagnostic_clarity": 0.2,
                        "cross_project_generality": 0.2,
                        "replay_stability": 0.15,
                        "noise_reduction": 0.1,
                        "compression_value": 0.1,
                    },
                    "allowed_actions": [],
                    "blocked_actions": ["prompt_patch_without_replay"],
                    "next_replay_targets": [],
                    "last_primary_outcome": "prompt_patch_without_replay",
                    "acceptable_band": {},
                    "prompt_or_policy_complexity": {},
                },
            )

            result = run_script("validate_hl_learning_state.py", str(state))

            self.assertEqual(result.returncode, 1)
            self.assertIn(
                "last_primary_outcome repeats blocked action: prompt_patch_without_replay",
                result.stdout,
            )

    def test_validate_hl_learning_state_rejects_conflicting_allowed_and_blocked_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"
            write_json(
                state,
                {
                    "system_id": "heuristic_eval_dataset_system",
                    "version": "v1",
                    "updated_at": "2026-05-19",
                    "active_loop": "hl_pilot",
                    "core_dimensions": [],
                    "memory_layers": {},
                    "known_failure_patterns": [],
                    "open_gaps": [],
                    "reward_weights": {
                        "user_facing_relevance": 0.25,
                        "diagnostic_clarity": 0.2,
                        "cross_project_generality": 0.2,
                        "replay_stability": 0.15,
                        "noise_reduction": 0.1,
                        "compression_value": 0.1,
                    },
                    "allowed_actions": ["prompt_patch_without_replay"],
                    "blocked_actions": ["prompt_patch_without_replay"],
                    "next_replay_targets": [],
                    "last_primary_outcome": "failure_pattern_candidate",
                    "acceptable_band": {},
                    "prompt_or_policy_complexity": {},
                },
            )

            result = run_script("validate_hl_learning_state.py", str(state))

            self.assertEqual(result.returncode, 1)
            self.assertIn(
                "allowed_actions and blocked_actions overlap: prompt_patch_without_replay",
                result.stdout,
            )

    def test_validate_hl_learning_state_rejects_non_stop_outcome_without_next_replay_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"
            write_json(
                state,
                {
                    "system_id": "heuristic_eval_dataset_system",
                    "version": "v1",
                    "updated_at": "2026-05-19",
                    "active_loop": "hl_pilot",
                    "core_dimensions": [],
                    "memory_layers": {},
                    "known_failure_patterns": [],
                    "open_gaps": [],
                    "reward_weights": {
                        "user_facing_relevance": 0.25,
                        "diagnostic_clarity": 0.2,
                        "cross_project_generality": 0.2,
                        "replay_stability": 0.15,
                        "noise_reduction": 0.1,
                        "compression_value": 0.1,
                    },
                    "allowed_actions": [],
                    "blocked_actions": ["prompt_patch_without_replay"],
                    "next_replay_targets": [],
                    "last_primary_outcome": "failure_pattern_candidate",
                    "acceptable_band": {},
                    "prompt_or_policy_complexity": {},
                },
            )

            result = run_script("validate_hl_learning_state.py", str(state))

            self.assertEqual(result.returncode, 1)
            self.assertIn(
                "next_replay_targets must include a target for last_primary_outcome failure_pattern_candidate",
                result.stdout,
            )

    def test_validate_hl_learning_state_rejects_placeholder_next_replay_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"
            write_json(
                state,
                {
                    "system_id": "heuristic_eval_dataset_system",
                    "version": "v1",
                    "updated_at": "2026-05-19",
                    "active_loop": "hl_pilot",
                    "core_dimensions": [],
                    "memory_layers": {},
                    "known_failure_patterns": [],
                    "open_gaps": [],
                    "reward_weights": {
                        "user_facing_relevance": 0.25,
                        "diagnostic_clarity": 0.2,
                        "cross_project_generality": 0.2,
                        "replay_stability": 0.15,
                        "noise_reduction": 0.1,
                        "compression_value": 0.1,
                    },
                    "allowed_actions": [],
                    "blocked_actions": ["prompt_patch_without_replay"],
                    "next_replay_targets": ["later:review"],
                    "last_primary_outcome": "failure_pattern_candidate",
                    "acceptable_band": {},
                    "prompt_or_policy_complexity": {},
                },
            )

            result = run_script("validate_hl_learning_state.py", str(state))

            self.assertEqual(result.returncode, 1)
            self.assertIn(
                "next_replay_targets contains placeholder target: later:review",
                result.stdout,
            )

    def test_validate_hl_learning_state_rejects_stop_tuning_without_terminal_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"
            write_json(
                state,
                {
                    "system_id": "heuristic_eval_dataset_system",
                    "version": "v1",
                    "updated_at": "2026-05-19",
                    "active_loop": "hl_pilot",
                    "core_dimensions": [],
                    "memory_layers": {},
                    "known_failure_patterns": [],
                    "open_gaps": [],
                    "reward_weights": {
                        "user_facing_relevance": 0.25,
                        "diagnostic_clarity": 0.2,
                        "cross_project_generality": 0.2,
                        "replay_stability": 0.15,
                        "noise_reduction": 0.1,
                        "compression_value": 0.1,
                    },
                    "allowed_actions": [],
                    "blocked_actions": ["continue_prompt_tuning_for_narrow_failures"],
                    "next_replay_targets": [],
                    "last_primary_outcome": "stop_tuning",
                    "acceptable_band": {},
                    "prompt_or_policy_complexity": {},
                },
            )

            result = run_script("validate_hl_learning_state.py", str(state))

            self.assertEqual(result.returncode, 1)
            self.assertIn("stop_tuning requires next_replay_targets to include none:stop_tuning", result.stdout)

    def test_validate_hl_learning_state_rejects_candidate_unit_outcome_without_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"
            write_json(
                state,
                {
                    "system_id": "heuristic_eval_dataset_system",
                    "version": "v1",
                    "updated_at": "2026-05-19",
                    "active_loop": "tool_route",
                    "core_dimensions": [],
                    "memory_layers": {},
                    "known_failure_patterns": [],
                    "open_gaps": [],
                    "reward_weights": {
                        "user_facing_relevance": 0.25,
                        "diagnostic_clarity": 0.2,
                        "cross_project_generality": 0.2,
                        "replay_stability": 0.15,
                        "noise_reduction": 0.1,
                        "compression_value": 0.1,
                    },
                    "allowed_actions": [],
                    "blocked_actions": ["prompt_patch_without_replay"],
                    "next_replay_targets": ["tool_trace:argument_schema_failures"],
                    "last_primary_outcome": "create_dataset_candidate_unit",
                    "acceptable_band": {},
                    "prompt_or_policy_complexity": {},
                },
            )

            result = run_script("validate_hl_learning_state.py", str(state))

            self.assertEqual(result.returncode, 1)
            self.assertIn(
                "create_dataset_candidate_unit requires profile and adapter",
                result.stdout,
            )

    def test_validate_hl_learning_state_rejects_profile_adapter_update_without_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"
            write_json(
                state,
                {
                    "system_id": "heuristic_eval_dataset_system",
                    "version": "v1",
                    "updated_at": "2026-05-19",
                    "active_loop": "tool_trace",
                    "core_dimensions": [],
                    "memory_layers": {},
                    "known_failure_patterns": [],
                    "open_gaps": [],
                    "reward_weights": {
                        "user_facing_relevance": 0.25,
                        "diagnostic_clarity": 0.2,
                        "cross_project_generality": 0.2,
                        "replay_stability": 0.15,
                        "noise_reduction": 0.1,
                        "compression_value": 0.1,
                    },
                    "allowed_actions": [],
                    "blocked_actions": ["prompt_patch_without_replay"],
                    "next_replay_targets": ["tool_trace:argument_schema_failures"],
                    "last_primary_outcome": "profile_adapter_update",
                    "acceptable_band": {},
                    "prompt_or_policy_complexity": {},
                },
            )

            result = run_script("validate_hl_learning_state.py", str(state))

            self.assertEqual(result.returncode, 1)
            self.assertIn(
                "profile_adapter_update requires profile and adapter",
                result.stdout,
            )

    def test_validate_hl_learning_state_rejects_profile_adapter_update_without_evidence_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"
            write_json(
                state,
                {
                    "system_id": "heuristic_eval_dataset_system",
                    "version": "v1",
                    "updated_at": "2026-05-19",
                    "active_loop": "tool_trace",
                    "core_dimensions": [],
                    "memory_layers": {},
                    "known_failure_patterns": [],
                    "open_gaps": [],
                    "reward_weights": {
                        "user_facing_relevance": 0.25,
                        "diagnostic_clarity": 0.2,
                        "cross_project_generality": 0.2,
                        "replay_stability": 0.15,
                        "noise_reduction": 0.1,
                        "compression_value": 0.1,
                    },
                    "allowed_actions": [],
                    "blocked_actions": ["prompt_patch_without_replay"],
                    "next_replay_targets": ["tool_trace:argument_schema_failures"],
                    "evidence_refs": [],
                    "last_primary_outcome": "profile_adapter_update",
                    "project": "tool_project",
                    "profile": "tool_use_eval",
                    "adapter": "custom_tool_runner",
                    "acceptable_band": {},
                    "prompt_or_policy_complexity": {},
                },
            )

            result = run_script("validate_hl_learning_state.py", str(state))

            self.assertEqual(result.returncode, 1)
            self.assertIn(
                "profile_adapter_update requires evidence_refs for profile module",
                result.stdout,
            )
            self.assertIn(
                "profile_adapter_update requires evidence_refs for adapter module",
                result.stdout,
            )
            self.assertIn(
                "profile_adapter_update requires validate_run_intake evidence_ref",
                result.stdout,
            )

    def test_validate_hl_learning_state_rejects_profile_adapter_update_with_missing_module_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "eval_datasets" / "experiments" / "tool_trace" / "learning_state.v1.json"
            state.parent.mkdir(parents=True)
            write_json(
                state,
                {
                    "system_id": "heuristic_eval_dataset_system",
                    "version": "v1",
                    "updated_at": "2026-05-19",
                    "active_loop": "tool_trace",
                    "core_dimensions": [],
                    "memory_layers": {},
                    "known_failure_patterns": [],
                    "open_gaps": [],
                    "reward_weights": {
                        "user_facing_relevance": 0.25,
                        "diagnostic_clarity": 0.2,
                        "cross_project_generality": 0.2,
                        "replay_stability": 0.15,
                        "noise_reduction": 0.1,
                        "compression_value": 0.1,
                    },
                    "allowed_actions": [],
                    "blocked_actions": ["prompt_patch_without_replay"],
                    "next_replay_targets": ["tool_trace:argument_schema_failures"],
                    "evidence_refs": [
                        "eval_datasets/profiles/tool_use_eval/README.md",
                        "eval_datasets/adapters/custom_tool_runner/README.md",
                        "validate_run_intake:tool_trace_run:passed",
                    ],
                    "last_primary_outcome": "profile_adapter_update",
                    "project": "tool_project",
                    "profile": "tool_use_eval",
                    "adapter": "custom_tool_runner",
                    "acceptable_band": {},
                    "prompt_or_policy_complexity": {},
                },
            )

            result = run_script("validate_hl_learning_state.py", str(state))

            self.assertEqual(result.returncode, 1)
            self.assertIn(
                "profile_adapter_update profile module evidence_ref does not exist",
                result.stdout,
            )
            self.assertIn(
                "profile_adapter_update adapter module evidence_ref does not exist",
                result.stdout,
            )

    def test_validate_hl_learning_state_rejects_profile_adapter_update_without_event_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_ref = root / "eval_datasets" / "profiles" / "tool_use_eval" / "README.md"
            adapter_ref = root / "eval_datasets" / "adapters" / "custom_tool_runner" / "README.md"
            state = root / "eval_datasets" / "experiments" / "tool_trace" / "learning_state.v1.json"
            profile_ref.parent.mkdir(parents=True)
            adapter_ref.parent.mkdir(parents=True)
            state.parent.mkdir(parents=True)
            profile_ref.write_text("# Tool Use Eval\n", encoding="utf-8")
            adapter_ref.write_text("# Custom Tool Runner\n", encoding="utf-8")
            write_json(
                state,
                {
                    "system_id": "heuristic_eval_dataset_system",
                    "version": "v1",
                    "updated_at": "2026-05-19",
                    "active_loop": "tool_trace",
                    "core_dimensions": [],
                    "memory_layers": {},
                    "known_failure_patterns": [],
                    "open_gaps": [],
                    "reward_weights": {
                        "user_facing_relevance": 0.25,
                        "diagnostic_clarity": 0.2,
                        "cross_project_generality": 0.2,
                        "replay_stability": 0.15,
                        "noise_reduction": 0.1,
                        "compression_value": 0.1,
                    },
                    "allowed_actions": [],
                    "blocked_actions": ["prompt_patch_without_replay"],
                    "next_replay_targets": ["tool_trace:argument_schema_failures"],
                    "evidence_refs": [
                        "eval_datasets/profiles/tool_use_eval/README.md",
                        "eval_datasets/adapters/custom_tool_runner/README.md",
                        "validate_run_intake:tool_trace_run:passed",
                    ],
                    "last_primary_outcome": "profile_adapter_update",
                    "project": "tool_project",
                    "profile": "tool_use_eval",
                    "adapter": "custom_tool_runner",
                    "acceptable_band": {},
                    "prompt_or_policy_complexity": {},
                },
            )

            result = run_script("validate_hl_learning_state.py", str(state))

            self.assertEqual(result.returncode, 1)
            self.assertIn(
                "profile_adapter_update requires profile_adapter_updated event_ref",
                result.stdout,
            )

    def test_validate_hl_learning_state_rejects_profile_adapter_update_with_missing_event_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_ref = root / "eval_datasets" / "profiles" / "tool_use_eval" / "README.md"
            adapter_ref = root / "eval_datasets" / "adapters" / "custom_tool_runner" / "README.md"
            events = root / "eval_datasets" / "evolution" / "events.jsonl"
            state = root / "eval_datasets" / "experiments" / "tool_trace" / "learning_state.v1.json"
            profile_ref.parent.mkdir(parents=True)
            adapter_ref.parent.mkdir(parents=True)
            events.parent.mkdir(parents=True)
            state.parent.mkdir(parents=True)
            profile_ref.write_text("# Tool Use Eval\n", encoding="utf-8")
            adapter_ref.write_text("# Custom Tool Runner\n", encoding="utf-8")
            events.write_text(
                json.dumps(
                    {
                        "event_id": "evt_other",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "profile_adapter_updated",
                        "actor": "test",
                        "profile_ref": "eval_datasets/profiles/tool_use_eval/README.md",
                        "adapter_ref": "eval_datasets/adapters/custom_tool_runner/README.md",
                        "decision": {"to": "profile_adapter_update", "reason": "Module update."},
                        "evidence": {"kind": "module_ref", "summary": "Module refs."},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            write_json(
                state,
                {
                    "system_id": "heuristic_eval_dataset_system",
                    "version": "v1",
                    "updated_at": "2026-05-19",
                    "active_loop": "tool_trace",
                    "core_dimensions": [],
                    "memory_layers": {},
                    "known_failure_patterns": [],
                    "open_gaps": [],
                    "reward_weights": {
                        "user_facing_relevance": 0.25,
                        "diagnostic_clarity": 0.2,
                        "cross_project_generality": 0.2,
                        "replay_stability": 0.15,
                        "noise_reduction": 0.1,
                        "compression_value": 0.1,
                    },
                    "allowed_actions": [],
                    "blocked_actions": ["prompt_patch_without_replay"],
                    "next_replay_targets": ["tool_trace:argument_schema_failures"],
                    "evidence_refs": [
                        "eval_datasets/profiles/tool_use_eval/README.md",
                        "eval_datasets/adapters/custom_tool_runner/README.md",
                        "validate_run_intake:tool_trace_run:passed",
                        "eval_datasets/evolution/events.jsonl#evt_missing",
                    ],
                    "last_primary_outcome": "profile_adapter_update",
                    "project": "tool_project",
                    "profile": "tool_use_eval",
                    "adapter": "custom_tool_runner",
                    "acceptable_band": {},
                    "prompt_or_policy_complexity": {},
                },
            )

            result = run_script("validate_hl_learning_state.py", str(state))

            self.assertEqual(result.returncode, 1)
            self.assertIn(
                "profile_adapter_update event_ref not found: eval_datasets/evolution/events.jsonl#evt_missing",
                result.stdout,
            )

    def test_validate_hl_learning_state_rejects_profile_adapter_update_with_mismatched_event_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_ref = root / "eval_datasets" / "profiles" / "tool_use_eval" / "README.md"
            adapter_ref = root / "eval_datasets" / "adapters" / "custom_tool_runner" / "README.md"
            other_profile_ref = root / "eval_datasets" / "profiles" / "other_profile" / "README.md"
            other_adapter_ref = root / "eval_datasets" / "adapters" / "other_adapter" / "README.md"
            events = root / "eval_datasets" / "evolution" / "events.jsonl"
            state = root / "eval_datasets" / "experiments" / "tool_trace" / "learning_state.v1.json"
            for path in (profile_ref, adapter_ref, other_profile_ref, other_adapter_ref):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"# {path.parent.name}\n", encoding="utf-8")
            events.parent.mkdir(parents=True)
            state.parent.mkdir(parents=True)
            events.write_text(
                json.dumps(
                    {
                        "event_id": "evt_profile_adapter_update_1",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "profile_adapter_updated",
                        "actor": "test",
                        "profile_ref": "eval_datasets/profiles/other_profile/README.md",
                        "adapter_ref": "eval_datasets/adapters/other_adapter/README.md",
                        "decision": {"to": "profile_adapter_update", "reason": "Module update."},
                        "evidence": {"kind": "module_ref", "summary": "Module refs."},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            write_json(
                state,
                {
                    "system_id": "heuristic_eval_dataset_system",
                    "version": "v1",
                    "updated_at": "2026-05-19",
                    "active_loop": "tool_trace",
                    "core_dimensions": [],
                    "memory_layers": {},
                    "known_failure_patterns": [],
                    "open_gaps": [],
                    "reward_weights": {
                        "user_facing_relevance": 0.25,
                        "diagnostic_clarity": 0.2,
                        "cross_project_generality": 0.2,
                        "replay_stability": 0.15,
                        "noise_reduction": 0.1,
                        "compression_value": 0.1,
                    },
                    "allowed_actions": [],
                    "blocked_actions": ["prompt_patch_without_replay"],
                    "next_replay_targets": ["tool_trace:argument_schema_failures"],
                    "evidence_refs": [
                        "eval_datasets/profiles/tool_use_eval/README.md",
                        "eval_datasets/adapters/custom_tool_runner/README.md",
                        "validate_run_intake:tool_trace_run:passed",
                        "eval_datasets/evolution/events.jsonl#evt_profile_adapter_update_1",
                    ],
                    "last_primary_outcome": "profile_adapter_update",
                    "project": "tool_project",
                    "profile": "tool_use_eval",
                    "adapter": "custom_tool_runner",
                    "acceptable_band": {},
                    "prompt_or_policy_complexity": {},
                },
            )

            result = run_script("validate_hl_learning_state.py", str(state))

            self.assertEqual(result.returncode, 1)
            self.assertIn(
                "profile_adapter_update event_ref does not match profile/adapter scope",
                result.stdout,
            )

    def test_record_learning_outcome_creates_valid_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"

            result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state),
                "--active-loop",
                "therapy_tone_compression",
                "--primary-outcome",
                "failure_pattern_candidate",
                "--known-failure-pattern",
                "conversation_therapy_tone.v1",
                "--open-gap",
                "Replay therapy-tone pattern across another role family.",
                "--next-replay-target",
                "promptfoo:therapy-tone contrast batch",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--allowed-action",
                "create_targeted_replay",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(payload["last_primary_outcome"], "failure_pattern_candidate")
            self.assertEqual(payload["active_loop"], "therapy_tone_compression")
            self.assertIn("conversation_therapy_tone.v1", payload["known_failure_patterns"])
            self.assertIn("promptfoo:therapy-tone contrast batch", payload["next_replay_targets"])
            self.assertIn("prompt_patch_without_replay", payload["blocked_actions"])
            self.assertIn("create_targeted_replay", payload["allowed_actions"])

            validation = run_script("validate_hl_learning_state.py", str(state))
            self.assertEqual(validation.returncode, 0, validation.stdout)

    def test_record_learning_outcome_records_run_scope_when_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"

            result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state),
                "--active-loop",
                "tool_trace",
                "--primary-outcome",
                "create_dataset_candidate_unit",
                "--run-id",
                "tool_trace_replay_003",
                "--project",
                "tool_trace_project",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_tool_runner",
                "--next-replay-target",
                "tool_trace:argument_schema_failures",
                "--blocked-action",
                "prompt_patch_without_replay",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], "tool_trace_replay_003")
            self.assertEqual(payload["project"], "tool_trace_project")
            self.assertEqual(payload["profile"], "tool_use_eval")
            self.assertEqual(payload["adapter"], "custom_tool_runner")

    def test_record_learning_outcome_rejects_candidate_unit_without_profile_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"

            result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state),
                "--active-loop",
                "tool_trace",
                "--primary-outcome",
                "create_dataset_candidate_unit",
                "--next-replay-target",
                "tool_trace:argument_schema_failures",
                "--blocked-action",
                "prompt_patch_without_replay",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "primary outcome create_dataset_candidate_unit requires --profile and --adapter",
                result.stderr,
            )
            self.assertFalse(state.exists())

    def test_record_learning_outcome_rejects_profile_adapter_update_without_profile_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"

            result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state),
                "--active-loop",
                "tool_trace",
                "--primary-outcome",
                "profile_adapter_update",
                "--next-replay-target",
                "tool_trace:argument_schema_failures",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "primary outcome profile_adapter_update requires --profile and --adapter",
                result.stderr,
            )
            self.assertFalse(state.exists())

    def test_record_learning_outcome_rejects_profile_adapter_update_without_evidence_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"

            result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state),
                "--active-loop",
                "tool_trace",
                "--primary-outcome",
                "profile_adapter_update",
                "--project",
                "tool_project",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_tool_runner",
                "--next-replay-target",
                "tool_trace:argument_schema_failures",
                "--blocked-action",
                "prompt_patch_without_replay",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "profile_adapter_update requires evidence_refs for profile module",
                result.stderr,
            )
            self.assertIn(
                "profile_adapter_update requires evidence_refs for adapter module",
                result.stderr,
            )
            self.assertIn(
                "profile_adapter_update requires validate_run_intake evidence_ref",
                result.stderr,
            )
            self.assertFalse(state.exists())

    def test_record_learning_outcome_rejects_profile_adapter_update_with_missing_module_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "eval_datasets" / "experiments" / "tool_trace" / "learning_state.v1.json"

            result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state),
                "--active-loop",
                "tool_trace",
                "--primary-outcome",
                "profile_adapter_update",
                "--project",
                "tool_project",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_tool_runner",
                "--next-replay-target",
                "tool_trace:argument_schema_failures",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--evidence-ref",
                "eval_datasets/profiles/tool_use_eval/README.md",
                "--evidence-ref",
                "eval_datasets/adapters/custom_tool_runner/README.md",
                "--evidence-ref",
                "validate_run_intake:tool_trace_run:passed",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "profile_adapter_update profile module evidence_ref does not exist",
                result.stderr,
            )
            self.assertIn(
                "profile_adapter_update adapter module evidence_ref does not exist",
                result.stderr,
            )
            self.assertFalse(state.exists())

    def test_record_learning_outcome_rejects_profile_adapter_update_without_event_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_ref = root / "eval_datasets" / "profiles" / "tool_use_eval" / "README.md"
            adapter_ref = root / "eval_datasets" / "adapters" / "custom_tool_runner" / "README.md"
            state = root / "eval_datasets" / "experiments" / "tool_trace" / "learning_state.v1.json"
            profile_ref.parent.mkdir(parents=True)
            adapter_ref.parent.mkdir(parents=True)
            profile_ref.write_text("# Tool Use Eval\n", encoding="utf-8")
            adapter_ref.write_text("# Custom Tool Runner\n", encoding="utf-8")

            result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state),
                "--active-loop",
                "tool_trace",
                "--primary-outcome",
                "profile_adapter_update",
                "--project",
                "tool_project",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_tool_runner",
                "--next-replay-target",
                "tool_trace:argument_schema_failures",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--evidence-ref",
                "eval_datasets/profiles/tool_use_eval/README.md",
                "--evidence-ref",
                "eval_datasets/adapters/custom_tool_runner/README.md",
                "--evidence-ref",
                "validate_run_intake:tool_trace_run:passed",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "profile_adapter_update requires profile_adapter_updated event_ref",
                result.stderr,
            )
            self.assertFalse(state.exists())

    def test_record_learning_outcome_records_profile_adapter_update_with_event_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_ref = root / "eval_datasets" / "profiles" / "tool_use_eval" / "README.md"
            adapter_ref = root / "eval_datasets" / "adapters" / "custom_tool_runner" / "README.md"
            events = root / "eval_datasets" / "evolution" / "events.jsonl"
            state = root / "eval_datasets" / "experiments" / "tool_trace" / "learning_state.v1.json"
            profile_ref.parent.mkdir(parents=True)
            adapter_ref.parent.mkdir(parents=True)
            events.parent.mkdir(parents=True)
            profile_ref.write_text("# Tool Use Eval\n", encoding="utf-8")
            adapter_ref.write_text("# Custom Tool Runner\n", encoding="utf-8")
            events.write_text(
                json.dumps(
                    {
                        "event_id": "evt_profile_adapter_update_1",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "profile_adapter_updated",
                        "actor": "test",
                        "profile_ref": "eval_datasets/profiles/tool_use_eval/README.md",
                        "adapter_ref": "eval_datasets/adapters/custom_tool_runner/README.md",
                        "decision": {"to": "profile_adapter_update", "reason": "Module update."},
                        "evidence": {"kind": "module_ref", "summary": "Module refs."},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state),
                "--active-loop",
                "tool_trace",
                "--primary-outcome",
                "profile_adapter_update",
                "--project",
                "tool_project",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_tool_runner",
                "--next-replay-target",
                "tool_trace:argument_schema_failures",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--evidence-ref",
                "eval_datasets/profiles/tool_use_eval/README.md",
                "--evidence-ref",
                "eval_datasets/adapters/custom_tool_runner/README.md",
                "--evidence-ref",
                "validate_run_intake:tool_trace_run:passed",
                "--evidence-ref",
                "eval_datasets/evolution/events.jsonl#evt_profile_adapter_update_1",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(payload["last_primary_outcome"], "profile_adapter_update")
            self.assertIn(
                "eval_datasets/evolution/events.jsonl#evt_profile_adapter_update_1",
                payload["evidence_refs"],
            )
            validation = run_script("validate_hl_learning_state.py", str(state))
            self.assertEqual(validation.returncode, 0, validation.stdout)

    def test_record_learning_outcome_rejects_profile_adapter_update_with_mismatched_event_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_ref = root / "eval_datasets" / "profiles" / "tool_use_eval" / "README.md"
            adapter_ref = root / "eval_datasets" / "adapters" / "custom_tool_runner" / "README.md"
            other_profile_ref = root / "eval_datasets" / "profiles" / "other_profile" / "README.md"
            other_adapter_ref = root / "eval_datasets" / "adapters" / "other_adapter" / "README.md"
            events = root / "eval_datasets" / "evolution" / "events.jsonl"
            state = root / "eval_datasets" / "experiments" / "tool_trace" / "learning_state.v1.json"
            for path in (profile_ref, adapter_ref, other_profile_ref, other_adapter_ref):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"# {path.parent.name}\n", encoding="utf-8")
            events.parent.mkdir(parents=True)
            events.write_text(
                json.dumps(
                    {
                        "event_id": "evt_profile_adapter_update_1",
                        "timestamp": "2026-05-27T10:00:00+08:00",
                        "event_type": "profile_adapter_updated",
                        "actor": "test",
                        "profile_ref": "eval_datasets/profiles/other_profile/README.md",
                        "adapter_ref": "eval_datasets/adapters/other_adapter/README.md",
                        "decision": {"to": "profile_adapter_update", "reason": "Module update."},
                        "evidence": {"kind": "module_ref", "summary": "Module refs."},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state),
                "--active-loop",
                "tool_trace",
                "--primary-outcome",
                "profile_adapter_update",
                "--project",
                "tool_project",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_tool_runner",
                "--next-replay-target",
                "tool_trace:argument_schema_failures",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--evidence-ref",
                "eval_datasets/profiles/tool_use_eval/README.md",
                "--evidence-ref",
                "eval_datasets/adapters/custom_tool_runner/README.md",
                "--evidence-ref",
                "validate_run_intake:tool_trace_run:passed",
                "--evidence-ref",
                "eval_datasets/evolution/events.jsonl#evt_profile_adapter_update_1",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "profile_adapter_update event_ref does not match profile/adapter scope",
                result.stderr,
            )
            self.assertFalse(state.exists())

    def test_record_learning_outcome_records_evidence_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"

            result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state),
                "--active-loop",
                "tool_trace",
                "--primary-outcome",
                "create_dataset_candidate_unit",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_tool_runner",
                "--next-replay-target",
                "tool_trace:argument_schema_failures",
                "--blocked-action",
                "prompt_patch_without_replay",
                "--evidence-ref",
                "dataset_candidate_units/tool_error_recovery.v1.json",
                "--evidence-ref",
                "validation:validate_hl_dataset_candidate_units#passed",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["evidence_refs"],
                [
                    "dataset_candidate_units/tool_error_recovery.v1.json",
                    "validation:validate_hl_dataset_candidate_units#passed",
                ],
            )

            validation = run_script("validate_hl_learning_state.py", str(state))
            self.assertEqual(validation.returncode, 0, validation.stdout)

    def test_record_learning_outcome_rejects_blocked_primary_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"

            result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state),
                "--active-loop",
                "blocked_loop",
                "--primary-outcome",
                "prompt_patch_without_replay",
                "--blocked-action",
                "prompt_patch_without_replay",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "primary outcome cannot also be blocked: prompt_patch_without_replay",
                result.stderr,
            )
            self.assertFalse(state.exists())

    def test_record_learning_outcome_rejects_conflicting_allowed_and_blocked_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"

            result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state),
                "--active-loop",
                "conflict_loop",
                "--primary-outcome",
                "needs_review",
                "--allowed-action",
                "prompt_patch_without_replay",
                "--blocked-action",
                "prompt_patch_without_replay",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "allowed_actions and blocked_actions overlap: prompt_patch_without_replay",
                result.stderr,
            )
            self.assertFalse(state.exists())

    def test_record_learning_outcome_rejects_non_stop_outcome_without_next_replay_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"

            result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state),
                "--active-loop",
                "pattern_loop",
                "--primary-outcome",
                "failure_pattern_candidate",
                "--blocked-action",
                "prompt_patch_without_replay",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "primary outcome failure_pattern_candidate requires --next-replay-target",
                result.stderr,
            )
            self.assertFalse(state.exists())

    def test_record_learning_outcome_rejects_placeholder_next_replay_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"

            result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state),
                "--active-loop",
                "pattern_loop",
                "--primary-outcome",
                "failure_pattern_candidate",
                "--next-replay-target",
                "later:review",
                "--blocked-action",
                "prompt_patch_without_replay",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("next replay target contains placeholder: later:review", result.stderr)
            self.assertFalse(state.exists())

    def test_record_learning_outcome_rejects_existing_placeholder_next_replay_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"
            write_json(
                state,
                {
                    "system_id": "heuristic_eval_dataset_system",
                    "version": "v1",
                    "updated_at": "2026-05-19",
                    "active_loop": "old_loop",
                    "core_dimensions": [],
                    "memory_layers": {},
                    "known_failure_patterns": [],
                    "open_gaps": [],
                    "reward_weights": {
                        "user_facing_relevance": 0.25,
                        "diagnostic_clarity": 0.2,
                        "cross_project_generality": 0.2,
                        "replay_stability": 0.15,
                        "noise_reduction": 0.1,
                        "compression_value": 0.1,
                    },
                    "allowed_actions": [],
                    "blocked_actions": ["prompt_patch_without_replay"],
                    "next_replay_targets": ["later:review"],
                    "last_primary_outcome": "failure_pattern_candidate",
                    "acceptable_band": {},
                    "prompt_or_policy_complexity": {},
                },
            )

            result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state),
                "--active-loop",
                "pattern_loop",
                "--primary-outcome",
                "failure_pattern_candidate",
                "--next-replay-target",
                "promptfoo:therapy-tone contrast batch",
                "--blocked-action",
                "prompt_patch_without_replay",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("next replay target contains placeholder: later:review", result.stderr)
            payload = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(payload["next_replay_targets"], ["later:review"])

    def test_record_learning_outcome_rejects_stop_tuning_without_terminal_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"

            result = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state),
                "--active-loop",
                "stop_loop",
                "--primary-outcome",
                "stop_tuning",
                "--blocked-action",
                "continue_prompt_tuning_for_narrow_failures",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("stop_tuning requires --next-replay-target none:stop_tuning", result.stderr)
            self.assertFalse(state.exists())

    def test_record_learning_outcome_updates_state_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "learning_state.v1.json"
            first = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state),
                "--active-loop",
                "loop",
                "--primary-outcome",
                "stop_tuning",
                "--next-replay-target",
                "none:stop_tuning",
                "--blocked-action",
                "continue_prompt_tuning_for_narrow_failures",
            )
            second = run_script(
                "record_learning_outcome.py",
                "--state",
                str(state),
                "--active-loop",
                "loop",
                "--primary-outcome",
                "stop_tuning",
                "--next-replay-target",
                "none:stop_tuning",
                "--blocked-action",
                "continue_prompt_tuning_for_narrow_failures",
            )

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            payload = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(payload["next_replay_targets"], ["none:stop_tuning"])
            self.assertEqual(payload["blocked_actions"], ["continue_prompt_tuning_for_narrow_failures"])

    def test_import_legacy_learning_bootstraps_manifest_and_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "legacy"
            output_dir = root / "bootstrap"
            (legacy / "role_eval" / "testsets" / "canonical").mkdir(parents=True)
            (legacy / "role_eval" / "testsets" / "results" / "full30").mkdir(parents=True)
            (legacy / "tapdoki" / "experiments").mkdir(parents=True)
            write_json(
                legacy / "role_eval" / "testsets" / "canonical" / "old_tests.json",
                {"records": []},
            )
            write_json(
                legacy / "role_eval" / "testsets" / "results" / "full30" / "results.json",
                {"results": []},
            )
            (legacy / "tapdoki" / "experiments" / "eval_summary.md").write_text(
                "# Summary\nRepeated stiffness under simple requests.\n",
                encoding="utf-8",
            )

            result = run_script(
                "import_legacy_learning.py",
                str(legacy),
                "--output-dir",
                str(output_dir),
                "--project",
                "demo",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            state = json.loads((output_dir / "learning_state.v1.json").read_text(encoding="utf-8"))
            decision = json.loads((output_dir / "decision.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["import_type"], "legacy_learning_bootstrap")
            self.assertTrue(manifest["rules"]["legacy_import_is_not_accepted"])
            self.assertEqual(manifest["asset_counts"]["legacy_canonical_json"], 1)
            self.assertEqual(manifest["asset_counts"]["legacy_eval_result"], 1)
            self.assertEqual(manifest["asset_counts"]["legacy_summary"], 1)
            self.assertEqual(state["active_loop"], "legacy_import_bootstrap")
            self.assertIn("mutate_prompt_from_legacy_import_only", state["blocked_actions"])
            self.assertEqual(decision["primary_outcome"], "needs_human_review")

    def test_normalize_legacy_canonical_adds_traceability_and_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "character_eval_dataset.v1.json"
            output = Path(tmp) / "candidate.json"
            write_json(
                source,
                {
                    "version": "v1",
                    "project": "tapdoki",
                    "dataset_type": "character_prompt_eval",
                    "records": [
                        {
                            "id": "star_001",
                            "role": "star",
                            "review_status": "accepted",
                            "input": "hello",
                        }
                    ],
                },
            )

            result = run_script(
                "normalize_legacy_canonical.py",
                str(source),
                str(output),
                "--project",
                "demo",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            record = payload["records"][0]
            self.assertEqual(record["review_status"], "candidate")
            self.assertEqual(record["source_path"], str(source))
            self.assertEqual(record["source_index"], 0)
            self.assertEqual(record["source_id"], "star_001")
            self.assertTrue(payload["import_boundary"]["legacy_import_is_not_accepted"])

    def test_normalize_promptfoo_results_and_failure_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "promptfoo.json"
            summary_path = root / "failures.json"
            output = root / "observations.json"
            write_json(
                result_path,
                {
                    "results": {
                        "results": [
                            {
                                "success": True,
                                "score": 0.9,
                                "vars": {"question": "hi"},
                                "metadata": {
                                    "id": "case_001",
                                    "role": "star",
                                    "scene_type": "greeting",
                                    "prompt_variant": "compact",
                                },
                                "response": {"output": "hello"},
                                "gradingResult": {"reason": "pass"},
                            }
                        ]
                    }
                },
            )
            write_json(
                summary_path,
                [
                    {
                        "id": "case_002",
                        "role": "star",
                        "scene": "message_reply",
                        "question": "reply",
                        "output": "bad",
                        "reason": "too stiff",
                    }
                ],
            )

            result = run_script(
                "normalize_promptfoo_results.py",
                str(result_path),
                str(summary_path),
                "--output",
                str(output),
                "--project",
                "demo",
                "--run-id",
                "run_001",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["total"], 2)
            self.assertEqual(payload["summary"]["successes"], 1)
            self.assertEqual(payload["summary"]["failures"], 1)
            self.assertEqual(payload["observations"][0]["record_id"], "case_001")
            self.assertEqual(payload["observations"][1]["scene_type"], "message_reply")

    def test_normalize_custom_tool_results_writes_trace_observations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "tool_traces.json"
            output = root / "observations.json"
            write_json(
                raw,
                {
                    "version": "v1",
                    "traces": [
                        {
                            "record_id": "tool_route_case_1",
                            "trace_id": "trace_tool_001",
                            "user_task": "Use the internal lookup tool.",
                            "tool_steps": [
                                {
                                    "tool": "internal.lookup",
                                    "arguments": {},
                                    "result": "validation_error",
                                }
                            ],
                            "final_outcome": "The agent asked for the missing status identifier.",
                            "failure_type": "missing_tool_argument_coverage",
                            "failure_tags": ["coverage_gap"],
                            "judge": {
                                "pass": False,
                                "score": 0.0,
                                "reason": "Missing required argument coverage.",
                            },
                            "metadata": {"severity": "medium", "scope": "repeated_pattern"},
                        }
                    ],
                },
            )

            result = run_script(
                "normalize_custom_tool_results.py",
                str(raw),
                "--output",
                str(output),
                "--project",
                "tool_project",
                "--run-id",
                "tool_route_run",
                "--profile",
                "tool_use_eval",
                "--adapter",
                "custom_tool_runner",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["project"], "tool_project")
            self.assertEqual(payload["run_id"], "tool_route_run")
            self.assertEqual(payload["profile"], "tool_use_eval")
            self.assertEqual(payload["adapter"], "custom_tool_runner")
            self.assertEqual(payload["source_paths"], [str(raw)])
            self.assertEqual(payload["summary"]["failures"], 1)
            self.assertEqual(payload["records"][0]["record_id"], "tool_route_case_1")
            self.assertEqual(payload["records"][0]["trace_id"], "trace_tool_001")
            self.assertEqual(payload["records"][0]["input"], "Use the internal lookup tool.")
            self.assertEqual(payload["records"][0]["failure_type"], "missing_tool_argument_coverage")
            self.assertEqual(payload["records"][0]["failure_tags"], ["coverage_gap"])

    def test_run_hl_replay_dry_run_writes_observations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = root / "batch.json"
            context = root / "context.yaml"
            config = root / "config.yaml"
            output = root / "observations.json"
            write_json(
                batch,
                {
                    "records": [
                        {
                            "id": "core_001",
                            "source_record_id": "source_001",
                            "role": "demo",
                            "project": "demo",
                            "generic_dimension": "lightweight_practical_help",
                            "input": "Give me one small step.",
                            "target_behavior": ["one step"],
                            "avoid_behavior": ["long plan"],
                        }
                    ]
                },
            )
            context.write_text(
                """
version: v1
variables:
  series_core:
    value: Demo series.
prompt:
  system_template: |
    {{series_core}}
    {{character_context}}
  user_template: |
    {{input}}
""".strip(),
                encoding="utf-8",
            )
            config.write_text(
                f"""
version: v1
run_id: test_dry_run
input_path: {batch}
output_path: {output}
context_path: {context}
provider:
  type: openai_compatible
  model: demo-model
  api_key_env: MISSING_TEST_API_KEY
judge:
  enabled: false
""".strip(),
                encoding="utf-8",
            )

            result = run_script("run_hl_replay.py", str(config), "--dry-run")

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["records"][0]["record_id"], "core_001")
            self.assertEqual(payload["records"][0]["source_record_id"], "source_001")
            self.assertEqual(payload["records"][0]["output"], "[DRY RUN]")
            self.assertIn("Demo series.", payload["records"][0]["rendered_messages"][0]["content"])

    def test_run_hl_replay_requires_api_key_for_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = root / "batch.json"
            context = root / "context.yaml"
            config = root / "config.yaml"
            output = root / "observations.json"
            write_json(batch, {"records": [{"id": "core_001", "input": "hello"}]})
            context.write_text(
                "version: v1\nprompt:\n  system_template: ''\n  user_template: '{{input}}'\n",
                encoding="utf-8",
            )
            config.write_text(
                f"""
version: v1
run_id: test_missing_env
input_path: {batch}
output_path: {output}
context_path: {context}
provider:
  type: openai_compatible
  model: demo-model
  api_key_env: MISSING_TEST_API_KEY
judge:
  enabled: false
""".strip(),
                encoding="utf-8",
            )

            result = run_script("run_hl_replay.py", str(config))

            self.assertEqual(result.returncode, 1)
            self.assertIn("MISSING_TEST_API_KEY", result.stdout)

    def test_run_hl_replay_dry_run_supports_enabled_judge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = root / "batch.json"
            context = root / "context.yaml"
            config = root / "config.yaml"
            output = root / "observations.json"
            write_json(
                batch,
                {
                    "records": [
                        {
                            "id": "core_001",
                            "input": "hello",
                            "target_behavior": ["brief greeting"],
                            "avoid_behavior": ["long lecture"],
                        }
                    ]
                },
            )
            context.write_text(
                "version: v1\nprompt:\n  user_template: '{{input}}'\n",
                encoding="utf-8",
            )
            config.write_text(
                f"""
version: v1
run_id: test_judge_dry_run
input_path: {batch}
output_path: {output}
context_path: {context}
provider:
  type: openai_compatible
  model: demo-model
  api_key_env: MISSING_TEST_API_KEY
judge:
  enabled: true
""".strip(),
                encoding="utf-8",
            )

            result = run_script("run_hl_replay.py", str(config), "--dry-run")

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(payload["records"][0]["judge"]["enabled"])
            self.assertEqual(payload["records"][0]["judge"]["reason"], "dry run")

    def test_validate_hl_observations_accepts_valid_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            observations = Path(tmp) / "observations.json"
            write_json(
                observations,
                {
                    "version": "v1",
                    "run_id": "demo",
                    "input_path": "batch.json",
                    "context_path": "context.yaml",
                    "dry_run": True,
                    "records": [
                        {
                            "record_id": "core_001",
                            "input": "hello",
                            "output": "[DRY RUN]",
                            "judge": {"enabled": False},
                            "metadata": {"role": "demo"},
                            "failure_tags": [],
                        }
                    ],
                },
            )

            result = run_script("validate_hl_observations.py", str(observations))

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Validated 1 observations", result.stdout)

    def test_validate_hl_observations_rejects_missing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            observations = Path(tmp) / "observations.json"
            write_json(
                observations,
                {
                    "version": "v1",
                    "run_id": "demo",
                    "records": [
                        {
                            "record_id": "core_001",
                            "input": "hello",
                            "judge": {"enabled": False},
                            "metadata": {"role": "demo"},
                            "failure_tags": [],
                        }
                    ],
                },
            )

            result = run_script("validate_hl_observations.py", str(observations))

            self.assertEqual(result.returncode, 1)
            self.assertIn("records[0].output", result.stdout)

    def test_validate_hl_dataset_candidate_units_accepts_valid_unit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            unit = Path(tmp) / "unit.json"
            write_json(
                unit,
                    {
                        "unit_id": "tapdoki_dreamer_exact_help_001",
                        "version": "v1",
                        "profile": "conversation_role",
                        "trigger": {
                            "type": "human_review",
                            "summary": "Dreamer becomes poetic before completing exact-help requests.",
                        },
                    "diagnosis": "both_generic_and_role_specific",
                    "dataset_intent": "Build records that check whether the role completes exact small help before adding flavor.",
                    "candidate_destination": ["conversation_core_candidate", "project_failure_pattern"],
                    "reward_expectation": ["diagnostic_clarity", "noise_reduction"],
                    "replay_requirements": ["same role", "nearby exact-help scenes"],
                    "records": [
                        {
                            "id": "tapdoki_dreamer_exact_help_candidate_001",
                            "layer": "tiny_practical",
                            "role": "dreamer",
                            "scene_type": "message_reply",
                            "difficulty": "medium",
                            "input": "帮我回一句“我今天有点累，晚点再说”，不要解释。",
                            "target_behavior": ["provide one directly usable reply"],
                            "avoid_behavior": ["poetic atmosphere without usable reply"],
                            "tags": ["exact_help"],
                            "rubric_ref": "lightweight_practical_help",
                            "source_path": "eval_datasets/experiments/hl_pilot/dataset_candidate_units/unit.json",
                            "review_status": "candidate",
                        }
                    ],
                },
            )

            result = run_script("validate_hl_dataset_candidate_units.py", str(unit))

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Validated 1 dataset candidate units", result.stdout)

    def test_validate_hl_dataset_candidate_units_rejects_missing_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            unit = Path(tmp) / "unit.json"
            write_json(
                unit,
                {
                    "unit_id": "tapdoki_dreamer_exact_help_001",
                    "version": "v1",
                    "trigger": {
                        "type": "human_review",
                        "summary": "Dreamer becomes poetic before completing exact-help requests.",
                    },
                    "diagnosis": "both_generic_and_role_specific",
                    "dataset_intent": "Build records that check whether the role completes exact small help before adding flavor.",
                    "candidate_destination": ["conversation_core_candidate", "project_failure_pattern"],
                    "reward_expectation": ["diagnostic_clarity", "noise_reduction"],
                    "replay_requirements": ["same role", "nearby exact-help scenes"],
                    "records": [
                        {
                            "id": "tapdoki_dreamer_exact_help_candidate_001",
                            "layer": "tiny_practical",
                            "role": "dreamer",
                            "scene_type": "message_reply",
                            "difficulty": "medium",
                            "input": "帮我回一句“我今天有点累，晚点再说”，不要解释。",
                            "target_behavior": ["provide one directly usable reply"],
                            "avoid_behavior": ["poetic atmosphere without usable reply"],
                            "tags": ["exact_help"],
                            "rubric_ref": "lightweight_practical_help",
                            "source_path": "eval_datasets/experiments/hl_pilot/dataset_candidate_units/unit.json",
                            "review_status": "candidate",
                        }
                    ],
                },
            )

            result = run_script("validate_hl_dataset_candidate_units.py", str(unit))

            self.assertEqual(result.returncode, 1)
            self.assertIn("missing profile", result.stdout)

    def test_validate_hl_dataset_candidate_units_supports_custom_profile_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            unit = Path(tmp) / "tool_unit.json"
            write_json(
                unit,
                {
                    "unit_id": "tool_timeout_recovery_001",
                    "version": "v1",
                    "profile": "tool_use_eval",
                    "trigger": {
                        "type": "run_observation",
                        "summary": "Tool timeout recovery failed across two task families.",
                    },
                    "diagnosis": "failure_pattern_candidate",
                    "dataset_intent": "Build records that check fallback behavior after tool failure.",
                    "candidate_destination": ["profile_extension"],
                    "reward_expectation": ["diagnostic_clarity", "replay_stability"],
                    "replay_requirements": ["same tool family", "different tool family"],
                    "records": [
                        {
                            "id": "tool_timeout_candidate_001",
                            "input_or_task": "Get the weather after the weather tool times out.",
                            "tool_call": "weather.lookup",
                            "expected_behavior": ["explain the timeout and provide a fallback answer"],
                            "quality_signals": ["no blind retry loop", "user-facing fallback"],
                            "source_ref": "runs/custom_tool_runner/tool_run_001/observations.json#1",
                            "review_status": "candidate",
                        }
                    ],
                },
            )

            result = run_script(
                "validate_hl_dataset_candidate_units.py",
                str(unit),
                "--profile",
                "tool_use_eval",
                "--record-field",
                "id",
                "--record-field",
                "input_or_task",
                "--record-field",
                "tool_call",
                "--record-field",
                "expected_behavior",
                "--record-field",
                "quality_signals",
                "--record-field",
                "source_ref",
                "--record-field",
                "review_status",
            )

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("Validated 1 dataset candidate units", result.stdout)

    def test_validate_hl_dataset_candidate_units_reads_required_fields_from_profile_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            unit = Path(tmp) / "tool_unit.json"
            profile_ref = Path(tmp) / "eval_datasets" / "profiles" / "tool_use_eval" / "README.md"
            profile_ref.parent.mkdir(parents=True)
            profile_ref.write_text(
                """
# Tool Use Eval Profile

## Required Fields

- `id`
- `input_or_task`
- `tool_call`
- `expected_behavior`
- `quality_signals`
- `source_ref`
- `review_status`
""".strip()
                + "\n",
                encoding="utf-8",
            )
            write_json(
                unit,
                {
                    "unit_id": "tool_timeout_recovery_001",
                    "version": "v1",
                    "profile": "tool_use_eval",
                    "trigger": {
                        "type": "run_observation",
                        "summary": "Tool timeout recovery failed across two task families.",
                    },
                    "diagnosis": "failure_pattern_candidate",
                    "dataset_intent": "Build records that check fallback behavior after tool failure.",
                    "candidate_destination": ["profile_extension"],
                    "reward_expectation": ["diagnostic_clarity", "replay_stability"],
                    "replay_requirements": ["same tool family", "different tool family"],
                    "records": [
                        {
                            "id": "tool_timeout_candidate_001",
                            "input_or_task": "Get the weather after the weather tool times out.",
                            "tool_call": "weather.lookup",
                            "expected_behavior": ["explain the timeout and provide a fallback answer"],
                            "quality_signals": ["no blind retry loop", "user-facing fallback"],
                            "source_ref": "runs/custom_tool_runner/tool_run_001/observations.json#1",
                            "review_status": "candidate",
                        }
                    ],
                },
            )

            result = run_script(
                "validate_hl_dataset_candidate_units.py",
                str(unit),
                "--profile",
                "tool_use_eval",
                "--profile-ref",
                str(profile_ref),
            )

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("Validated 1 dataset candidate units", result.stdout)

    def test_validate_hl_dataset_candidate_units_rejects_route_backed_record_without_trigger_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            unit = Path(tmp) / "tool_unit.json"
            profile_ref = Path(tmp) / "eval_datasets" / "profiles" / "tool_use_eval" / "README.md"
            profile_ref.parent.mkdir(parents=True)
            profile_ref.write_text(
                """
# Tool Use Eval Profile

## Required Fields

- `id`
- `trace_id`
- `user_task`
- `tool_steps`
- `final_outcome`
- `evidence_ref`
- `review_status`
""".strip()
                + "\n",
                encoding="utf-8",
            )
            write_json(
                unit,
                {
                    "unit_id": "tool_route_candidate_unit_001",
                    "version": "v1",
                    "profile": "tool_use_eval",
                    "trigger": {
                        "type": "observation_route",
                        "summary": "Route evidence selected a small tool-use candidate unit.",
                        "evidence_refs": ["observations.json#case_1"],
                    },
                    "diagnosis": "coverage_gap",
                    "dataset_intent": "Create candidate records for a routed coverage gap.",
                    "candidate_destination": ["profile_extension"],
                    "reward_expectation": ["diagnostic_clarity"],
                    "replay_requirements": ["review:create_dataset_candidate_unit"],
                    "source_route_ref": "route.json",
                    "records": [
                        {
                            "id": "tool_route_candidate_001",
                            "trace_id": "trace_tool_001",
                            "user_task": "Use the internal lookup tool.",
                            "tool_steps": ["internal.lookup missing required argument"],
                            "final_outcome": "Ask for the missing argument.",
                            "evidence_ref": "observations.json#ghost",
                            "review_status": "candidate",
                        }
                    ],
                },
            )

            result = run_script(
                "validate_hl_dataset_candidate_units.py",
                str(unit),
                "--profile",
                "tool_use_eval",
                "--profile-ref",
                str(profile_ref),
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn(
                "records[0].evidence_ref not listed in trigger.evidence_refs",
                result.stdout,
            )

    def test_validate_hl_dataset_candidate_units_accepts_route_backed_traceable_unit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            unit = Path(tmp) / "tool_unit.json"
            profile_ref = Path(tmp) / "eval_datasets" / "profiles" / "tool_use_eval" / "README.md"
            profile_ref.parent.mkdir(parents=True)
            profile_ref.write_text(
                """
# Tool Use Eval Profile

## Required Fields

- `id`
- `trace_id`
- `user_task`
- `tool_steps`
- `final_outcome`
- `evidence_ref`
- `review_status`
""".strip()
                + "\n",
                encoding="utf-8",
            )
            write_json(
                unit,
                {
                    "unit_id": "tool_route_candidate_unit_001",
                    "version": "v1",
                    "profile": "tool_use_eval",
                    "trigger": {
                        "type": "observation_route",
                        "summary": "Route evidence selected a small tool-use candidate unit.",
                        "evidence_refs": ["observations.json#case_1"],
                    },
                    "diagnosis": "coverage_gap",
                    "dataset_intent": "Create candidate records for a routed coverage gap.",
                    "candidate_destination": ["profile_extension"],
                    "reward_expectation": ["diagnostic_clarity"],
                    "replay_requirements": ["review:create_dataset_candidate_unit"],
                    "source_route_ref": "route.json",
                    "records": [
                        {
                            "id": "tool_route_candidate_001",
                            "trace_id": "trace_tool_001",
                            "user_task": "Use the internal lookup tool.",
                            "tool_steps": ["internal.lookup missing required argument"],
                            "final_outcome": "Ask for the missing argument.",
                            "evidence_ref": "observations.json#case_1",
                            "review_status": "candidate",
                        }
                    ],
                },
            )

            result = run_script(
                "validate_hl_dataset_candidate_units.py",
                str(unit),
                "--profile",
                "tool_use_eval",
                "--profile-ref",
                str(profile_ref),
            )

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("Validated 1 dataset candidate units", result.stdout)

    def test_validate_hl_dataset_candidate_units_rejects_wrong_profile_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            unit = Path(tmp) / "tool_unit.json"
            profile_ref = Path(tmp) / "eval_datasets" / "profiles" / "conversation_role" / "README.md"
            profile_ref.parent.mkdir(parents=True)
            profile_ref.write_text(
                """
# Conversation Role Profile

## Required Fields

- `id`
- `input_or_task`
- `tool_call`
- `expected_behavior`
- `quality_signals`
- `source_ref`
- `review_status`
""".strip()
                + "\n",
                encoding="utf-8",
            )
            write_json(
                unit,
                {
                    "unit_id": "tool_timeout_recovery_001",
                    "version": "v1",
                    "profile": "tool_use_eval",
                    "trigger": {
                        "type": "run_observation",
                        "summary": "Tool timeout recovery failed across two task families.",
                    },
                    "diagnosis": "failure_pattern_candidate",
                    "dataset_intent": "Build records that check fallback behavior after tool failure.",
                    "candidate_destination": ["profile_extension"],
                    "reward_expectation": ["diagnostic_clarity", "replay_stability"],
                    "replay_requirements": ["same tool family", "different tool family"],
                    "records": [
                        {
                            "id": "tool_timeout_candidate_001",
                            "input_or_task": "Get the weather after the weather tool times out.",
                            "tool_call": "weather.lookup",
                            "expected_behavior": ["explain the timeout and provide a fallback answer"],
                            "quality_signals": ["no blind retry loop", "user-facing fallback"],
                            "source_ref": "runs/custom_tool_runner/tool_run_001/observations.json#1",
                            "review_status": "candidate",
                        }
                    ],
                },
            )

            result = run_script(
                "validate_hl_dataset_candidate_units.py",
                str(unit),
                "--profile",
                "tool_use_eval",
                "--profile-ref",
                str(profile_ref),
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("profile_ref must point to profiles/tool_use_eval/README.md", result.stdout)

    def test_validate_hl_dataset_candidate_units_record_fields_do_not_weaken_profile_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            unit = Path(tmp) / "tool_unit.json"
            profile_ref = Path(tmp) / "eval_datasets" / "profiles" / "tool_use_eval" / "README.md"
            profile_ref.parent.mkdir(parents=True)
            profile_ref.write_text(
                """
# Tool Use Eval Profile

## Required Fields

- `id`
- `input_or_task`
- `expected_behavior`
- `quality_signals`
- `source_ref`
- `review_status`
""".strip()
                + "\n",
                encoding="utf-8",
            )
            write_json(
                unit,
                {
                    "unit_id": "tool_timeout_recovery_001",
                    "version": "v1",
                    "profile": "tool_use_eval",
                    "trigger": {
                        "type": "run_observation",
                        "summary": "Tool timeout recovery failed across two task families.",
                    },
                    "diagnosis": "failure_pattern_candidate",
                    "dataset_intent": "Build records that check fallback behavior after tool failure.",
                    "candidate_destination": ["profile_extension"],
                    "reward_expectation": ["diagnostic_clarity", "replay_stability"],
                    "replay_requirements": ["same tool family", "different tool family"],
                    "records": [
                        {
                            "id": "tool_timeout_candidate_001",
                            "input_or_task": "Get the weather after the weather tool times out.",
                            "expected_behavior": ["explain the timeout and provide a fallback answer"],
                            "quality_signals": ["no blind retry loop", "user-facing fallback"],
                            "review_status": "candidate",
                        }
                    ],
                },
            )

            result = run_script(
                "validate_hl_dataset_candidate_units.py",
                str(unit),
                "--profile",
                "tool_use_eval",
                "--profile-ref",
                str(profile_ref),
                "--record-field",
                "id",
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("missing records[0].source_ref", result.stdout)

    def test_validate_hl_dataset_candidate_units_list_fields_do_not_weaken_profile_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            unit = Path(tmp) / "tool_unit.json"
            profile_ref = Path(tmp) / "eval_datasets" / "profiles" / "tool_use_eval" / "README.md"
            profile_ref.parent.mkdir(parents=True)
            profile_ref.write_text(
                """
# Tool Use Eval Profile

## Required Fields

- `id`
- `input_or_task`
- `tool_steps`
- `expected_behavior`
- `quality_signals`
- `source_ref`
- `review_status`

## List Fields

- `tool_steps`
""".strip()
                + "\n",
                encoding="utf-8",
            )
            write_json(
                unit,
                {
                    "unit_id": "tool_timeout_recovery_001",
                    "version": "v1",
                    "profile": "tool_use_eval",
                    "trigger": {
                        "type": "run_observation",
                        "summary": "Tool timeout recovery failed across two task families.",
                    },
                    "diagnosis": "failure_pattern_candidate",
                    "dataset_intent": "Build records that check fallback behavior after tool failure.",
                    "candidate_destination": ["profile_extension"],
                    "reward_expectation": ["diagnostic_clarity", "replay_stability"],
                    "replay_requirements": ["same tool family", "different tool family"],
                    "records": [
                        {
                            "id": "tool_timeout_candidate_001",
                            "input_or_task": "Get the weather after the weather tool times out.",
                            "tool_steps": "weather.lookup timed out",
                            "expected_behavior": ["explain timeout and provide fallback answer"],
                            "quality_signals": ["no blind retry loop", "user-facing fallback"],
                            "source_ref": "runs/custom_tool_runner/tool_run_001/observations.json#1",
                            "review_status": "candidate",
                        }
                    ],
                },
            )

            result = run_script(
                "validate_hl_dataset_candidate_units.py",
                str(unit),
                "--profile",
                "tool_use_eval",
                "--profile-ref",
                str(profile_ref),
                "--list-field",
                "quality_signals",
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("records[0].tool_steps must be a list", result.stdout)

    def test_validate_hl_dataset_candidate_units_rejects_profile_ref_without_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            unit = Path(tmp) / "tool_unit.json"
            profile_ref = Path(tmp) / "eval_datasets" / "profiles" / "tool_use_eval" / "README.md"
            profile_ref.parent.mkdir(parents=True)
            profile_ref.write_text("# Tool Use Eval Profile\n\n## Domain Purpose\n\nTool tasks.\n", encoding="utf-8")
            write_json(
                unit,
                {
                    "unit_id": "tool_timeout_recovery_001",
                    "version": "v1",
                    "profile": "tool_use_eval",
                    "trigger": {
                        "type": "run_observation",
                        "summary": "Tool timeout recovery failed across two task families.",
                    },
                    "diagnosis": "failure_pattern_candidate",
                    "dataset_intent": "Build records that check fallback behavior after tool failure.",
                    "candidate_destination": ["profile_extension"],
                    "reward_expectation": ["diagnostic_clarity", "replay_stability"],
                    "replay_requirements": ["same tool family", "different tool family"],
                    "records": [
                        {
                            "id": "tool_timeout_candidate_001",
                            "input_or_task": "Get the weather after the weather tool times out.",
                            "tool_call": "weather.lookup",
                            "expected_behavior": ["explain timeout and provide fallback answer"],
                            "quality_signals": ["no blind retry loop", "user-facing fallback"],
                            "source_ref": "runs/custom_tool_runner/tool_run_001/observations.json#1",
                            "review_status": "candidate",
                        }
                    ],
                },
            )

            result = run_script(
                "validate_hl_dataset_candidate_units.py",
                str(unit),
                "--profile",
                "tool_use_eval",
                "--profile-ref",
                str(profile_ref),
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("profile_ref does not define required record fields", result.stdout)

    def test_validate_hl_dataset_candidate_units_rejects_unedited_scaffold_profile_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            unit = Path(tmp) / "tool_unit.json"
            profile_ref = Path(tmp) / "eval_datasets" / "profiles" / "tool_use_eval" / "README.md"
            profile_ref.parent.mkdir(parents=True)
            profile_ref.write_text(
                """
# Tool Use Eval Profile

## Domain Purpose

Define what this eval domain is trying to keep inside an acceptable experience
band.

## Required Fields

- `id`
- `input_or_task`
- `expected_behavior`
- `quality_signals`
- `source_ref`
- `review_status`
""".strip()
                + "\n",
                encoding="utf-8",
            )
            write_json(
                unit,
                {
                    "unit_id": "tool_timeout_recovery_001",
                    "version": "v1",
                    "profile": "tool_use_eval",
                    "trigger": {
                        "type": "run_observation",
                        "summary": "Tool timeout recovery failed across two task families.",
                    },
                    "diagnosis": "failure_pattern_candidate",
                    "dataset_intent": "Build records that check fallback behavior after tool failure.",
                    "candidate_destination": ["profile_extension"],
                    "reward_expectation": ["diagnostic_clarity", "replay_stability"],
                    "replay_requirements": ["same tool family", "different tool family"],
                    "records": [
                        {
                            "id": "tool_timeout_candidate_001",
                            "input_or_task": "Get the weather after the weather tool times out.",
                            "expected_behavior": ["explain timeout and provide fallback answer"],
                            "quality_signals": ["no blind retry loop", "user-facing fallback"],
                            "source_ref": "runs/custom_tool_runner/tool_run_001/observations.json#1",
                            "review_status": "candidate",
                        }
                    ],
                },
            )

            result = run_script(
                "validate_hl_dataset_candidate_units.py",
                str(unit),
                "--profile",
                "tool_use_eval",
                "--profile-ref",
                str(profile_ref),
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("profile_ref contains scaffold placeholder text", result.stdout)

    def test_validate_hl_dataset_candidate_units_reads_list_fields_from_profile_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            unit = Path(tmp) / "tool_unit.json"
            profile_ref = Path(tmp) / "eval_datasets" / "profiles" / "tool_use_eval" / "README.md"
            profile_ref.parent.mkdir(parents=True)
            profile_ref.write_text(
                """
# Tool Use Eval Profile

## Required Fields

- `id`
- `input_or_task`
- `tool_steps`
- `expected_behavior`
- `quality_signals`
- `source_ref`
- `review_status`

## List Fields

- `tool_steps`
""".strip()
                + "\n",
                encoding="utf-8",
            )
            write_json(
                unit,
                {
                    "unit_id": "tool_timeout_recovery_001",
                    "version": "v1",
                    "profile": "tool_use_eval",
                    "trigger": {
                        "type": "run_observation",
                        "summary": "Tool timeout recovery failed across two task families.",
                    },
                    "diagnosis": "failure_pattern_candidate",
                    "dataset_intent": "Build records that check fallback behavior after tool failure.",
                    "candidate_destination": ["profile_extension"],
                    "reward_expectation": ["diagnostic_clarity", "replay_stability"],
                    "replay_requirements": ["same tool family", "different tool family"],
                    "records": [
                        {
                            "id": "tool_timeout_candidate_001",
                            "input_or_task": "Get the weather after the weather tool times out.",
                            "tool_steps": "weather.lookup timed out",
                            "expected_behavior": ["explain timeout and provide fallback answer"],
                            "quality_signals": ["no blind retry loop", "user-facing fallback"],
                            "source_ref": "runs/custom_tool_runner/tool_run_001/observations.json#1",
                            "review_status": "candidate",
                        }
                    ],
                },
            )

            result = run_script(
                "validate_hl_dataset_candidate_units.py",
                str(unit),
                "--profile",
                "tool_use_eval",
                "--profile-ref",
                str(profile_ref),
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("records[0].tool_steps must be a list", result.stdout)

    def test_validate_hl_dataset_candidate_units_rejects_non_role_profile_without_record_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            unit = Path(tmp) / "tool_unit.json"
            write_json(
                unit,
                {
                    "unit_id": "tool_timeout_recovery_001",
                    "version": "v1",
                    "profile": "tool_use_eval",
                    "trigger": {
                        "type": "run_observation",
                        "summary": "Tool timeout recovery failed across two task families.",
                    },
                    "diagnosis": "failure_pattern_candidate",
                    "dataset_intent": "Build records that check fallback behavior after tool failure.",
                    "candidate_destination": ["profile_extension"],
                    "reward_expectation": ["diagnostic_clarity", "replay_stability"],
                    "replay_requirements": ["same tool family", "different tool family"],
                    "records": [
                        {
                            "id": "tool_timeout_candidate_001",
                            "input_or_task": "Get the weather after the weather tool times out.",
                            "tool_call": "weather.lookup",
                            "expected_behavior": ["explain timeout and provide fallback answer"],
                            "quality_signals": ["no blind retry loop", "user-facing fallback"],
                            "source_ref": "runs/custom_tool_runner/tool_run_001/observations.json#1",
                            "review_status": "candidate",
                        }
                    ],
                },
            )

            result = run_script(
                "validate_hl_dataset_candidate_units.py",
                str(unit),
                "--profile",
                "tool_use_eval",
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn(
                "non-conversation profiles require profile-specific --record-field values",
                result.stdout,
            )
            self.assertNotIn("records[0].role", result.stdout)

    def test_validate_hl_dataset_candidate_units_rejects_missing_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            unit = Path(tmp) / "unit.json"
            write_json(
                unit,
                {
                    "unit_id": "bad_unit",
                    "version": "v1",
                    "trigger": {"type": "human_review", "summary": "Observed."},
                    "diagnosis": "case_issue",
                    "candidate_destination": ["experiment"],
                    "reward_expectation": ["diagnostic_clarity"],
                    "replay_requirements": ["same role"],
                    "records": [],
                },
            )

            result = run_script("validate_hl_dataset_candidate_units.py", str(unit))

            self.assertEqual(result.returncode, 1)
            self.assertIn("dataset_intent", result.stdout)

    def test_score_hl_mutation_writes_reward_assessment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            observations = root / "observations.json"
            output = root / "reward.json"
            write_json(
                observations,
                {
                    "version": "v1",
                    "run_id": "demo_run",
                    "records": [
                        {
                            "record_id": "core_001",
                            "input": "Give one small step.",
                            "output": "Put one shoe next to your foot.",
                            "judge": {"enabled": True, "pass": True, "score": 0.9, "reason": "Usable and brief."},
                            "metadata": {
                                "role": "slowpoke",
                                "generic_dimension": "lightweight_practical_help",
                            },
                            "failure_tags": [],
                        },
                        {
                            "record_id": "core_002",
                            "input": "Reply in one sentence.",
                            "output": "A poetic but unusable reply.",
                            "judge": {"enabled": True, "pass": False, "score": 0.3, "reason": "No usable sentence."},
                            "metadata": {
                                "role": "dreamer",
                                "generic_dimension": "lightweight_practical_help",
                            },
                            "failure_tags": ["task_completion_miss"],
                        },
                    ],
                },
            )

            result = run_script(
                "score_hl_mutation.py",
                str(observations),
                str(output),
                "--mutation-id",
                "demo_mutation",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["mutation_id"], "demo_mutation")
            self.assertIn(payload["decision"], {"keep_experiment", "needs_revision", "compress_candidate"})
            self.assertIn("diagnostic_clarity", payload["scores"])
            self.assertEqual(payload["hard_gates"]["has_replay"], True)

    def test_route_hl_observations_prefers_failure_pattern_for_mixed_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            observations = Path(tmp) / "observations.json"
            output = Path(tmp) / "route.json"
            write_json(
                observations,
                {
                    "version": "v1",
                    "run_id": "mixed_run",
                    "records": [
                        {
                            "record_id": "case_1",
                            "input": "Reply like a friend.",
                            "output": "How does that make you feel?",
                            "judge": {"enabled": True, "pass": False, "score": 0.35},
                            "metadata": {"role": "star", "severity": "medium"},
                            "failure_tags": ["therapy_tone"],
                        },
                        {
                            "record_id": "case_2",
                            "input": "Just answer casually.",
                            "output": "Let us explore your emotions.",
                            "judge": {"enabled": True, "pass": False, "score": 0.4},
                            "metadata": {"role": "cook", "severity": "medium"},
                            "failure_tags": ["therapy_tone"],
                        },
                        {
                            "record_id": "case_3",
                            "input": "Check exact-help behavior.",
                            "output": "No matching test exists.",
                            "judge": {"enabled": True, "pass": False, "score": 0.45},
                            "metadata": {"role": "dreamer", "severity": "medium"},
                            "failure_tags": ["coverage_gap"],
                        },
                        {
                            "record_id": "case_4",
                            "input": "Give a tiny reply.",
                            "output": "Sure, one small step.",
                            "judge": {"enabled": True, "pass": True, "score": 0.9},
                            "metadata": {"role": "star"},
                            "failure_tags": [],
                        },
                    ],
                },
            )

            result = run_script("route_hl_observations.py", str(observations), str(output))

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["primary_outcome"], "failure_pattern_candidate")
            outcomes = {candidate["outcome"] for candidate in payload["outcome_candidates"]}
            self.assertIn("failure_pattern_candidate", outcomes)
            self.assertIn("create_dataset_candidate_unit", outcomes)
            self.assertNotIn("compact_prompt_candidate", outcomes)
            self.assertFalse(payload["prompt_mutation_allowed"])
            self.assertIn("mutate_prompt_or_policy", payload["blocked_actions"])
            self.assertIn("add_case_specific_if_rule", payload["blocked_actions"])

    def test_route_hl_observations_does_not_merge_role_specific_failures_across_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            observations = Path(tmp) / "observations.json"
            output = Path(tmp) / "route.json"
            write_json(
                observations,
                {
                    "version": "v1",
                    "run_id": "role_specific_cross_role_run",
                    "profile": "conversation_role",
                    "adapter": "promptfoo",
                    "records": [
                        {
                            "record_id": "cook_axis_1",
                            "input": "Make the cook feel grounded.",
                            "output": "Generic comfort without taste or temperature.",
                            "judge": {"enabled": True, "pass": False, "score": 0.3},
                            "metadata": {
                                "role": "cook",
                                "learning_scope": "role_specific",
                                "severity": "medium",
                            },
                            "failure_tags": ["role_axis_collapse"],
                        },
                        {
                            "record_id": "vampire_axis_1",
                            "input": "Make the vampire feel grounded.",
                            "output": "Generic comfort without nocturnal restraint.",
                            "judge": {"enabled": True, "pass": False, "score": 0.35},
                            "metadata": {
                                "role": "vampire",
                                "learning_scope": "role_specific",
                                "severity": "medium",
                            },
                            "failure_tags": ["role_axis_collapse"],
                        },
                    ],
                },
            )

            result = run_script("route_hl_observations.py", str(observations), str(output))

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["primary_outcome"], "needs_review")
            self.assertNotIn(
                "failure_pattern_candidate",
                {candidate["outcome"] for candidate in payload["outcome_candidates"]},
            )
            self.assertFalse(payload["prompt_mutation_allowed"])
            self.assertIn("add_case_specific_if_rule", payload["blocked_actions"])

    def test_route_hl_observations_infers_role_specific_scope_from_known_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            observations = Path(tmp) / "observations.json"
            output = Path(tmp) / "route.json"
            write_json(
                observations,
                {
                    "version": "v1",
                    "run_id": "legacy_role_axis_run",
                    "profile": "conversation_role",
                    "adapter": "promptfoo",
                    "records": [
                        {
                            "record_id": "cook_axis_legacy",
                            "input": "Keep the cook's perception axis present.",
                            "output": "Generic comfort without taste or temperature.",
                            "judge": {"enabled": True, "pass": False, "score": 0.3},
                            "metadata": {"role": "cook", "severity": "medium"},
                            "failure_tags": ["role_axis_collapse"],
                        },
                        {
                            "record_id": "vampire_axis_legacy",
                            "input": "Keep the vampire's perception axis present.",
                            "output": "Generic comfort without nocturnal restraint.",
                            "judge": {"enabled": True, "pass": False, "score": 0.35},
                            "metadata": {"role": "vampire", "severity": "medium"},
                            "failure_tags": ["role_axis_collapse"],
                        },
                    ],
                },
            )

            result = run_script("route_hl_observations.py", str(observations), str(output))

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["primary_outcome"], "needs_review")
            self.assertNotIn(
                "failure_pattern_candidate",
                {candidate["outcome"] for candidate in payload["outcome_candidates"]},
            )

    def test_route_hl_observations_preserves_role_specific_pattern_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            observations = Path(tmp) / "observations.json"
            output = Path(tmp) / "route.json"
            write_json(
                observations,
                {
                    "version": "v1",
                    "run_id": "role_specific_single_role_run",
                    "profile": "conversation_role",
                    "adapter": "promptfoo",
                    "records": [
                        {
                            "record_id": "cook_axis_1",
                            "input": "Make the cook answer from taste.",
                            "output": "Generic comfort without taste or temperature.",
                            "judge": {"enabled": True, "pass": False, "score": 0.3},
                            "metadata": {
                                "role": "cook",
                                "learning_scope": "role_specific",
                                "severity": "medium",
                            },
                            "failure_tags": ["role_axis_collapse"],
                        },
                        {
                            "record_id": "cook_axis_2",
                            "input": "Keep the cook's perception axis present.",
                            "output": "A plain helper answer with no cook perception.",
                            "judge": {"enabled": True, "pass": False, "score": 0.35},
                            "metadata": {
                                "role": "cook",
                                "learning_scope": "role_specific",
                                "severity": "medium",
                            },
                            "failure_tags": ["role_axis_collapse"],
                        },
                    ],
                },
            )

            result = run_script("route_hl_observations.py", str(observations), str(output))

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["primary_outcome"], "failure_pattern_candidate")
            pattern = payload["outcome_candidates"][0]
            self.assertEqual(pattern["learning_scope"], "role_specific")
            self.assertEqual(pattern["scope_key"], "role:cook")
            self.assertEqual(
                pattern["evidence_refs"],
                ["observations.json#cook_axis_1", "observations.json#cook_axis_2"],
            )

    def test_route_hl_observations_uses_run_relative_evidence_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            observations = run_dir / "observations.json"
            output = run_dir / "route.json"
            run_dir.mkdir()
            write_json(
                observations,
                {
                    "version": "v1",
                    "run_id": "portable_run",
                    "profile": "conversation_role",
                    "adapter": "promptfoo",
                    "records": [
                        {
                            "record_id": "case_1",
                            "input": "Reply like a friend.",
                            "output": "How does that make you feel?",
                            "judge": {"enabled": True, "pass": False, "score": 0.35},
                            "metadata": {"role": "star", "severity": "medium"},
                            "failure_tags": ["therapy_tone"],
                        },
                        {
                            "record_id": "case_2",
                            "input": "Just answer casually.",
                            "output": "Let us explore your emotions.",
                            "judge": {"enabled": True, "pass": False, "score": 0.4},
                            "metadata": {"role": "cook", "severity": "medium"},
                            "failure_tags": ["therapy_tone"],
                        },
                    ],
                },
            )

            result = run_script("route_hl_observations.py", str(observations), str(output))

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["observation_path"], "observations.json")
            self.assertEqual(payload["profile"], "conversation_role")
            self.assertEqual(payload["adapter"], "promptfoo")
            evidence_refs = payload["outcome_candidates"][0]["evidence_refs"]
            self.assertEqual(evidence_refs, ["observations.json#case_1", "observations.json#case_2"])
            self.assertNotIn(str(run_dir), json.dumps(payload, ensure_ascii=False))

    def test_route_hl_observations_stops_tuning_inside_acceptable_band(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            observations = Path(tmp) / "observations.json"
            output = Path(tmp) / "route.json"
            records = []
            for index in range(6):
                records.append(
                    {
                        "record_id": f"pass_{index}",
                        "input": "Keep it easy.",
                        "output": "A natural short answer.",
                        "judge": {"enabled": True, "pass": True, "score": 0.9},
                        "metadata": {"role": "slowpoke"},
                        "failure_tags": [],
                    }
                )
            records.append(
                {
                    "record_id": "minor_1",
                    "input": "One narrow edge case.",
                    "output": "Slightly stiff but usable.",
                    "judge": {"enabled": True, "pass": False, "score": 0.68},
                    "metadata": {"role": "slowpoke", "severity": "low", "scope": "narrow"},
                    "failure_tags": ["low_severity", "narrow_failure"],
                }
            )
            write_json(observations, {"version": "v1", "run_id": "band_run", "records": records})

            result = run_script("route_hl_observations.py", str(observations), str(output))

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["primary_outcome"], "stop_tuning")
            self.assertGreaterEqual(payload["pass_rate"], 0.85)
            self.assertIn("none:stop_tuning", payload["next_replay_targets"])
            self.assertFalse(payload["prompt_mutation_allowed"])

    def test_score_hl_mutation_supports_profile_specific_generality_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            observations = root / "observations.json"
            output = root / "reward.json"
            write_json(
                observations,
                {
                    "version": "v1",
                    "run_id": "tool_run",
                    "records": [
                        {
                            "record_id": "tool_001",
                            "input": "Call the weather tool.",
                            "output": "Weather API timed out; fallback answer provided.",
                            "judge": {"enabled": True, "pass": True, "score": 0.8},
                            "metadata": {
                                "task_family": "weather_lookup",
                                "generic_dimension": "tool_recovery",
                            },
                            "failure_tags": [],
                        },
                        {
                            "record_id": "tool_002",
                            "input": "Call the calendar tool.",
                            "output": "Calendar API timed out; fallback answer provided.",
                            "judge": {"enabled": True, "pass": True, "score": 0.8},
                            "metadata": {
                                "task_family": "calendar_lookup",
                                "generic_dimension": "tool_recovery",
                            },
                            "failure_tags": [],
                        },
                    ],
                },
            )

            result = run_script(
                "score_hl_mutation.py",
                str(observations),
                str(output),
                "--mutation-id",
                "tool_mutation",
                "--generality-key",
                "task_family",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["generality_key"], "task_family")
            self.assertEqual(payload["scores"]["cross_project_generality"], 2)

    def test_score_hl_mutation_blocks_compression_when_naturalness_or_bloat_risk_is_high(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            observations = Path(tmp) / "observations.json"
            output = Path(tmp) / "reward.json"
            write_json(
                observations,
                {
                    "version": "v1",
                    "run_id": "overfit_run",
                    "records": [
                        {
                            "record_id": "case_1",
                            "input": "Reply naturally.",
                            "output": "I will now comply with the exact optimized rule.",
                            "judge": {"enabled": True, "pass": True, "score": 0.95},
                            "metadata": {
                                "role": "slowpoke",
                                "generic_dimension": "natural_conversation",
                                "naturalness": "low",
                                "prompt_bloat_risk": "high",
                            },
                            "failure_tags": ["stiff_response", "prompt_bloat_risk"],
                        },
                        {
                            "record_id": "case_2",
                            "input": "Say one simple thing.",
                            "output": "Here is the highly constrained answer.",
                            "judge": {"enabled": True, "pass": True, "score": 0.95},
                            "metadata": {
                                "role": "dreamer",
                                "generic_dimension": "natural_conversation",
                                "naturalness": "low",
                            },
                            "failure_tags": ["stiff_response"],
                        },
                        {
                            "record_id": "case_3",
                            "input": "Keep it casual.",
                            "output": "Optimized response pattern satisfied.",
                            "judge": {"enabled": True, "pass": True, "score": 0.95},
                            "metadata": {
                                "role": "cook",
                                "generic_dimension": "natural_conversation",
                                "regression_risk": "high",
                            },
                            "failure_tags": ["case_by_case_overfit"],
                        },
                    ],
                },
            )

            result = run_script("score_hl_mutation.py", str(observations), str(output))

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertNotEqual(payload["decision"], "compress_candidate")
            self.assertFalse(payload["hard_gates"]["risk_signals_clear"])
            self.assertIn("naturalness_low", payload["risk_flags"])
            self.assertIn("prompt_bloat_risk", payload["risk_flags"])
            self.assertIn("regression_risk_high", payload["risk_flags"])

    def test_score_hl_mutation_marks_dry_run_as_not_assessed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            observations = Path(tmp) / "dry_run.observations.json"
            output = Path(tmp) / "reward.json"
            write_json(
                observations,
                {
                    "version": "v1",
                    "run_id": "dry_run",
                    "dry_run": True,
                    "records": [
                        {
                            "record_id": "dry_001",
                            "input": "hello",
                            "output": "[DRY RUN]",
                            "judge": {"enabled": True, "pass": None, "score": None, "reason": "dry run"},
                            "metadata": {"role": "demo", "generic_dimension": "wiring_check"},
                            "failure_tags": [],
                            "dry_run": True,
                        }
                    ],
                },
            )

            result = run_script("score_hl_mutation.py", str(observations), str(output))

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["decision"], "not_assessed")
            self.assertEqual(payload["assessment_level"], "dry_run")
            self.assertFalse(payload["hard_gates"]["has_real_model_output"])
            self.assertFalse(payload["hard_gates"]["has_real_judge_score"])

    def test_score_hl_mutation_marks_unjudged_replay_as_not_assessed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            observations = Path(tmp) / "unjudged.observations.json"
            output = Path(tmp) / "reward.json"
            write_json(
                observations,
                {
                    "version": "v1",
                    "run_id": "unjudged_run",
                    "records": [
                        {
                            "record_id": "case_001",
                            "input": "hello",
                            "output": "A real model output without a judge score.",
                            "judge": {"enabled": False, "pass": None, "score": None},
                            "metadata": {"role": "demo", "generic_dimension": "natural_conversation"},
                            "failure_tags": [],
                        }
                    ],
                },
            )

            result = run_script("score_hl_mutation.py", str(observations), str(output))

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["decision"], "not_assessed")
            self.assertEqual(payload["assessment_level"], "unjudged_replay")
            self.assertTrue(payload["hard_gates"]["has_real_model_output"])
            self.assertFalse(payload["hard_gates"]["has_real_judge_score"])

    def test_score_hl_mutation_rejects_empty_observations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            observations = Path(tmp) / "observations.json"
            output = Path(tmp) / "reward.json"
            write_json(observations, {"version": "v1", "run_id": "empty", "records": []})

            result = run_script("score_hl_mutation.py", str(observations), str(output))

            self.assertEqual(result.returncode, 1)
            self.assertIn("records must be a non-empty list", result.stdout)

    def test_voice_clone_asset_public_profile_files_exist(self) -> None:
        profile = ROOT / "eval_datasets" / "profiles" / "voice_clone_asset"
        expected = {
            "README.md",
            "schema.md",
            "failure_taxonomy.md",
            "routing_rules.yaml",
            "review_protocol.md",
            "calibration.md",
        }

        files = {path.name for path in profile.iterdir() if path.is_file()}
        self.assertTrue(expected.issubset(files), files)

        readme = (profile / "README.md").read_text(encoding="utf-8")
        self.assertIn("source material issue", readme)
        self.assertIn("speaker identity drift", readme)
        self.assertIn("speed or pause regression", readme)
        self.assertIn("stop fixing cloned voices at the wrong layer", readme)

    def test_voice_clone_asset_minimal_examples_are_synthetic_json(self) -> None:
        example = ROOT / "eval_datasets" / "examples" / "voice_clone_asset_minimal"
        json_files = sorted(example.glob("*.json"))

        self.assertGreaterEqual(len(json_files), 5)
        for path in json_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            text = json.dumps(payload)
            self.assertIn("sample_character", text)
            self.assertNotIn("/Users/", text)
            self.assertNotIn("private_voice_evals", text)
            self.assertNotIn(".promptfoo", text)
            self.assertNotIn("secret", text.lower())
            self.assertNotIn("token", text.lower())

            for value in self._walk_json_values(payload):
                if isinstance(value, str) and value.endswith(".wav"):
                    self.assertTrue(value.startswith("local://"), value)

    def test_capability_registry_examples_are_public_safe(self) -> None:
        paths = [
            ROOT / "eval_datasets" / "templates" / "capabilities.local.example.json",
            ROOT
            / "eval_datasets"
            / "examples"
            / "voice_clone_asset_minimal"
            / "capabilities.sample.json",
        ]

        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "heuristic_eval.capability_registry.v0")
            self.assertIn("capabilities", payload)
            self.assertGreaterEqual(len(payload["capabilities"]), 4)
            text = json.dumps(payload)
            forbidden = [
                "/Users/",
                "private_voice_evals",
                ".promptfoo",
                "bearer ",
                "api_key",
            ]
            for needle in forbidden:
                self.assertNotIn(needle, text)
            self.assertNotIn("token", text.lower())
            self.assertNotIn("secret", text.lower())

            for capability in payload["capabilities"]:
                self.assertIn("id", capability)
                self.assertIn("kind", capability)
                self.assertIn("mode", capability)
                self.assertIn("signals", capability)
                self.assertIn("privacy", capability)
                self.assertEqual(capability["privacy"], "local_only")
                self.assertIn("runner", capability)
                self.assertIn("blocked_uses", capability)

    def test_voice_clone_asset_run_example_architecture(self) -> None:
        run_dir = ROOT / "eval_datasets" / "examples" / "voice_clone_asset_minimal" / "run_example"
        expected_files = {
            "manifest.json",
            "outputs.jsonl",
            "observations.jsonl",
            "review_tasks.jsonl",
            "review_results.jsonl",
            "route_decisions.jsonl",
            "summary.json",
            "summary.md",
        }
        self.assertEqual(expected_files, {path.name for path in run_dir.iterdir() if path.is_file()})

        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "voice_clone_asset.voice_output_run.v0")
        self.assertEqual(manifest["profile"], "voice_clone_asset")
        self.assertEqual(manifest["character_id"], "sample_character")
        self.assertIn(manifest["source_type"], {"tts_clone_api", "audio_files", "audio_generation_model", "ab_candidates"})
        self.assertIn(manifest["input_mode"], {"text_to_audio", "audio_only", "prompt_to_audio", "pairwise_audio"})

        outputs = self._read_jsonl(run_dir / "outputs.jsonl")
        observations = self._read_jsonl(run_dir / "observations.jsonl")
        review_tasks = self._read_jsonl(run_dir / "review_tasks.jsonl")
        review_results = self._read_jsonl(run_dir / "review_results.jsonl")
        routes = self._read_jsonl(run_dir / "route_decisions.jsonl")
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(len(outputs), manifest["counts"]["outputs"])
        self.assertEqual(len(observations), manifest["counts"]["observations"])
        self.assertEqual(len(review_tasks), manifest["counts"]["review_tasks"])
        self.assertEqual(len(review_results), manifest["counts"]["review_results"])
        self.assertEqual(len(routes), manifest["counts"]["route_decisions"])
        self.assertEqual(summary["checkpoint_status"], "awaiting_human_review")

        capability_ids = {record["capability_id"] for record in observations}
        self.assertIn("local.audio_qc.vad", capability_ids)
        self.assertIn("local.audio_quality.no_reference", capability_ids)
        self.assertIn("local.audio_judge.speech_attributes", capability_ids)
        self.assertIn("local.audio_judge.naturalness_preference", capability_ids)

        for task in review_tasks:
            self.assertEqual(task["review_mode"], "blind_first")
            self.assertEqual(task["auto_signal_visibility"], "collapsed")
            self.assertEqual(task["return_to_run_id"], manifest["run_id"])
            self.assertIn("checkpoint_id", task)

        result = review_results[0]
        self.assertFalse(result["first_impression"]["auto_signals_seen"])
        self.assertTrue(result["auto_signals_seen"])
        self.assertEqual(result["return_to_run_id"], manifest["run_id"])

        route = routes[0]
        self.assertEqual(route["primary_failure"], "speed_regression")
        self.assertIn("observed_issue", route)
        self.assertIn("suspected_layer", route)
        self.assertIn("confirmed_layer", route)
        self.assertEqual(route["feedback_usefulness"]["status"], "unknown")

    def test_voice_clone_asset_run_example_is_public_safe(self) -> None:
        run_dir = ROOT / "eval_datasets" / "examples" / "voice_clone_asset_minimal" / "run_example"
        combined = "\n".join(path.read_text(encoding="utf-8") for path in sorted(run_dir.iterdir()) if path.is_file())
        forbidden = [
            "/Users/",
            "private_voice_evals",
            ".promptfoo",
            "bearer ",
            "api_key",
        ]
        for needle in forbidden:
            self.assertNotIn(needle, combined)
        self.assertNotIn("token", combined.lower())
        self.assertNotIn("secret", combined.lower())

        for path in sorted(run_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for value in self._walk_json_values(payload):
                if isinstance(value, str) and value.endswith(".wav"):
                    self.assertTrue(value.startswith("local://"), value)

        for path in sorted(run_dir.glob("*.jsonl")):
            for payload in self._read_jsonl(path):
                for value in self._walk_json_values(payload):
                    if isinstance(value, str) and value.endswith(".wav"):
                        self.assertTrue(value.startswith("local://"), value)

    def test_voice_clone_asset_route_blocks_wrong_layer_fixes(self) -> None:
        route = json.loads(
            (
                ROOT
                / "eval_datasets"
                / "examples"
                / "voice_clone_asset_minimal"
                / "route_decision.expected.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(route["primary_failure"], "speed_regression")
        self.assertIn("do_not_add_random_source_audio", route["blocked_actions"])
        self.assertIn("do_not_replace_clone_model_blindly", route["blocked_actions"])
        self.assertIn("rerun_regression_bank", route["next_actions"])
        self.assertTrue(route["human_review_required"])

    def test_review_window_examples_are_audio_ref_only(self) -> None:
        review_window = ROOT / "eval_datasets" / "review_window"
        examples = sorted((review_window / "examples").glob("*.json"))

        self.assertEqual(
            {
                "clone_ab_review_task.sample.json",
                "output_issue_review_task.sample.json",
                "source_clip_review_task.sample.json",
            },
            {path.name for path in examples},
        )
        for path in examples:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["profile"], "voice_clone_asset")
            self.assertIn(payload["task_type"], {"source_clip_review", "clone_ab_review", "output_issue_review"})
            self.assertIn("checkpoint_id", payload)
            self.assertIn("return_to_run_id", payload)
            self.assertEqual(payload["review_mode"], "blind_first")
            self.assertEqual(payload["auto_signal_visibility"], "collapsed")
            self.assertIn("allowed_tags", payload)
            self.assertGreater(len(payload["allowed_tags"]), 0)
            self.assertIn("tag_groups", payload)
            grouped_tags = {
                tag
                for group in payload["tag_groups"]
                for tag in group.get("tags", [])
            }
            self.assertTrue(set(payload["allowed_tags"]).issubset(grouped_tags))
            self.assertIsInstance(payload["audio_refs"], dict)
            self.assertTrue(payload["audio_refs"])
            for audio_ref in payload["audio_refs"].values():
                self.assertTrue(audio_ref.startswith("local://"), audio_ref)

    def test_review_window_web_is_local_file_backed_checkpoint(self) -> None:
        web_dir = ROOT / "eval_datasets" / "review_window" / "web"
        index = web_dir / "index.html"
        readme = web_dir / "README.md"

        self.assertTrue(index.exists())
        self.assertTrue(readme.exists())

        html = index.read_text(encoding="utf-8")
        readme_text = readme.read_text(encoding="utf-8")
        combined = html + "\n" + readme_text

        self.assertIn("review_tasks.jsonl", combined)
        self.assertIn("observations.jsonl", combined)
        self.assertIn("summary.json", combined)
        self.assertIn("review_results.jsonl", combined)
        self.assertIn("blind-first", combined)
        self.assertIn("auto_signals_seen", html)
        self.assertIn("return_to_run_id", html)
        self.assertIn("checkpoint_id", html)
        self.assertIn("Blob", html)
        self.assertIn("URLSearchParams", html)
        self.assertIn("fetchRunFile", html)
        self.assertIn("resolveRunFileUrl", html)
        self.assertIn("translations", html)
        self.assertIn("initialLanguage", html)
        self.assertIn("languageButton", html)
        self.assertIn("status-badge", html)
        self.assertIn("pendingStatus", html)
        self.assertIn("savedStatus", html)
        self.assertIn("currentStatus", html)
        self.assertIn("reviewCacheKey", html)
        self.assertIn("persistReviewCache", html)
        self.assertIn("restoreReviewCache", html)
        self.assertIn("clearDraftButton", html)
        self.assertIn("export-panel", html)
        self.assertIn("exportJsonlText", html)
        self.assertIn("copyExportButton", html)
        self.assertIn("copyExportJsonl", html)
        self.assertIn("ab_dimension_tags", html)
        self.assertIn("abTagGroups", html)
        self.assertIn("A 的标签", html)
        self.assertIn("B 的标签", html)
        self.assertIn("singleTagGroups", html)
        self.assertIn("tag_groups", combined)
        self.assertIn("正向证据", html)
        self.assertIn("问题标签", html)
        self.assertIn("lang=zh", readme_text)
        self.assertIn("lang=en", readme_text)
        self.assertIn("tasks=", readme_text)
        self.assertIn("same-origin", readme_text)
        self.assertIn("评估窗口", html)
        self.assertIn("Review Window", html)
        self.assertNotIn("/Users/", combined)
        self.assertNotIn("private_voice_evals", combined)
        self.assertNotIn(".promptfoo", combined)
        self.assertNotIn("api_key", combined.lower())
        self.assertNotIn("bearer ", combined.lower())

    def test_voice_clone_asset_docs_keep_public_data_boundary(self) -> None:
        paths = [
            ROOT / "eval_datasets" / "profiles" / "voice_clone_asset" / "README.md",
            ROOT / "eval_datasets" / "profiles" / "voice_clone_asset" / "schema.md",
            ROOT / "eval_datasets" / "review_window" / "README.md",
            ROOT / "eval_datasets" / "examples" / "voice_clone_asset_minimal" / "README.md",
        ]

        combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        self.assertIn("Do not commit real audio", combined)
        self.assertIn("private", combined)
        self.assertIn("human", combined.lower())

    def _walk_json_values(self, value: Any) -> list[Any]:
        if isinstance(value, dict):
            values: list[Any] = []
            for child in value.values():
                values.extend(self._walk_json_values(child))
            return values
        if isinstance(value, list):
            values = []
            for child in value:
                values.extend(self._walk_json_values(child))
            return values
        return [value]

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records


if __name__ == "__main__":
    unittest.main()
