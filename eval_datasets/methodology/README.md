# Methodology Index

Use this index to choose the smallest reference set for the current task. Do not
load every methodology document by default.

## Start Here

- `framework_profile_adapter.md`: boundary between shared framework,
  domain-specific profiles, and evaluator adapters.
- `autonomous_controller.md`: when and how an agent should run the
  observe/classify/action/replay/reward/compress loop.
- `human_signal_capture.md`: how short user judgments become durable heuristic
  evidence without a heavy labeling workflow.
- `calibration_heuristics.md`: how weak machine-human calibration notes become
  lifecycle-bound heuristics without turning into a bloated rule notebook.

## Common Routes

### Old Content Intake

Read:

- `legacy_learning_import.md`
- `evolution_protocol.md`
- `evolution_event_log.md`

Use when old tests, old eval results, summaries, or prior conclusions need to
enter the system as evidence without becoming accepted truth.

### Conversation Role Eval

Read:

- `dataset.schema.md`
- `conversation_core_taxonomy.md`
- `scenario_taxonomy.md`
- `rubric_templates.md`
- `update_guide.md`

Use when records are conversation-role shaped and may be exported to Promptfoo.

### Returned Eval Results

Read:

- `heuristic_learning_loop.md`
- `hl_reward_assessment.md`
- `dataset_traction_audit.md`
- `evolution_event_log.md`

Use when evaluator output, replay output, judge comments, or human review should
drive a decision.

### Candidate Dataset Generation

Read:

- `hl_dataset_generation.md`
- `experiment_modules.md`
- `hl_replay_executor.md`
- `hl_reward_assessment.md`

Use only after the signal is repeated, judgeable, and not better handled as
acceptable variance, case revision, rubric revision, or noisy eval.

### Optimization Without Dataset Generation

Read:

- `eval_optimization_without_dataset_generation.md`
- `human_signal_capture.md`
- `dataset_traction_audit.md`

Use when a project already has local generated outputs, diagnostics, LLM review,
or human judgment, and the right artifact is a run decision rather than a new
canonical dataset.

### Voice Clone Asset Eval

Read:

- `../profiles/voice_clone_asset/README.md`
- `../profiles/voice_clone_asset/failure_taxonomy.md`
- `../profiles/voice_clone_asset/review_protocol.md`
- `../profiles/voice_clone_asset/calibration.md`

Use when cloned character voices need source-bank governance, anchor/negative
management, human A/B review, clone-version regression, and failure routing.
This route should not ship real audio, speaker identities, transcripts, or
private voice prompts in the public framework repo.

### Machine-Human Calibration

Read:

- `calibration_heuristics.md`
- `evolution_event_log.md`

Use when machine metrics and human review disagree or partially agree, and the
agent needs to decide whether a signal is only a run note, a weak hypothesis, or
a candidate heuristic. Do not promote single-run observations into active rules.

## Advanced System References

- `heuristic_system_spec.md`: state, action space, reward model, replay policy,
  prompt-bloat gate, and compression policy for the system itself.
- `hl_replay_executor.md`: optional lightweight replay wiring.
- `experiment_modules.md`: compact experiments, prompt variants, rubric
  experiments, and promotion/retirement boundaries.

These documents are useful when maintaining the framework. Most project runs
should not need all of them.
