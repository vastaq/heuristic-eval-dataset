# voice_clone_asset

`voice_clone_asset` is an experimental profile for cloned character voice
assets.

It is designed for workflows where a voice is cloned from curated audio
material, then evaluated and improved over multiple versions. The profile does
not try to produce one universal audio score. It turns voice clone failures into
traceable evidence and routing decisions:

- source material issue;
- preprocessing issue;
- clone version regression;
- speaker identity drift;
- character style drift;
- speed or pause regression;
- human-review uncertainty;
- acceptable variance.

The goal is to stop fixing cloned voices at the wrong layer.

For a public, data-free shape example, see
`eval_datasets/examples/voice_clone_asset_minimal/`. For the lightweight
human-review task/result protocol, see `eval_datasets/review_window/`.

## Profile Goal

Manage, evaluate, regress, and improve a cloned voice asset without turning
every weak output into random source-pack changes.

A failed clone eval should answer:

- what failed;
- which layer owns the failure;
- which actions are blocked;
- which evidence supports the decision;
- what the next source, review, regression, or calibration action should be.

## Core Objects

- `raw_bank`: original source clips. Append-only; do not overwrite.
- `candidate_bank`: clips that pass basic automatic quality checks but are not
  confirmed anchors.
- `anchor_bank`: human-reviewed clips that represent the desired voice asset.
- `negative_bank`: clips that should not be learned from, such as noise,
  wrong style, wrong speed, multi-speaker pollution, or role mismatch.
- `regression_bank`: fixed texts and scenarios that every clone version should
  be tested against.
- `source_pack`: selected clips used to build one clone version.
- `clone_version`: one generated voice model or voice asset version.
- `review_task`: a human listening task for source clips, clone A/B comparison,
  or output issue diagnosis.
- `route_decision`: the compressed decision after evidence review.
- `voice_output_run`: one batch of generated or imported voice outputs being
  evaluated.
- `judge_observation`: one diagnostic signal from local QC, VAD, no-reference
  quality, audio LLM labels, pairwise preference, or another capability.
- `human_checkpoint`: a targeted human review step created by the agent and
  returned to the same run.

## Required Fields

Use these fields when the profile creates dataset candidate records. Do not
reuse conversation-role fields unless they truly belong to the voice workflow.

- `id`: stable candidate record id.
- `layer`: intake, experiment, regression, project extension, core, or gate.
- `character_id`: stable voice asset owner or role id.
- `voice_model_version`: clone version under evaluation, when applicable.
- `source_pack_id`: source pack used to create the voice version, when known.
- `test_item_id`: stable regression or review item id.
- `text`: text spoken by the clone output, when applicable.
- `audio_uri`: local or external URI for the audio artifact.
- `asset_stage`: source_clip, source_pack, clone_output, regression_output, or
  review_task.
- `expected_behavior`: judgeable behaviors or quality expectations.
- `failure_behavior`: failure behaviors that should lower score or block use.
- `quality_signals`: metric, review, route, or diagnostic refs.
- `rubric_ref`: profile-local rubric key.
- `source_path`: source manifest, report, or run path.
- `review_status`: candidate, needs_revision, accepted, or retired.

## Run Architecture

Use `voice_output_run` as the main intake for clone outputs and adjacent voice
generation workflows. A run may start from a clone/TTS API, existing audio
files, an end-to-end audio model, or A/B candidate outputs. Normalize those
sources into output records before judging them.

Recommended run files:

- `manifest.json`: run identity, profile, source type, input mode, and file
  refs.
- `outputs.jsonl`: normalized audio outputs with optional text, prompt, model,
  and voice refs.
- `observations.jsonl`: diagnostic judge signals from local capabilities.
- `review_tasks.jsonl`: agent-created human checkpoint tasks.
- `review_results.jsonl`: human checkpoint results.
- `route_decisions.jsonl`: compressed routing decisions after evidence review.
- `summary.json` and `summary.md`: global state for the agent and human reader.

Judge observations are evidence, not truth. Store capability output with
`observed_issue`, `suspected_layer`, `confirmed_layer`, `confidence`, and
`blocked_uses` so later agents know what the signal can and cannot prove.

Human review is part of the agent loop. The agent creates targeted tasks,
waits for the checkpoint result, then resumes the run and writes route
decisions. Do not treat the review window as a separate annotation project.

## List Fields

These fields should be lists when present:

- `expected_behavior`
- `failure_behavior`
- `quality_signals`
- `tags`
- `evidence_refs`
- `blocked_actions`
- `next_actions`

## Rubric Vocabulary

Recommended dimensions:

- `source_quality`: source clip has usable audio quality, no clipping, limited
  noise, and stable loudness.
- `source_identity_fit`: source clip belongs to the intended speaker or voice
  asset.
- `source_style_fit`: source clip matches the desired character style, pace,
  emotion, and delivery.
- `transcript_fit`: transcript and spoken content match enough for source-pack
  use.
- `output_quality`: generated output is clear and free of obvious artifacts.
- `intelligibility`: generated output is understandable.
- `speaker_identity`: generated output remains close to the anchor voice.
- `character_style`: generated output preserves the intended character feeling.
- `speed_and_pause`: generated output stays inside calibrated pacing and pause
  ranges.
- `version_regression`: new clone version is not worse than a previous version
  on regression items.
- `human_review_confidence`: human review is decisive enough to route the
  failure.

## Acceptable Band

A voice clone asset is inside the acceptable band when:

- technical output quality is high enough for the target use;
- intelligibility is stable;
- speaker identity is close enough to reviewed anchors;
- character style is not drifting in important scenarios;
- speed, pause, and prosody are inside per-character calibrated ranges;
- remaining weak outputs are low severity, known variance, or review-only
  uncertainties;
- optimization pressure is not causing random source-pack churn.

Do not use one universal threshold for every character, language, clone vendor,
or speaking style. Calibrate from anchor and negative banks.

## Promotion Evidence

Promote a candidate source clip, regression item, route decision, or clone
version only with evidence appropriate to the asset:

- source auto-QC report;
- human source review;
- clone version card;
- regression output metrics;
- human A/B review or output issue review;
- route decision with blocked actions and next actions;
- calibration note explaining thresholds.

Dry-run or metric-only evidence should not promote a voice version or anchor.

## Bloat Guardrail

Before changing a source pack, anchor bank, negative bank, or clone version
because one output sounds wrong, ask:

1. Is this repeated across texts, source packs, or versions?
2. Did technical quality pass before judging character similarity?
3. Is the failure owned by source material, preprocessing, clone version,
   character definition, or reviewer uncertainty?
4. Would adding random source audio make the average voice less stable?
5. Should the case become a regression item instead of a source-pack change?

If the answer is unclear, keep the issue as experiment evidence and request
human review.
