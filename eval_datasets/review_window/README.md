# Review Window Protocol

`review_window` is a lightweight JSON protocol for human review tasks. It is
not a UI framework and not a crowdsourcing platform.

Profiles can use this protocol when automatic metrics are not enough and a
human decision must become structured evidence.

## Scope

The first version should support only:

- `source_clip_review`
- `clone_ab_review`
- `output_issue_review`

The review surface can be a local HTML file, notebook, CLI prompt, or any other
project-local tool. The public framework only defines the task and result shape.

## Agent Checkpoint Flow

Human review is a checkpoint inside an agent-owned eval loop:

```text
agent creates review_tasks.jsonl
-> human completes a local review surface
-> review_results.jsonl is written
-> agent resumes the same run and compresses route decisions
```

Tasks should include `checkpoint_id` and `return_to_run_id` when they are part
of a run. Results should repeat those fields so the agent can match them back to
the pending checkpoint. If results are incomplete, the agent should report the
remaining `task_id` values instead of routing prematurely.

## Blind-First Review

Use blind-first review by default. Show audio, text or prompt, and the direct
question first. Keep machine judge signals collapsed until after the reviewer
records a first impression.

Recommended task fields:

- `review_mode`: `blind_first`
- `auto_signal_visibility`: `collapsed`
- `auto_observations`: observation refs to reveal after first impression
- `route_guess`: optional agent guess, also collapsed
- `allowed_tags`: short dimension tags for structured feedback

Recommended result fields:

- `first_impression`: decision and confidence before seeing auto signals
- `dimension_tags`: affected dimensions selected by the reviewer
- `auto_signals_seen`: whether machine signals were revealed
- `auto_signal_agreement`: agree, partially_agree, disagree, or not_seen
- `final_decision`: the reviewer decision after optional context

## Task Shape

Every task should include:

- `schema_version`
- `checkpoint_id`, when created from a run
- `return_to_run_id`, when created from a run
- `task_id`
- `task_type`
- `profile`
- `target_id`
- `audio_refs`
- `context`
- `choices`
- `evidence_refs`

Audio refs should point to local or private project paths. Do not commit real
audio assets to the public framework repo.

## Result Shape

Every result should include:

- `schema_version`
- `checkpoint_id`, when returning to a run
- `return_to_run_id`, when returning to a run
- `review_id`
- `task_id`
- `task_type`
- `reviewer_role`
- `decision`
- `confidence`
- `tags`
- `notes`

Human review results are evidence for routing. They are not automatic promotion
unless the active profile explicitly says so.
