# sample_character_voice_run_001

This synthetic run shows the first architecture layer for `voice_clone_asset`.
It is not a Web review implementation and it does not call a real generation
provider.

## Flow

```text
manifest.json
-> outputs.jsonl
-> observations.jsonl
-> review_tasks.jsonl
-> review_results.jsonl
-> route_decisions.jsonl
-> summary.json
```

## Current State

- `sample_character_out_001`: automatic signals are inside the expected band.
- `sample_character_out_002`: light QC and optional heavy judge signals suggest
  speed and pause regression; human review confirmed the issue.
- `sample_character_out_003`: optional style signal is uncertain and still
  awaits human review.

## Mainline Rule

The agent owns the run. Human review is a checkpoint: the agent creates targeted
tasks, waits for `review_results.jsonl`, then resumes by writing route decisions
and keeping feedback usefulness as `unknown` until a later run validates it.
