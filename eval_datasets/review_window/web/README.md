# Local Review Window

`index.html` is a no-build local review surface for the `review_window`
checkpoint protocol.

It is intentionally file-backed:

- Load `review_tasks.jsonl`.
- Optionally load `observations.jsonl` and `summary.json`.
- Record blind-first human decisions.
- Reveal machine signals only after first impression.
- Keep in-progress drafts and saved rows in browser-local cache for the current
  task URL, so accidental refreshes do not wipe the review.
- Export `review_results.jsonl` so the agent can resume the run.

For an agent-owned checkpoint, serve the workspace locally and pass same-origin
run files in the URL:

```text
index.html?tasks=/path/to/review_tasks.jsonl&observations=/path/to/observations.jsonl&summary=/path/to/summary.json
```

Add `lang=zh` or `lang=en` for a Chinese or English interface:

```text
index.html?lang=zh&tasks=/path/to/review_tasks.jsonl&observations=/path/to/observations.jsonl&summary=/path/to/summary.json
```

The file inputs remain a fallback for ad hoc review. The URL handoff is the
preferred path when an agent asks a human to review one run and then return
`review_results.jsonl` to the main loop.

Tags are grouped in both review modes:

- Single-output tasks group tags by meaning, such as positive evidence, issues,
  and other notes. A task may provide `tag_groups`; otherwise the window uses a
  conservative built-in grouping for common voice-clone tags.
- A/B tasks group tags by target (`A`, `B`, and shared/overall), so a positive
  or negative label is not ambiguous. The exported result preserves this as
  `ab_dimension_tags` and also includes flattened `dimension_tags` for simple
  downstream readers.

It is not a hosted service, database, annotation platform, or audio asset
manager. Real audio paths, signed URLs, model outputs, and private prompts
should stay in the downstream workspace.
