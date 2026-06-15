# voice_clone_asset Minimal Example

This example shows the public shape of a voice clone asset loop without shipping
real audio, real transcripts, speaker identities, or private project data.

All audio paths are placeholders using `local://...`. They are meant to point to
project-local files in a downstream workspace.

## Flow

```text
source_clip_report.sample.json
-> human_source_review.sample.json
-> clone_version_card.sample.json
-> regression_result.sample.json
-> human_ab_review.sample.json
-> route_decision.expected.json
```

The route decision shows the key behavior: a technically clean output can still
route to `speed_regression`, while blocking random source-pack changes.

`run_example/` shows the first agent-owned run architecture:

```text
voice output intake
-> judge observations
-> targeted human checkpoint tasks
-> human checkpoint results
-> route decisions
-> run summary
```

The example keeps generation broad. The outputs could come from a clone/TTS API,
an existing audio folder, an end-to-end audio model, or A/B candidate files. The
profile evaluates normalized voice outputs rather than assuming every run starts
with a provider `voice_id`.

## Capability Registry

`capabilities.sample.json` demonstrates how a downstream project can expose
local audio evaluators to an agent without publishing private model paths,
speaker identities, or audio files.

The important design choice is that agents select by capability:

```text
audio_qc.vad
audio_quality.no_reference
audio_judge.speech_attributes
audio_judge.naturalness_preference
human_review.voice_clone_spot_check
```

They should not select by private model name. In a real workspace, copy the
template from `eval_datasets/templates/capabilities.local.example.json` to a
gitignored file such as `.heuristic-eval.local/capabilities.json`, then replace
runner placeholders with local scripts or services.

Do not commit real audio, real model paths, API keys, signed URLs, speaker
identities, or private prompts.
