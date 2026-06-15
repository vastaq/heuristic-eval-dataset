# Framework, Profiles, And Adapters

This repository should not force every project to use the bundled
conversation-role schema or Promptfoo file shape. The reusable part is the
heuristic maintenance framework. The encoding belongs to profiles and adapters.

## Three Layers

### Framework

The framework is shared across eval domains:

- Raw / Intake / Core / Experiment / Gate layers.
- Autonomous controller.
- Learning state.
- Run intake.
- Reward and compression decisions.
- Calibration heuristic lifecycle.
- Prompt-bloat or policy-bloat gates.
- Stop rules.
- Legacy learning bootstrap.

Framework documents should avoid project-specific field assumptions whenever
possible.
For calibration work, the framework owns the lifecycle: claim strength, scope,
support and contradiction counts, promotion, decay, and retirement. The profile
owns signal interpretation.

### Profile

A profile defines the meaning of data for one eval domain.

Examples:

- `conversation_role`
- `coding_eval`
- `support_agent`
- `retrieval_eval`
- `tool_use_eval`

A profile owns:

- canonical record schema
- required and optional fields
- taxonomy and coverage dimensions
- rubric vocabulary
- acceptable-band definition
- promotion evidence
- bloat guardrail
- signal interpretation for profile-specific metrics and reviews

Profiles may reuse framework terms, but they should not force other profiles to
inherit their fields. For example, `role`, `character_context`, and `scene_type`
belong to the current conversation-role profile. They are not universal fields.
Shared failure-pattern files should follow the same rule: use `role` when the
owner really is a role, or use `profile` / `domain` for non-role systems such as
tool-use, retrieval, or content generation evals.
Shared calibration heuristics follow the same rule: the lifecycle is shared, but
prompt pass-rate interpretation belongs to prompt/conversation profiles, and
audio quality or speaker-identity interpretation belongs to audio profiles.

### Adapter

An adapter maps an evaluator or external file format into and out of a profile.

Examples:

- Promptfoo YAML import/export for conversation-role tests.
- A coding benchmark result importer.
- A support-ticket review exporter.
- A custom judge result normalizer.

Adapters own:

- parser assumptions
- evaluator-specific variables
- assertion or judge-result mapping
- export file shape
- result normalization

For Promptfoo, fields such as `vars`, `assert`, `input_var`, and
`legacy_asserts` are adapter compatibility fields, not framework requirements.

## Agent Rule

When a new project does not fit the bundled conversation-role profile, the agent
should create a new profile and adapter rather than bending the existing schema.

New-domain agent checklist:

1. Inspect the task domain and decide whether the bundled profile/adapter fits.
2. Inspect profile and adapter modules before creating records.
3. Update profile and adapter modules with the smallest useful domain
   boundaries, fields, rubric vocabulary, and result mapping.
4. Init run with profile_ref and adapter_ref in the manifest.
5. Validate with `validate_run_intake.py --require-module-refs`; add
   `--validate-module-notes` and `--validate-adapter-boundaries` when the refs
   must pass calibrated structural readiness.
6. Run `route_hl_observations.py` when normalized observations exist, before
   prompt mutation, canonical promotion, or dataset generation.

Do not modify the shared framework just to satisfy one evaluator's file shape.
Do not stuff non-conversation data into conversation-role fields.

When one product needs multiple profiles, model it as a `project_run` that
references profile-owned sub-runs rather than merging fields. For example, a
character product can have one `conversation_role` run for text behavior and one
`voice_clone_asset` run for voice behavior, then write a combined route decision
that checks whether both are inside their acceptable bands.

## Minimum Module Patch

When a task needs a new eval domain or evaluator shape, the agent should make the
smallest useful module update before importing records or changing prompts.

Create or update these files:

- `eval_datasets/profiles/<profile_id>/README.md`
- `eval_datasets/adapters/<adapter_id>/README.md`

`eval_datasets/scripts/init_profile_adapter.py` is intentionally idempotent:
without `--force`, it keeps existing notes and creates only missing notes. This
lets agents add a new adapter to an existing profile without overwriting local
domain judgment.

The profile note should state:

- domain purpose and domain boundary;
- minimum artifact shape and required fields;
- optional fields and local extension points;
- taxonomy or coverage dimensions;
- quality signals and rubric vocabulary;
- acceptable band, stop rule, and bloat guardrail;
- when canonical records are needed and when run intake is enough.

When validating dataset candidate units for a non-conversation profile, pass
`--profile-ref` first so
`eval_datasets/scripts/validate_hl_dataset_candidate_units.py` reads the
profile README's `## Required Fields` and `## List Fields`. Repeated
`--record-field` / `--list-field` options are additive one-off requirements;
they do not replace the profile-declared fields. Do not create fake `role` or
`scene_type` fields just to satisfy the bundled conversation-role defaults.
Omitting `--profile` means conversation_role only; it is not a new-domain
shortcut. New domains should pass `--profile` with `--profile-ref` instead of
relying on the default.

The adapter note should state:

- evaluator or runner file shape;
- source files the adapter expects;
- import, export, and result-normalization assumptions;
- how structural diagnostics differ from final quality evidence;
- where raw runs, observations, human signals, and decisions should be written;
- what actions are blocked until human review, replay, or audit evidence exists.

Do not change the shared framework for one local evaluator. Add a new profile,
adapter note, or small script in the profile/adapter layer, then keep private run
data in ignored `runs/`, `experiments/`, or project-local paths.

When a run needs structural evidence that a new profile or adapter is ready to
guide work, include `profile_ref` and `adapter_ref` in `manifest.json` and
validate the run with
`eval_datasets/scripts/validate_run_intake.py --require-module-refs`. This keeps
profile/adapter self-updates observable instead of relying on a bare profile or
adapter id string, but it does not prove live autonomous agent behavior or
independent agent choice. The profile and adapter refs should come from the
same `eval_datasets` root; mixing roots can combine unrelated domain and
evaluator assumptions.
When the module update itself becomes durable project knowledge, append a
`profile_adapter_updated` event with `evidence.kind: module_ref` so future
agents can see why the profile or adapter was created.
If the next loop should remember the module update, record a
`profile_adapter_update` learning outcome with `record_learning_outcome.py`.
That state must carry the same profile and adapter scope, concrete
evidence_refs for profile module and adapter module files, and a
validate_run_intake evidence_ref from the passing module-ref validation. It
must also include a profile_adapter_updated event_ref that points to the
append-only event log entry for the module update.

## Bundled Python Scripts

The Python scripts in `eval_datasets/scripts/` are reference implementations for
the bundled setup:

- the current conversation-role profile
- the Promptfoo adapter
- lightweight HL replay and validation utilities
- legacy bootstrap scaffolding

They are useful examples and working tools, but they are not the universal
encoding of this framework. New domains should add new profile/adapter logic
instead of pretending their data is conversation-role Promptfoo data.

Promptfoo scripts are adapter-specific examples, not the framework encoding. Do
not use Promptfoo import/export scripts for non-Promptfoo evaluators; create an
adapter-specific normalizer, importer, or exporter that preserves the local
runner's evidence shape.
