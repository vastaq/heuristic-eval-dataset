# heuristic-eval

`heuristic-eval` is an agent-native framework for turning evaluation feedback
into maintainable evidence, state, decisions, and eval assets.

It is for projects where prompts, roles, rubrics, or agent behaviors are being
iterated from eval results, but where blindly adding more prompt rules would
make the system stiff, overfit, or hard to maintain.

The repository is not only a skill package:

- `skills/heuristic-eval/SKILL.md` is the short agent entrypoint.
- `eval_datasets/` is the file-backed workspace for profiles, adapters,
  templates, scripts, run intake, methodology, and local data placeholders.
- `README.md` is the public project orientation.

No project datasets are included. Canonical data, evaluator exports, run
outputs, replay outputs, event logs, and private prompts stay local by default.

## What It Does

Use this framework to:

1. Feed old tests, YAML files, eval results, and summaries into traceable
   evidence.
2. Maintain reviewed eval records without treating broad imports as gates.
3. Generate small candidate testsets from repeated failures or coverage gaps.
4. Absorb returned evaluator results, such as Promptfoo output, without
   immediately patching the prompt.
5. Capture short human judgments as heuristic memory.
6. Decide when to revise, retire, accept variance, create a failure pattern, or
   stop tuning.

The core idea is simple: a failed eval is evidence, not an automatic prompt
constraint.

## Name

The name is intentionally broader than datasets:

- **heuristic**: agents use feedback to update rubrics, replay plans, failure
  patterns, datasets, and decisions.
- **eval**: the framework keeps prompts, roles, and agent behaviors inside an
  acceptable experience band.

Datasets are one maintained asset inside the framework, alongside run intake,
human signals, learning state, routes, reward assessments, and decisions.

The project is inspired by
[Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/#zh)
and the accompanying
[Trinkle23897/learning-beyond-gradients](https://github.com/Trinkle23897/learning-beyond-gradients)
repository. The useful lesson here is not to optimize every local score; it is
to keep a heuristic system that observes feedback, mutates small assets, replays
evidence, and compresses the result into smaller maintainable state.

## Scope

The framework is evaluator-agnostic. Promptfoo is the bundled default adapter
and example path because the first concrete use case is conversation role eval.

Do not force every task into the bundled conversation-role schema. For TTS,
coding, retrieval, tool use, content generation, or another domain, create a
profile and adapter that match the local evidence shape.

The repository includes an experimental `voice_clone_asset` profile for cloned
voice asset governance. It focuses on source banks, anchor and negative
examples, clone-version regression, human A/B review, and routing failures to
the right layer. It does not include real audio or claim fully automatic voice
quality judgment.

Optional local tools should be exposed through capability registries, not hard
coded into public docs. The public template
`eval_datasets/templates/capabilities.local.example.json` shows how a downstream
workspace can tell an agent that local VAD, no-reference quality, audio-LLM
judge, pairwise preference, or human-review capabilities are available without
committing private model paths or audio assets. Copy it to a gitignored local
path such as `.heuristic-eval.local/capabilities.json` and replace placeholders
with project-local runners.

Use three layers:

- **Framework**: shared concepts such as run intake, learning state, reward
  interpretation, prompt-bloat gates, stop rules, and legacy bootstrap.
- **Profile**: domain-specific schema, rubric vocabulary, acceptable band, and
  promotion evidence.
- **Adapter**: evaluator-specific import, export, and result normalization.

The Python scripts are working primitives and examples, not the universal
encoding of the framework. Agents should compose, adapt, or add profile/adapter
tools when a project has a different runner or evidence shape.

## Agent-Native Usage

This framework is meant to be handed to an agent, not manually installed like a
traditional CLI package.

Give the agent this repository URL and ask it to use `skills/heuristic-eval/SKILL.md`
as the operating entrypoint for eval feedback work:

```text
Use https://github.com/vastaq/heuristic-eval as the heuristic-eval framework for
this project. Read skills/heuristic-eval/SKILL.md, inspect the target eval
artifacts, and create run intake, observations, decisions, or candidate eval
assets as needed.
```

The agent should locate, clone, vendor, or reference the repository in whatever
way fits the current workspace:

```bash
git clone https://github.com/vastaq/heuristic-eval.git
cd heuristic-eval
npm install
npm test
```

For agent runtimes that support skills, the agent can symlink the skill folder
so the entrypoint stays synced with the repository:

```bash
mkdir -p .agents/skills
ln -s "$PWD/skills/heuristic-eval" .agents/skills/heuristic-eval
```

If the runtime uses a different skill root, the agent should keep the same
folder shape:

```text
<tool-skill-root>/heuristic-eval/SKILL.md
```

The skill does not install data. It tells the agent how to use this repository
as the local eval-maintenance framework. Human setup can stay minimal: point the
agent at the repo and the target project's eval artifacts.

## Core Workflows

### 1. Feed Old Tests And References

Use this when a project has legacy Promptfoo YAML, old canonical JSON, test
folders, eval summaries, or human notes.

Rules:

- Preserve raw assets or references to them.
- Convert old tests into `candidate` records, not `accepted` records.
- Keep source traceability such as `source_path`, `source_index`, evaluator
  variables, and legacy assertions when the active adapter uses them.
- Treat summaries as candidate failure patterns, state gaps, stop rules, or
  review notes, not final truth.

Common commands:

```bash
python3 eval_datasets/scripts/import_legacy_learning.py \
  path/to/legacy/tests path/to/legacy/results \
  --output-dir eval_datasets/experiments/legacy_bootstrap \
  --project example_project
```

```bash
python3 eval_datasets/scripts/import_promptfoo_tests.py \
  path/to/test.yaml \
  eval_datasets/seeds/example.from_legacy.json \
  --project example_project \
  --role example_role
```

```bash
python3 eval_datasets/scripts/normalize_legacy_canonical.py \
  path/to/character_eval_dataset.v1.json \
  eval_datasets/canonical/character_eval_dataset.candidate.json \
  --project example_project
```

### 2. Maintain Existing Datasets

Use this when editing canonical records, changing review status, retiring noisy
cases, or regenerating evaluator views.

Rules:

- Edit canonical JSON, not generated exports.
- Keep record IDs stable unless the sample becomes a different test.
- Promote to `accepted` only with review evidence.
- Use `retired` instead of deleting obsolete tests.
- Stop prompt tuning when behavior is inside the acceptable band.

Common commands:

```bash
python3 eval_datasets/scripts/audit_testset_balance.py \
  eval_datasets/canonical/example.v1.json
```

```bash
python3 eval_datasets/scripts/export_promptfoo_tests.py \
  eval_datasets/canonical/example.v1.json \
  eval_datasets/exports/example.yaml \
  --include-status accepted
```

### 3. Generate New Candidate Testsets

Use this when repeated evaluator failures, human review, or coverage gaps should
become new tests.

Rules:

- Do not create isolated one-off tests when the signal is weak.
- Diagnose whether the failure is generic, project-specific, role-specific, or
  noisy.
- Prefer a small contrast set over a single case.
- Keep generation in `eval_datasets/experiments/` until evidence justifies
  canonical changes.
- Validate candidate units before recording a successful learning outcome.

Common command:

```bash
python3 eval_datasets/scripts/validate_hl_dataset_candidate_units.py \
  eval_datasets/experiments/example/dataset_candidate_units
```

### 4. Absorb Returned Evaluations

Use this after running Promptfoo or another evaluator. The goal is to classify
what the result means before changing prompts or datasets.

Rules:

- Store full run intake under `eval_datasets/runs/<adapter>/<run_id>/`.
- Normalize returned results into observations when possible.
- Route observations before prompt mutation, canonical promotion, or dataset
  generation.
- Reject dry-run or unjudged reward files as formal promotion evidence.
- Treat pass rate as one signal among naturalness, role consistency, regression
  risk, judge noise, user goal, and prompt-bloat risk.

Promptfoo example:

```bash
python3 eval_datasets/scripts/init_eval_run.py \
  --run-id example_run_001 \
  --project example_project \
  --profile conversation_role \
  --adapter promptfoo \
  --profile-ref eval_datasets/profiles/conversation_role/README.md \
  --adapter-ref eval_datasets/adapters/promptfoo/README.md \
  --source-root path/to/project/local/output \
  --source-file summary=path/to/project/local/output/summary.json
```

```bash
python3 eval_datasets/scripts/normalize_promptfoo_results.py \
  path/to/promptfoo_result.json \
  --output eval_datasets/runs/promptfoo/example_run_001/observations.json \
  --project example_project \
  --run-id example_run_001
```

```bash
python3 eval_datasets/scripts/route_hl_observations.py \
  eval_datasets/runs/promptfoo/example_run_001/observations.json \
  eval_datasets/runs/promptfoo/example_run_001/route.json
```

### 5. Optimize Without Generating A Dataset

Some projects already have generated outputs, review summaries, diagnostics,
and human judgment. In that case, use this framework as an optimization ledger
without immediately creating canonical records.

Use a run manifest, human signals, decision file, and optional observations to
record why a prompt, variable boundary, rubric, or stop decision is reasonable.

See
`eval_datasets/methodology/eval_optimization_without_dataset_generation.md`.

## When To Stop

This project deliberately supports stopping. Do not continue prompt tuning just
to pass one or two narrow tests when the system is already inside the acceptable
band.

Prefer these outcomes when they fit the evidence:

- `accept_variance`
- `mark_noisy_eval`
- `mark_case_failure`
- `mark_rubric_failure`
- `add_or_update_failure_pattern`
- `create_targeted_replay`
- `create_dataset_candidate_unit`
- `stop_tuning`

Prompt or policy mutation should come late, after the bloat gate and only when
the failure is repeated, user-visible, severe, and better solved by the prompt
than by the dataset, rubric, or case.

## Repository Layout

```text
skills/heuristic-eval/
  SKILL.md                 # Short agent entrypoint

eval_datasets/
  methodology/             # Framework guidance and profile references
  profiles/                # Domain-specific schema and rubric notes
  adapters/                # Evaluator-specific adapter notes
  review_window/           # Lightweight human-review task/result protocol
  templates/               # Copyable run, decision, and signal templates
  examples/                # Public examples without private data
  scripts/                 # Working primitives and adapter examples
  tests/                   # Script-level regression tests
  canonical/               # Local canonical datasets; ignored by default
  exports/                 # Generated evaluator views; ignored by default
  seeds/                   # Source snapshots; ignored by default
  experiments/             # Candidate units and assessments; ignored by default
  runs/                    # Full evaluator run intake; ignored by default
  replay/                  # Optional lightweight replay configs and outputs
  evolution/               # Event logs and failure patterns; ignored by default
```

For doc routing, start with `eval_datasets/methodology/README.md`.
For script support levels, read `eval_datasets/scripts/README.md`.

## Data Policy

Data stays local by default. The `.gitignore` keeps these paths empty except for
placeholders:

- `eval_datasets/canonical/`
- `eval_datasets/exports/`
- `eval_datasets/runs/`
- `eval_datasets/seeds/`
- `eval_datasets/experiments/`
- `eval_datasets/replay/outputs/`

Local capability registries also stay private by default:

- `.heuristic-eval.local/capabilities.json`
- `eval_datasets/**/*.local.*`
- `eval_datasets/evolution/events.jsonl`
- `eval_datasets/evolution/failure_patterns/`

Store private or large project data locally, in the downstream project, or in a
private data store. Use Hugging Face or another shared dataset host only after
the data is scrubbed, licensed, intentionally public, and separated from private
prompt or product evidence.

## License

MIT. See `LICENSE`.

The license covers this framework repository, not downstream private eval data,
prompts, or project artifacts.

## Quick Checks

```bash
npm test
npm run py-compile
git diff --check
```
