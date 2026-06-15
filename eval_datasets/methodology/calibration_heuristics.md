# Calibration Heuristics

Calibration heuristics are small, lifecycle-bound claims about how machine
signals relate to human review. They are not run summaries, hard rules, or a
place to store every observation.

Use this layer when repeated work suggests that a metric is useful, misleading,
or only safe for routing. New notes should stay weak and reversible until
multiple runs support them.

Calibration is shared across profiles, but interpretation is not. The framework
owns the lifecycle: claim strength, scope, evidence counts, promotion, decay,
and retirement. The profile owns signal interpretation: what a pass rate means
for prompt work, what a no-reference audio metric means for voice work, and
which decisions those signals are allowed to influence.

## Layers

```text
run observation
-> weak hypothesis
-> candidate heuristic
-> active heuristic
-> retired or archived
```

- `run observation`: what happened in one run. Keep this in the run folder.
- `weak hypothesis`: a possible interpretation with limited scope.
- `candidate heuristic`: a repeated hypothesis that may guide routing.
- `active heuristic`: a small, reviewed rule with support and known limits.
- `retired` or `archived`: contradicted, stale, or no longer useful.

## Required Fields

Each JSONL record in `eval_datasets/evolution/calibration_heuristics.jsonl`
should include:

- `schema_version`: use `heuristic_eval.calibration_heuristic.v0`.
- `id`: stable heuristic id.
- `claim`: short natural-language claim.
- `claim_strength`: `weak`, `medium`, or `strong`.
- `status`: `run_note`, `candidate`, `active`, `retired`, or `archived`.
- `scope`: profile, task type, metric family, and decision target.
- `evidence.support_count`: supporting run or review count.
- `evidence.contradiction_count`: known counterexamples.
- `lifecycle.promotion_threshold`: support needed before promotion.
- `lifecycle.retire_if_contradictions`: contradiction limit before downgrade.
- `use_policy`: how an agent may use the heuristic.
- `blocked_uses`: actions this heuristic must not justify.

## Use Policy

Weak claims should not drive automatic decisions. They may only:

- route a case to human review;
- warn that a metric is low-confidence for a given decision target;
- suggest what to sample next;
- prevent overconfident promotion from one metric.

Medium or strong claims still need scope checks. A heuristic about technical
audio quality should not become a speaker identity rule unless it has direct
identity evidence.

## Prompt And Audio Profiles

Prompt eval and audio eval can use the same lifecycle without sharing the same
fields.

For prompt or conversation-role work:

```text
Promptfoo result / judge reason / pass rate
-> observation
-> weak calibration hypothesis
-> candidate heuristic
-> route decision: accept variance, revise rubric, revise case, or run a
   guarded prompt experiment
```

Prompt-side examples should stay close to the existing project references:

- `eval_datasets/profiles/conversation_role/README.md`
- `eval_datasets/adapters/promptfoo/README.md`
- `eval_datasets/evolution/failure_patterns/prompt_patch_pressure.json`
- `eval_datasets/evolution/failure_patterns/eval_reward_shape_bias.json`

Do not invent a new prompt-eval scenario just to demonstrate calibration. The
current concrete path is conversation-role evaluation, usually through
Promptfoo-normalized observations. A prompt-side calibration claim should
therefore be phrased as a caution around existing signals such as pass rate,
judge reason, role naturalness, prompt-bloat pressure, or reward-shape bias.

Example weak claim, grounded in the existing failure-pattern references:

```text
Pass-rate improvement is not enough to accept a conversation-role prompt change
when it increases case-by-case prompt patching or reward-shape bias.
```

For audio or voice-clone work:

```text
Local QC / no-reference quality / audio LLM labels / speaker metrics
-> observation
-> weak calibration hypothesis
-> candidate heuristic
-> route decision: accept output, reject candidate, request human A/B review,
   or generate a follow-up voice variant
```

Example weak claim:

```text
A technical no-reference quality proxy is not enough to decide character voice
identity by itself.
```

These examples are intentionally weak. They may route future cases to human
review or sampling, but they must not become automatic promotion rules until
their lifecycle evidence supports that promotion.

## Composite Project Runs

Some products need both prompt and audio evaluation. Do not merge every field
into one large schema. Use a `project_run` summary that references profile-owned
sub-runs:

```text
project_run
  conversation_role run
  voice_clone_asset run
  combined route decision
```

The combined decision can ask whether text behavior and voice behavior are both
inside the acceptable band, but each profile should keep its own schema,
observations, review tasks, and calibration interpretation.

## Compaction

Agents should periodically compact this file:

- merge duplicate claims;
- archive weak claims that were not reused;
- split broad claims when contradictions appear;
- promote only when evidence meets the lifecycle threshold;
- keep private run paths, private audio, provider ids, prompts, and people out
  of the public framework repository.

The public example is synthetic. Downstream projects can keep private
calibration records in gitignored local paths and promote only scrubbed,
general-purpose heuristics upstream.

## File Split Rule

Keep this document only for cross-profile calibration lifecycle. Put
profile-specific judging details in profile docs, and evaluator file-shape
details in adapter docs.

Create a new methodology file only when the idea changes the shared workflow
across profiles. If the idea only explains Promptfoo fields, put it in the
Promptfoo adapter. If it only explains conversation-role behavior, put it in the
conversation-role profile or the existing conversation methodology documents.
