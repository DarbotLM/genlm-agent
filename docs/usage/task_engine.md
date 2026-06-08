# Task engine

GenLM-Agent includes a narrow task-engine mode for smaller generative SWE models that act as task processors rather than open-ended chat agents.

## Design goals

- Use a fixed task vocabulary.
- Expose repository state as a structured snapshot.
- Route tasks through a finite semantic decision tree instead of open-ended chain-of-thought branching.
- Require explicit file, command, and verification scope targets.
- Keep each model step narrow enough for smaller transformer models to execute reliably.
- Compress verified facts and anti-patterns into adaptive flashcards instead of replaying the full history.

## Core schema

The task engine expects a top-level `task` object with fields such as:

- `kind`
- `title`
- `rationale`
- `files`
- `command`
- `patch`
- `scope_items`
- `evidence_required`
- `recency_window_seconds`
- `completion_claim`

Use `TaskEngineAgent` when you want a coding agent to behave like a deterministic task router instead of a free-form shell planner.

## Why this shape

Recent SWE-bench agents have converged on a few patterns:

- Constrained interfaces and narrow action vocabularies outperform unconstrained shell chatter.
- Strong localization and verification loops matter more than longer free-form reasoning traces.
- Lightweight state compression beats replaying the entire trajectory to the model every step.

GenLM-Agent now reflects that with Generative Layer Management:

- `triage -> localize -> edit -> verify -> complete`
- layer-specific allowed task kinds
- adaptive flashcards that retain verified facts and common failure modes
