# GenLM-Agent

GenLM-Agent is a verification-first coding-agent toolkit.

The framework is built around:

- `VerifiedAgent` for guarded execution loops.
- `TaskEngineAgent` for repository-aware narrow task execution with processor-style models.
- `SemanticDecisionTree` and `AdaptiveFlashcardDeck` for layered control and compact verified memory.
- Typed evidence extraction for command and verification output.
- A lightweight CLI for observation verification.

## Modules

- `glmagent.agent`: runtime loop and execution model.
- `glmagent.task_engine`: task-engine schema and repository-aware constrained agent.
- `glmagent.verification`: evidence types, freshness checks, and claim grounding helpers.

## Next steps

Start with the usage guide, then review the API reference for the core classes.
