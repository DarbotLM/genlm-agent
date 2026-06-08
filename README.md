# GenLM-Agent

GenLM-Agent is a verification-first coding-agent toolkit.

This repository currently ships:

- A structured `VerifiedAgent` execution loop with intent, action-review, evidence, and verdict gates.
- A repository-aware `TaskEngineAgent` for small structured coding models that operate as task processors.
- Generative Layer Management with a finite semantic decision tree and adaptive flashcard memory.
- Typed evidence extraction with freshness and evidence-strength checks.
- A CLI entrypoint for verifying observations from the shell.
- A regression test suite covering the core agent and verifier behavior.

## Quick start

Inspect evidence from an observation:

```bash
python -m glmagent.run.run verify-observation --text "repo tests passed at 2026-03-09T12:00:00Z" --scope repo
```

Use the Python API:

```python
from glmagent.agent import VerifiedAgent
from glmagent.verification import EvidenceLevel
from glmagent.task_engine import TaskEngineAgent, TaskSpec
```

## Documentation

Project docs live under [`docs/`](docs).

## License

MIT. See [`LICENSE`](LICENSE).
