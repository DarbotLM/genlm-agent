# GenLM-Agent

[![Pytest](https://github.com/DarbotLM/genlm-agent/actions/workflows/pytest.yaml/badge.svg)](https://github.com/DarbotLM/genlm-agent/actions/workflows/pytest.yaml)
[![codecov](https://codecov.io/gh/DarbotLM/genlm-agent/branch/main/graph/badge.svg)](https://codecov.io/gh/DarbotLM/genlm-agent)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://pypi.org/project/glmagent/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

GenLM-Agent is a **verification-first** coding-agent toolkit. Every action passes through structured gates before the agent proceeds — no silent failures, no ungrounded claims.

## What's included

| Component | Description |
|-----------|-------------|
| `VerifiedAgent` | Guarded execution loop with four mandatory verification gates |
| `TaskEngineAgent` | Repository-aware agent for small structured coding models |
| `SemanticDecisionTree` | Finite-state layer controller (`triage → localize → edit → verify → complete`) |
| `AdaptiveFlashcardDeck` | Compact verified memory that replaces full trajectory replay |
| Evidence framework | Typed evidence with freshness, strength, and scope checks |
| CLI | Shell entrypoint for observation verification |

## Installation

```bash
pip install glmagent
```

For development (tests, docs, linting):

```bash
git clone https://github.com/DarbotLM/genlm-agent.git
cd genlm-agent
pip install -e ".[dev]"
pre-commit install
```

## Architecture

### Verification gates

Every `VerifiedAgent` step passes through four gates:

```
Intent Gate → Action Gate → Observation Gate → Verdict Gate
```

| Gate | When | Purpose |
|------|------|---------|
| **Intent** | Before acting | Declare what success looks like |
| **Action** | Before executing | Assess risk and prerequisites |
| **Observation** | After executing | Verify evidence freshness and depth |
| **Verdict** | After verifying | Confirm all claims are grounded |

### Generative Layer Management

`TaskEngineAgent` routes work through a finite semantic decision tree:

```
triage → localize → edit → verify → complete
```

Each layer exposes only the task kinds appropriate for that phase. Verified facts and failure patterns are compressed into adaptive flashcards — no full trajectory replay needed.

## Quick start

### CLI

Verify an observation from the shell:

```bash
glmagent verify-observation --text "repo tests passed at 2026-03-09T12:00:00Z" --scope repo
```

From a file:

```bash
glmagent verify-observation --file output.txt --scope repo
```

### Python API — evidence extraction

```python
from glmagent.verification import extract_evidence, EvidenceLevel

evidence = extract_evidence(
    source="ci",
    text="All 42 tests passed in 3.2s",
    scope_items=["tests/"],
)

for e in evidence:
    print(e.level, e.content)
# EvidenceLevel.FUNCTIONAL  All 42 tests passed in 3.2s
```

### Python API — VerifiedAgent

```python
from glmagent.agent import VerifiedAgent, AgentConfig

config = AgentConfig(max_steps=10, require_verdict=True)
agent = VerifiedAgent(model=my_model, environment=my_env, config=config)
result = agent.run("Fix the failing test in tests/test_foo.py")
```

### Python API — TaskEngineAgent

```python
from glmagent.task_engine import (
    TaskEngineAgent, TaskEngineConfig,
    TaskSpec, TaskKind, RepositorySnapshot,
)

config = TaskEngineConfig(max_steps=15)
agent = TaskEngineAgent(model=my_model, environment=my_env, config=config)

snapshot = RepositorySnapshot(root="/repo", branch="fix/issue-42")
task = TaskSpec(
    kind=TaskKind.TEST,
    title="Run failing test suite",
    rationale="Confirm baseline before editing",
    command="pytest tests/ -x",
    scope_items=["tests/"],
    evidence_required="functional",
    completion_claim="Test suite passes",
)
result = agent.step(task, snapshot)
```

## CLI reference

```
glmagent [--version]
glmagent verify-observation --text TEXT [--scope SCOPE]
glmagent verify-observation --file PATH  [--scope SCOPE]
```

## Running tests

```bash
pytest tests/
# parallel:
pytest tests/ -n auto
```

## Documentation

Full documentation lives under [`docs/`](docs) and is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

```bash
mkdocs serve   # live preview at http://127.0.0.1:8000
mkdocs build   # static output in site/
```

## Contributing

1. Fork the repository and create a feature branch.
2. Install dev dependencies: `pip install -e ".[dev]"`
3. Install pre-commit hooks: `pre-commit install`
4. Add or update tests for your change.
5. Open a pull request against `main`.

Please follow the [Code of Conduct](.github/CODE_OF_CONDUCT.md).

## License

MIT. See [`LICENSE`](LICENSE).
