# Hello world

This guide walks through the two fastest ways to use GenLM-Agent: the CLI and the Python API.

## 1. CLI — verify an observation

Extract typed evidence from any observation string:

```bash
glmagent verify-observation \
  --text "repo tests passed at 2026-03-09T12:00:00Z" \
  --scope repo
```

Example output:

```json
[
  {
    "source": "cli",
    "timestamp": "2026-03-09T12:00:00+00:00",
    "data_timestamp": "2026-03-09T12:00:00+00:00",
    "level": "functional",
    "content": "repo tests passed at 2026-03-09T12:00:00Z",
    "scope_item": "repo"
  }
]
```

You can also read from a file:

```bash
glmagent verify-observation --file pytest_output.txt --scope tests/
```

## 2. Python API — evidence extraction

```python
from glmagent.verification import extract_evidence, EvidenceLevel

evidence = extract_evidence(
    source="ci",
    text="42 passed, 0 failed in 3.2s",
    scope_items=["tests/"],
)

for e in evidence:
    print(f"{e.level:12}  age={e.age_seconds:.0f}s  {e.content}")
```

## 3. Python API — SuccessCriteria + Evidence

```python
from glmagent.verification import SuccessCriteria, EvidenceLevel, extract_evidence

criteria = SuccessCriteria(
    goal="All tests pass",
    evidence_required=EvidenceLevel.FUNCTIONAL,
    recency_window_seconds=300,
    scope_items=["tests/"],
)

evidence = extract_evidence(
    source="pytest",
    text="42 passed in 3.2s",
    scope_items=criteria.scope_items,
)

if evidence and evidence[0].level >= criteria.evidence_required:
    print("Criteria met:", evidence[0].content)
else:
    print("Insufficient evidence")
```

## Next steps

- [Task engine usage guide](task_engine.md) — using `TaskEngineAgent` with structured task specs.
- [Agent API reference](../reference/agent.md) — full `VerifiedAgent` documentation.
