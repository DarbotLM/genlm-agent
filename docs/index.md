# GenLM-Agent

GenLM-Agent is a **verification-first** coding-agent toolkit. Every action passes through structured gates before the agent proceeds — no silent failures, no ungrounded claims.

## Core components

### VerifiedAgent

A guarded execution loop that enforces four verification gates on every step:

| Gate | Role |
|------|------|
| **Intent** | Declare success criteria before acting |
| **Action** | Assess risk and prerequisites |
| **Observation** | Verify evidence freshness and depth |
| **Verdict** | Confirm claims are grounded in evidence |

The gates are structural — the loop cannot skip them.

### TaskEngineAgent

A repository-aware agent for smaller generative models that act as constrained task processors. It uses:

- A **fixed task vocabulary** (`inspect`, `search`, `read`, `write`, `patch`, `run`, `test`, `verify`, `complete`).
- A **`SemanticDecisionTree`** that routes work through finite layers: `triage → localize → edit → verify → complete`.
- An **`AdaptiveFlashcardDeck`** that compresses verified facts and failure patterns — replacing full trajectory replay.

### Evidence framework

Typed evidence with an ordered strength hierarchy:

| Level | Meaning |
|-------|---------|
| `proxy` | Process exists, exit code 0 |
| `indicator` | Service shows "active", no errors |
| `behavioral` | Recent activity observed |
| `functional` | Desired outcome confirmed (tests passed) |
| `comparative` | Before/after comparison confirms change |

Evidence carries a `data_timestamp`, a freshness window, and a scope. Stale or out-of-scope evidence fails the Observation Gate.

## Modules

| Module | Purpose |
|--------|---------|
| `glmagent.agent` | `VerifiedAgent`, `AgentConfig`, `StepResult` |
| `glmagent.task_engine` | `TaskEngineAgent`, `TaskSpec`, `SemanticDecisionTree`, `AdaptiveFlashcardDeck` |
| `glmagent.verification` | `Evidence`, `EvidenceLevel`, `extract_evidence`, claim helpers |
| `glmagent.run` | CLI entrypoint |

## Getting started

Install the package:

```bash
pip install glmagent
```

Then follow the [hello world guide](usage/hello_world.md) or jump straight to the [API reference](reference/agent.md).
