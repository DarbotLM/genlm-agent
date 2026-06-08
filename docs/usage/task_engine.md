# Task engine

GenLM-Agent includes a narrow task-engine mode for smaller generative SWE models that act as task processors rather than open-ended chat agents.

## Design goals

- Use a fixed task vocabulary.
- Expose repository state as a structured snapshot.
- Route tasks through a finite semantic decision tree instead of open-ended chain-of-thought branching.
- Require explicit file, command, and verification scope targets.
- Keep each model step narrow enough for smaller transformer models to execute reliably.
- Compress verified facts and anti-patterns into adaptive flashcards instead of replaying the full history.

## Why this shape

Recent SWE-bench agents have converged on a few patterns:

- Constrained interfaces and narrow action vocabularies outperform unconstrained shell chatter.
- Strong localization and verification loops matter more than longer free-form reasoning traces.
- Lightweight state compression beats replaying the entire trajectory to the model every step.

GenLM-Agent reflects this with Generative Layer Management:

- `triage → localize → edit → verify → complete`
- Layer-specific allowed task kinds.
- Adaptive flashcards that retain verified facts and common failure modes.

## TaskSpec fields

| Field | Type | Purpose |
|-------|------|---------|
| `kind` | `TaskKind` | One of `inspect`, `search`, `read`, `write`, `patch`, `run`, `test`, `verify`, `complete` |
| `title` | `str` | Human-readable task name |
| `rationale` | `str` | Why this task is being run |
| `files` | `list[str]` | File paths in scope |
| `command` | `str \| None` | Shell command to execute |
| `patch` | `str \| None` | Unified diff to apply |
| `scope_items` | `list[str]` | Paths/identifiers for evidence scoping |
| `evidence_required` | `str` | Minimum evidence level (`proxy`…`comparative`) |
| `recency_window_seconds` | `int` | Maximum allowed evidence age |
| `completion_claim` | `str` | Claim to verify for task completion |

## Example: inspect then patch

```python
from glmagent.task_engine import (
    TaskEngineAgent, TaskEngineConfig,
    TaskSpec, TaskKind, RepositorySnapshot,
)

config = TaskEngineConfig(max_steps=20)
agent = TaskEngineAgent(model=my_model, environment=my_env, config=config)

snapshot = RepositorySnapshot(
    root="/repo",
    branch="fix/off-by-one",
    failing_targets=["tests/test_parser.py::test_offset"],
)

# Step 1: read the failing test
read_task = TaskSpec(
    kind=TaskKind.READ,
    title="Read failing test",
    rationale="Understand what the test expects",
    files=["tests/test_parser.py"],
    scope_items=["tests/test_parser.py"],
    evidence_required="indicator",
    completion_claim="Test expectations understood",
)
step1 = agent.step(read_task, snapshot)

# Step 2: apply a patch
patch_task = TaskSpec(
    kind=TaskKind.PATCH,
    title="Fix off-by-one in parser",
    rationale="Offset was 0-indexed instead of 1-indexed",
    files=["glmagent/parser.py"],
    patch="--- a/glmagent/parser.py\n+++ b/glmagent/parser.py\n...",
    scope_items=["glmagent/parser.py"],
    evidence_required="indicator",
    completion_claim="Patch applied cleanly",
)
step2 = agent.step(patch_task, snapshot)

# Step 3: run tests
test_task = TaskSpec(
    kind=TaskKind.TEST,
    title="Run failing test",
    rationale="Confirm fix resolves the failure",
    command="pytest tests/test_parser.py::test_offset -v",
    scope_items=["tests/test_parser.py"],
    evidence_required="functional",
    completion_claim="test_offset passes",
)
step3 = agent.step(test_task, snapshot)
```

## SemanticDecisionTree

The `SemanticDecisionTree` enforces which task kinds are allowed at each layer:

| Layer | Allowed task kinds |
|-------|--------------------|
| `triage` | `inspect`, `search`, `read`, `test`, `verify` |
| `localize` | `search`, `read`, `verify` |
| `edit` | `read`, `write`, `patch`, `run` |
| `verify` | `read`, `run`, `test`, `verify` |
| `complete` | `complete`, `read`, `test`, `verify` |

```python
from glmagent.task_engine import SemanticDecisionTree, GenerativeLayer, TaskKind

tree = SemanticDecisionTree()
print(tree.current_layer)         # GenerativeLayer.TRIAGE
print(tree.allowed_tasks())       # (inspect, search, read, test, verify)

errors = tree.validate(TaskKind.PATCH)
print(errors)  # ["Task kind 'patch' is not allowed in layer 'triage'"]

next_layer = tree.advance(TaskKind.READ, verified=True, done=False)
print(next_layer)  # GenerativeLayer.LOCALIZE
```

## AdaptiveFlashcardDeck

Flashcards replace full trajectory replay with a compact verified-facts summary:

```python
from glmagent.task_engine.layers import AdaptiveFlashcardDeck, AdaptiveFlashcard

deck = AdaptiveFlashcardDeck()
deck.add(AdaptiveFlashcard(
    fact="Parser uses 1-indexed offsets",
    kind="verified",
    source="test_parser.py::test_offset",
))
deck.add(AdaptiveFlashcard(
    fact="Writing to read-only path raises PermissionError",
    kind="failure",
    source="step-3",
))

print(deck.render(max_cards=6))
```
