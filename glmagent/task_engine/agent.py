"""Task-engine agent for constrained repository workflows."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from glmagent.agent import AgentConfig, StepResult, VerifiedAgent
from glmagent.runtime import SessionEventType, SessionPresenter
from glmagent.task_engine.layers import (
    AdaptiveFlashcardDeck,
    GenerativeLayer,
    SemanticDecisionTree,
)
from glmagent.task_engine.spec import (
    RepositorySnapshot,
    TaskEnvironment,
    TaskKind,
    TaskPolicy,
    TaskSpec,
)
from glmagent.verification import Evidence, SuccessCriteria


@dataclass
class TaskEngineConfig(AgentConfig):
    """Configuration for task-engine agents tuned for smaller models."""

    require_intent: bool = False
    max_flashcards_per_prompt: int = 6
    policy: TaskPolicy = field(default_factory=TaskPolicy)


@dataclass
class TaskStepResult(StepResult):
    """Step result extended with task-engine context."""

    task: TaskSpec | None = None
    snapshot_before: RepositorySnapshot | None = None
    snapshot_after: RepositorySnapshot | None = None
    layer_before: GenerativeLayer | None = None
    layer_after: GenerativeLayer | None = None


class TaskEngineAgent(VerifiedAgent):
    """Repository-aware agent that constrains models to a task schema."""

    agent_type = "task-engine"

    def __init__(
        self,
        model: Any,
        env: TaskEnvironment,
        config: TaskEngineConfig | None = None,
        presenter: SessionPresenter | None = None,
    ):
        resolved_config = config or TaskEngineConfig()
        super().__init__(
            model=model,
            env=env,
            config=resolved_config,
            presenter=presenter,
        )
        self.config = resolved_config
        self._decision_tree = SemanticDecisionTree()
        self._flashcards = AdaptiveFlashcardDeck()
        self._verified_evidence_by_scope: dict[str, Evidence] = {}

    def step(self) -> TaskStepResult:
        """Execute one repository task with verification."""
        result = TaskStepResult()
        result.layer_before = self._decision_tree.current_layer
        result.snapshot_before = self.env.snapshot()

        raw_output = self.model.query(self._build_messages(result.snapshot_before))
        task = self._parse_task_output(raw_output)
        result.task = task
        result.thought = self._build_thought(task)
        result.action = task.to_action_string()
        result.done = task.kind == TaskKind.COMPLETE
        result.intent = task.to_success_criteria(self.config.max_staleness_seconds)
        self._merge_scope_from_intent(result.intent)

        result.verification_gaps.extend(self._decision_tree.validate(task.kind))
        result.verification_gaps.extend(task.validate(self.config.policy))
        if result.verification_gaps:
            result.observation = "TASK INVALID: " + "; ".join(result.verification_gaps)
            self._finalize_learning(result)
            self._record_task_step(result)
            return result

        result.action_review = self._review_action(result.action)
        if result.action and not result.action_review.get("proceed", True):
            result.observation = "ACTION BLOCKED: " + ", ".join(result.action_review.get("concerns", []))
            self._finalize_learning(result)
            self._record_task_step(result)
            return result

        if task.kind == TaskKind.COMPLETE:
            result.snapshot_after = result.snapshot_before
            result.evidence = self._collect_completion_evidence(task)
            result.claims = self._extract_claims(
                task.completion_claim or result.thought,
                result.evidence,
            )
            result.verification_gaps.extend(self._check_completion(task))
            ungrounded = [claim for claim in result.claims if not claim.is_grounded]
            if ungrounded:
                result.verification_gaps.append(
                    "Ungrounded claims: " + ", ".join(claim.subject for claim in ungrounded)
                )
            result.observation = self._build_completion_observation(task, result.evidence)
            if result.verification_gaps:
                result.observation += "\nVERIFICATION GAPS: " + "; ".join(result.verification_gaps)
            result.verified = not result.verification_gaps
            self._finalize_learning(result)
            self._record_task_step(result)
            return result

        t0 = time.perf_counter()
        execution = self._normalize_execution_result(
            command=result.action,
            raw=self.env.execute_task(task, timeout=self.config.execution_timeout_seconds),
        )
        result.execution_time = time.perf_counter() - t0
        result.execution = execution
        result.observation = execution.combined_output
        result.snapshot_after = self.env.snapshot()

        result.evidence = self._collect_evidence(result.observation)
        result.verification_gaps.extend(self._check_evidence_quality(result.evidence, result.intent))
        result.claims = self._extract_claims(result.thought, result.evidence)
        ungrounded = [claim for claim in result.claims if not claim.is_grounded]
        if ungrounded:
            result.verification_gaps.append("Ungrounded claims: " + ", ".join(claim.subject for claim in ungrounded))

        self._mark_verified_scope_items(result.evidence, result.intent)
        self._cache_verified_evidence(result.evidence, result.intent)

        if result.verification_gaps:
            result.observation += "\nVERIFICATION GAPS: " + "; ".join(result.verification_gaps)

        result.verified = not result.verification_gaps
        self._finalize_learning(result)
        self._record_task_step(result)
        return result

    def run(self, problem: str) -> list[TaskStepResult]:
        """Run the task engine until it emits a verified completion task."""
        self.begin_session(
            problem,
            history=[
                {"role": "system", "content": self.config.policy.render()},
                {"role": "user", "content": problem},
            ],
        )
        self._decision_tree = SemanticDecisionTree()
        self._flashcards = AdaptiveFlashcardDeck()
        self._verified_evidence_by_scope = {}

        for _ in range(self.config.max_steps):
            result = self.step()
            if result.done and result.verified:
                break

        status = "verified" if self.trajectory and self.trajectory[-1].verified else "exhausted"
        self.finish_session(status=status)
        return self.trajectory

    def _build_messages(self, snapshot: RepositorySnapshot) -> list[dict[str, Any]]:
        """Add repository state to the model context."""
        return self.history + [
            {"role": "system", "content": self._decision_tree.render()},
            {
                "role": "system",
                "content": self._flashcards.render(
                    self._decision_tree.current_layer,
                    limit=self.config.max_flashcards_per_prompt,
                ),
            },
            {"role": "user", "content": "Repository snapshot:\n" + snapshot.render()},
        ]

    def _parse_task_output(self, raw: Any) -> TaskSpec:
        """Normalize a model reply into a task spec."""
        if not isinstance(raw, dict):
            msg = f"Task engine model must return a dict, got {type(raw).__name__}"
            raise TypeError(msg)
        task_payload = raw.get("task", raw)
        if not isinstance(task_payload, dict):
            msg = "Task engine model output must contain a task object"
            raise TypeError(msg)
        return TaskSpec.from_dict(task_payload)

    def _build_thought(self, task: TaskSpec) -> str:
        """Render a compact thought string from a task."""
        lines = [
            f"Task: {task.kind.value}",
            f"Title: {task.title}",
        ]
        if task.rationale:
            lines.append(f"Rationale: {task.rationale}")
        if task.scope_items:
            lines.append("Scope: " + ", ".join(task.scope_items))
        if task.completion_claim:
            lines.append("Status: " + task.completion_claim)
        return "\n".join(lines)

    def _cache_verified_evidence(
        self,
        evidence: list[Evidence],
        intent: SuccessCriteria | None,
    ) -> None:
        """Store the latest verified evidence for each scope item."""
        required_level = intent.evidence_required if intent is not None else self.config.min_evidence_level
        for entry in evidence:
            if entry.scope_item in self._verified_items and entry.level >= required_level:
                self._verified_evidence_by_scope[entry.scope_item] = entry

    def _collect_completion_evidence(self, task: TaskSpec) -> list[Evidence]:
        """Reuse prior verified evidence for completion claims."""
        scope_items = task.scope_items or self._scope_items
        if scope_items:
            return [
                self._verified_evidence_by_scope[item]
                for item in scope_items
                if item in self._verified_evidence_by_scope
            ]
        return list(self._verified_evidence_by_scope.values())

    def _check_completion(self, task: TaskSpec) -> list[str]:
        """Ensure completion is grounded in previously verified scope."""
        gaps: list[str] = []
        scope_items = task.scope_items or self._scope_items
        if self.config.policy.completion_requires_verified_scope and scope_items:
            missing = [item for item in scope_items if item not in self._verified_items]
            if missing:
                gaps.append("Completion requested without verified scope: " + ", ".join(missing))
        if not self._collect_completion_evidence(task):
            gaps.append("Completion requested without prior verified evidence")
        return gaps

    def _build_completion_observation(
        self,
        task: TaskSpec,
        evidence: list[Evidence],
    ) -> str:
        """Render the completion observation for the model history."""
        lines = [f"COMPLETION REQUESTED: {task.completion_claim or task.title}"]
        if evidence:
            lines.append("Grounded by evidence: " + ", ".join(entry.reference for entry in evidence))
        return "\n".join(lines)

    def _finalize_learning(self, result: TaskStepResult) -> None:
        """Advance the decision tree and refresh adaptive flashcards."""
        if result.task is None:
            result.layer_after = result.layer_before
            return

        result.layer_after = self._decision_tree.advance(
            result.task.kind,
            verified=result.verified,
            done=result.done and result.verified,
        )
        step_index = len(self.trajectory) + 1
        if result.verified:
            self._flashcards.remember_verified(
                result,
                result.layer_before or self._decision_tree.current_layer,
                step_index,
            )
        elif result.verification_gaps:
            self._flashcards.remember_gap(
                result,
                result.layer_before or self._decision_tree.current_layer,
                step_index,
            )

    def _record_task_step(self, result: TaskStepResult) -> None:
        """Append task-aware history entries."""
        assistant_lines = [result.thought or "Task: (empty)"]
        if result.layer_before is not None:
            if result.layer_after is not None:
                assistant_lines.append(f"Layer: {result.layer_before.value} -> {result.layer_after.value}")
            else:
                assistant_lines.append(f"Layer: {result.layer_before.value}")
        if result.task is not None and result.task.files:
            assistant_lines.append("Files: " + ", ".join(result.task.files))
        if result.action:
            assistant_lines.append(f"Action: {result.action}")
        if result.done:
            assistant_lines.append("Done: true")

        user_lines = [result.observation]
        snapshot = result.snapshot_after or result.snapshot_before
        if snapshot is not None:
            user_lines.append("Repository snapshot:\n" + snapshot.render())

        self.history.append({"role": "assistant", "content": "\n".join(assistant_lines)})
        self.history.append({"role": "user", "content": "\n\n".join(user_lines)})
        self.trajectory.append(result)
        self._publish_session(SessionEventType.STEP_RECORDED)
