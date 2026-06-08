"""Task-engine specification for repository-centric coding agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from glmagent.agent import ExecutionResult
from glmagent.verification import EvidenceLevel, SuccessCriteria


class TaskKind(StrEnum):
    """Narrow task vocabulary for processor-style coding agents."""

    INSPECT = "inspect"
    SEARCH = "search"
    READ = "read"
    WRITE = "write"
    PATCH = "patch"
    RUN = "run"
    TEST = "test"
    VERIFY = "verify"
    COMPLETE = "complete"


@dataclass
class RepositorySnapshot:
    """Structured repository state exposed to the task engine."""

    root: str
    branch: str | None = None
    dirty_files: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    staged_files: list[str] = field(default_factory=list)
    failing_targets: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Render the repository snapshot into compact text."""
        lines = [f"root: {self.root}"]
        if self.branch:
            lines.append(f"branch: {self.branch}")
        lines.append("dirty_files: " + _render_list(self.dirty_files))
        lines.append("changed_files: " + _render_list(self.changed_files))
        lines.append("staged_files: " + _render_list(self.staged_files))
        lines.append("failing_targets: " + _render_list(self.failing_targets))
        if self.notes:
            lines.append("notes: " + _render_list(self.notes))
        return "\n".join(lines)


@dataclass
class TaskPolicy:
    """Policy constraints for narrow task-engine agents."""

    allowed_tasks: tuple[TaskKind, ...] = (
        TaskKind.INSPECT,
        TaskKind.SEARCH,
        TaskKind.READ,
        TaskKind.WRITE,
        TaskKind.PATCH,
        TaskKind.RUN,
        TaskKind.TEST,
        TaskKind.VERIFY,
        TaskKind.COMPLETE,
    )
    max_files_per_task: int = 8
    require_scope_items: bool = True
    require_file_targets_for_writes: bool = True
    require_patch_for_patch_tasks: bool = True
    require_command_for_shell_tasks: bool = True
    completion_requires_verified_scope: bool = True

    def render(self) -> str:
        """Render the policy into a model-facing contract."""
        return "\n".join(
            [
                "Task engine contract:",
                "Return a JSON object with a top-level 'task' object.",
                "Task fields: kind, title, rationale, files, command, patch, scope_items, evidence_required, recency_window_seconds, completion_claim.",
                "Allowed task kinds: " + ", ".join(task.value for task in self.allowed_tasks),
                f"Max files per task: {self.max_files_per_task}",
                f"Require scope items: {self.require_scope_items}",
                f"Require explicit file targets for writes: {self.require_file_targets_for_writes}",
                f"Require explicit commands for run/test/verify tasks: {self.require_command_for_shell_tasks}",
                f"Completion requires verified scope: {self.completion_requires_verified_scope}",
                "Prefer narrow, single-purpose tasks over long free-form plans.",
            ]
        )


@dataclass
class TaskSpec:
    """One structured task emitted by a task-engine model."""

    kind: TaskKind
    title: str
    rationale: str = ""
    files: list[str] = field(default_factory=list)
    command: str | None = None
    patch: str | None = None
    scope_items: list[str] = field(default_factory=list)
    evidence_required: EvidenceLevel = EvidenceLevel.FUNCTIONAL
    recency_window_seconds: int | None = 300
    completion_claim: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskSpec:
        """Parse a task spec from a model payload."""
        kind = payload.get("kind")
        if not isinstance(kind, str):
            msg = "task.kind must be a string"
            raise TypeError(msg)

        return cls(
            kind=TaskKind(kind.lower()),
            title=str(payload.get("title", "")).strip(),
            rationale=str(payload.get("rationale", "")).strip(),
            files=_normalize_text_list(payload.get("files", [])),
            command=_normalize_optional_text(payload.get("command")),
            patch=_normalize_optional_text(payload.get("patch")),
            scope_items=_normalize_text_list(payload.get("scope_items", [])),
            evidence_required=_coerce_evidence_level(payload.get("evidence_required")),
            recency_window_seconds=_coerce_optional_int(payload.get("recency_window_seconds"), 300),
            completion_claim=_normalize_optional_text(payload.get("completion_claim")),
        )

    def validate(self, policy: TaskPolicy) -> list[str]:
        """Validate a task against the narrow task-engine policy."""
        gaps: list[str] = []
        if self.kind not in policy.allowed_tasks:
            gaps.append(f"Task kind '{self.kind.value}' is not allowed by policy")
        if not self.title:
            gaps.append("Task title is required")
        if len(self.files) > policy.max_files_per_task:
            gaps.append(f"Task references {len(self.files)} files; limit is {policy.max_files_per_task}")
        if policy.require_scope_items and self.kind != TaskKind.INSPECT and not self.scope_items:
            gaps.append("Task scope_items are required")
        if policy.require_file_targets_for_writes and self.kind in {TaskKind.WRITE, TaskKind.PATCH} and not self.files:
            gaps.append(f"{self.kind.value} tasks must target at least one file")
        if policy.require_patch_for_patch_tasks and self.kind == TaskKind.PATCH and not self.patch:
            gaps.append("Patch tasks must include a patch payload")
        if (
            policy.require_command_for_shell_tasks
            and self.kind in {TaskKind.RUN, TaskKind.TEST, TaskKind.VERIFY}
            and not self.command
        ):
            gaps.append(f"{self.kind.value} tasks must include a command")
        if self.kind == TaskKind.COMPLETE and not self.completion_claim:
            gaps.append("Completion tasks must include a completion_claim")
        return gaps

    def to_action_string(self) -> str:
        """Serialize the task into a reviewable action string."""
        if self.command:
            return self.command
        if self.kind == TaskKind.PATCH and self.files:
            return f"patch {' '.join(self.files)}"
        if self.kind == TaskKind.WRITE and self.files:
            return f"write {' '.join(self.files)}"
        if self.files:
            return f"{self.kind.value} {' '.join(self.files)}"
        return self.kind.value

    def to_success_criteria(self, default_recency: int) -> SuccessCriteria:
        """Convert the task into verification criteria."""
        scope_items = self.scope_items or self.files or [self.title]
        return SuccessCriteria(
            goal=self.title,
            evidence_required=self.evidence_required,
            recency_window_seconds=(
                self.recency_window_seconds if self.recency_window_seconds is not None else default_recency
            ),
            scope_items=scope_items,
        )


class TaskEnvironment(Protocol):
    """Repository-aware environment for task-engine agents."""

    def snapshot(self) -> RepositorySnapshot: ...

    def execute_task(
        self,
        task: TaskSpec,
        timeout: float = 30.0,
    ) -> ExecutionResult | str: ...


def _render_list(items: list[str]) -> str:
    return ", ".join(items) if items else "(none)"


def _normalize_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value or [])
    return [text for item in values if (text := str(item).strip())]


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_optional_int(value: Any, default: int | None) -> int | None:
    if value is None:
        return default
    return int(value)


def _coerce_evidence_level(value: Any) -> EvidenceLevel:
    if isinstance(value, EvidenceLevel):
        return value
    if isinstance(value, str):
        return EvidenceLevel(value.lower())
    return EvidenceLevel.FUNCTIONAL
