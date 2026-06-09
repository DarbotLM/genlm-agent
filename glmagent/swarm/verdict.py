"""Verdict, finding, and target types for observational swarm review."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from glmagent.runtime import to_json_compatible
from glmagent.verification import Claim, Evidence, extract_evidence


class ReviewSeverity(StrEnum):
    """Ordered severity of a single review finding.

    The ordering is by declaration, not lexicographic, so ``BLOCKER`` is the
    most severe and ``INFO`` the least.
    """

    INFO = "info"
    MINOR = "minor"
    MAJOR = "major"
    BLOCKER = "blocker"

    def __ge__(self, other: ReviewSeverity) -> bool:
        order = list(ReviewSeverity)
        return order.index(self) >= order.index(other)

    def __gt__(self, other: ReviewSeverity) -> bool:
        order = list(ReviewSeverity)
        return order.index(self) > order.index(other)

    def __le__(self, other: ReviewSeverity) -> bool:
        order = list(ReviewSeverity)
        return order.index(self) <= order.index(other)

    def __lt__(self, other: ReviewSeverity) -> bool:
        order = list(ReviewSeverity)
        return order.index(self) < order.index(other)


@dataclass
class ReviewFinding:
    """A single challenge raised by a reviewer against the target."""

    perspective: str
    lens: str
    severity: ReviewSeverity
    summary: str
    challenge: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    scope_item: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Render the finding into JSON-safe data."""
        return to_json_compatible(self)


@dataclass
class ReviewVerdict:
    """One reviewer's verdict on the target, with grounded findings."""

    perspective: str
    lens: str
    approved: bool
    confidence: float
    rationale: str = ""
    findings: list[ReviewFinding] = field(default_factory=list)
    round_index: int = 0

    @property
    def blockers(self) -> list[ReviewFinding]:
        """Findings that must block approval."""
        return [finding for finding in self.findings if finding.severity == ReviewSeverity.BLOCKER]

    @property
    def has_blockers(self) -> bool:
        """Whether any finding is a blocker."""
        return any(finding.severity == ReviewSeverity.BLOCKER for finding in self.findings)

    @property
    def max_severity(self) -> ReviewSeverity | None:
        """The most severe finding, or ``None`` when there are no findings."""
        if not self.findings:
            return None
        return max(finding.severity for finding in self.findings)

    def to_dict(self) -> dict[str, Any]:
        """Render the verdict into JSON-safe data."""
        return to_json_compatible(self)


@dataclass
class ReviewTarget:
    """The observation under review, plus the evidence and claims around it."""

    subject: str
    observation: str = ""
    action: str = ""
    claims: list[Claim] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    scope_items: list[str] = field(default_factory=list)
    recency_window_seconds: int | None = 300
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_observation(
        cls,
        observation: str,
        *,
        subject: str = "observation",
        action: str = "",
        scope_items: list[str] | None = None,
        source: str = "review",
        recency_window_seconds: int | None = 300,
    ) -> ReviewTarget:
        """Build a target from raw observation text, extracting evidence."""
        requested = scope_items or ["unknown"]
        evidence = extract_evidence(observation, source=source, scope_items=requested)
        meaningful_scope = [item for item in requested if item != "unknown"]
        return cls(
            subject=subject,
            observation=observation,
            action=action,
            evidence=evidence,
            scope_items=meaningful_scope,
            recency_window_seconds=recency_window_seconds,
        )

    @classmethod
    def from_step(cls, result: Any, *, subject: str | None = None) -> ReviewTarget:
        """Build a target from an agent ``StepResult`` or ``TaskStepResult``."""
        intent = getattr(result, "intent", None)
        scope_items = list(getattr(intent, "scope_items", []) or [])
        recency = getattr(intent, "recency_window_seconds", 300)
        resolved_subject = subject or getattr(intent, "goal", "") or "agent-step"
        return cls(
            subject=resolved_subject,
            observation=getattr(result, "observation", ""),
            action=getattr(result, "action", ""),
            claims=list(getattr(result, "claims", []) or []),
            evidence=list(getattr(result, "evidence", []) or []),
            scope_items=scope_items,
            recency_window_seconds=recency,
        )

    def render(self) -> str:
        """Render the target into prompt-friendly review context."""
        lines = [f"Subject: {self.subject}"]
        if self.action:
            lines.append(f"Proposed action: {self.action}")
        if self.scope_items:
            lines.append("Scope items: " + ", ".join(self.scope_items))
        lines.append(f"Recency window (seconds): {self.recency_window_seconds}")
        lines.append("Observation:")
        lines.append(self.observation or "(empty)")
        if self.claims:
            lines.append("Claims:")
            for claim in self.claims:
                state = "grounded" if claim.is_grounded else "UNGROUNDED"
                lines.append(f"- {claim.subject} [{state}]")
        if self.evidence:
            lines.append("Evidence:")
            for entry in self.evidence:
                age = f"{entry.age_seconds:.0f}s" if entry.age_seconds is not None else "n/a"
                lines.append(f"- {entry.scope_item}: level={entry.level.value} age={age} ref={entry.reference}")
        return "\n".join(lines)
