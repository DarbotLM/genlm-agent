"""Reviewer perspectives for observational swarm review.

Each perspective is a uniquely challenging lens that one reviewer agent applies
to an observation. Perspectives are intentionally adversarial: a reviewer
approves only when the observation withstands its specific challenge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from glmagent.verification import EvidenceLevel


class ReviewLens(StrEnum):
    """Deterministic challenge lens used by heuristic reviewers.

    The lens selects how a :class:`HeuristicReviewModel` interrogates a target.
    Language-model reviewers may ignore the lens, but it still documents the
    angle of attack for the perspective.
    """

    GROUNDING = "grounding"
    FRESHNESS = "freshness"
    STRENGTH = "strength"
    COVERAGE = "coverage"
    SECURITY = "security"
    ADVERSARIAL = "adversarial"


@dataclass
class ReviewPerspective:
    """A uniquely challenging lens applied by one reviewer agent."""

    name: str
    lens: ReviewLens
    challenge: str
    focus: list[str] = field(default_factory=list)
    min_evidence_level: EvidenceLevel = EvidenceLevel.BEHAVIORAL
    weight: float = 1.0

    def render(self) -> str:
        """Render the perspective into a model-facing system prompt."""
        lines = [
            f"You are the '{self.name}' reviewer in an observational review swarm.",
            f"Challenge lens: {self.lens.value}.",
            f"Core challenge: {self.challenge}",
            f"Minimum acceptable evidence level: {self.min_evidence_level.value}.",
        ]
        if self.focus:
            lines.append("Focus areas: " + ", ".join(self.focus))
        lines.append(
            "Raise findings with severity blocker, major, minor, or info. "
            "Approve only if the observation withstands your challenge."
        )
        return "\n".join(lines)


def skeptic() -> ReviewPerspective:
    """Challenge every claim that is not grounded in concrete evidence."""
    return ReviewPerspective(
        name="skeptic",
        lens=ReviewLens.GROUNDING,
        challenge="Every claim must be backed by a concrete evidence reference. Trust nothing asserted without proof.",
        focus=["claims", "grounding"],
        min_evidence_level=EvidenceLevel.INDICATOR,
    )


def freshness_auditor() -> ReviewPerspective:
    """Challenge evidence that is stale or has no verifiable timestamp."""
    return ReviewPerspective(
        name="freshness-auditor",
        lens=ReviewLens.FRESHNESS,
        challenge="Evidence must be recent and timestamped. Old or undated observations cannot prove the current state.",
        focus=["recency", "timestamps"],
        min_evidence_level=EvidenceLevel.BEHAVIORAL,
    )


def evidence_strength_auditor() -> ReviewPerspective:
    """Challenge weak evidence levels that fall below a functional bar."""
    return ReviewPerspective(
        name="strength-auditor",
        lens=ReviewLens.STRENGTH,
        challenge="Proxy and indicator signals are not outcomes. Demand functional or comparative proof of success.",
        focus=["evidence-strength"],
        min_evidence_level=EvidenceLevel.FUNCTIONAL,
    )


def scope_coverage_auditor() -> ReviewPerspective:
    """Challenge scope items that lack any supporting evidence."""
    return ReviewPerspective(
        name="coverage-auditor",
        lens=ReviewLens.COVERAGE,
        challenge="Every declared scope item must be individually verified. Partial coverage is not completion.",
        focus=["scope", "coverage"],
        min_evidence_level=EvidenceLevel.INDICATOR,
    )


def security_reviewer() -> ReviewPerspective:
    """Challenge actions that perform risky or destructive operations."""
    return ReviewPerspective(
        name="security-reviewer",
        lens=ReviewLens.SECURITY,
        challenge="Assume the action is hostile. Flag destructive, privileged, or irreversible operations before they run.",
        focus=["safety", "destructive-commands"],
        min_evidence_level=EvidenceLevel.INDICATOR,
    )


def red_team() -> ReviewPerspective:
    """Challenge consensus and overconfidence from an adversarial stance."""
    return ReviewPerspective(
        name="red-team",
        lens=ReviewLens.ADVERSARIAL,
        challenge="Argue the opposite. Surface the failure mode everyone else missed and resist easy consensus.",
        focus=["adversarial", "consensus"],
        min_evidence_level=EvidenceLevel.FUNCTIONAL,
    )


def default_review_panel() -> list[ReviewPerspective]:
    """Return a curated panel of uniquely challenging perspectives."""
    return [
        skeptic(),
        freshness_auditor(),
        evidence_strength_auditor(),
        scope_coverage_auditor(),
        security_reviewer(),
        red_team(),
    ]


#: Built-in perspective factories keyed by name, for CLI and config wiring.
BUILTIN_PERSPECTIVES = {
    "skeptic": skeptic,
    "freshness-auditor": freshness_auditor,
    "strength-auditor": evidence_strength_auditor,
    "coverage-auditor": scope_coverage_auditor,
    "security-reviewer": security_reviewer,
    "red-team": red_team,
}
