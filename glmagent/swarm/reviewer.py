"""Reviewer agents and a deterministic heuristic review model.

A :class:`ReviewerAgent` pairs a :class:`ReviewPerspective` with a
:class:`ReviewModel` backend. The backend may be a language model adapter or
the built-in :class:`HeuristicReviewModel`, which challenges a target using the
existing evidence framework and needs no external model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from glmagent.swarm.perspective import ReviewLens, ReviewPerspective
from glmagent.swarm.verdict import (
    ReviewFinding,
    ReviewSeverity,
    ReviewTarget,
    ReviewVerdict,
)
from glmagent.verification import EvidenceLevel


@dataclass
class ReviewRequest:
    """Everything a review model needs to evaluate one target."""

    perspective: ReviewPerspective
    target: ReviewTarget
    prior_verdicts: list[ReviewVerdict] = field(default_factory=list)
    messages: list[dict[str, str]] = field(default_factory=list)
    round_index: int = 0


class ReviewModel(Protocol):
    """Backend that produces a raw verdict payload for a review request."""

    def review(self, request: ReviewRequest) -> dict[str, Any]: ...


@dataclass
class ReviewerAgent:
    """A single swarm reviewer: one perspective backed by one review model."""

    perspective: ReviewPerspective
    model: ReviewModel

    def review(
        self,
        target: ReviewTarget,
        *,
        prior_verdicts: list[ReviewVerdict] | None = None,
        round_index: int = 0,
    ) -> ReviewVerdict:
        """Produce this reviewer's verdict on the target."""
        prior = list(prior_verdicts or [])
        request = ReviewRequest(
            perspective=self.perspective,
            target=target,
            prior_verdicts=prior,
            messages=self._build_messages(target, prior),
            round_index=round_index,
        )
        raw = self.model.review(request)
        return self._normalize(raw, round_index)

    def _build_messages(
        self,
        target: ReviewTarget,
        prior: list[ReviewVerdict],
    ) -> list[dict[str, str]]:
        """Render the prompt messages for a language-model review backend."""
        messages = [
            {"role": "system", "content": self.perspective.render()},
            {"role": "user", "content": "Review target:\n" + target.render()},
        ]
        if prior:
            lines = ["Prior reviewer verdicts in this chain:"]
            for verdict in prior:
                state = "approved" if verdict.approved else "rejected"
                lines.append(f"- {verdict.perspective} ({verdict.lens}): {state} - {verdict.rationale}")
            messages.append({"role": "user", "content": "\n".join(lines)})
        return messages

    def _normalize(self, raw: Any, round_index: int) -> ReviewVerdict:
        """Coerce a raw model payload into a structured verdict."""
        if not isinstance(raw, dict):
            msg = f"ReviewModel.review() must return a dict, got {type(raw).__name__}"
            raise TypeError(msg)

        findings = [self._normalize_finding(item) for item in raw.get("findings", []) if isinstance(item, dict)]
        has_blocker = any(finding.severity == ReviewSeverity.BLOCKER for finding in findings)
        approved = bool(raw.get("approved", not has_blocker)) and not has_blocker
        confidence = _clamp(raw.get("confidence", 0.5))
        return ReviewVerdict(
            perspective=self.perspective.name,
            lens=self.perspective.lens.value,
            approved=approved,
            confidence=confidence,
            rationale=str(raw.get("rationale", "")).strip(),
            findings=findings,
            round_index=round_index,
        )

    def _normalize_finding(self, item: dict[str, Any]) -> ReviewFinding:
        """Coerce a raw finding payload into a structured finding."""
        return ReviewFinding(
            perspective=self.perspective.name,
            lens=self.perspective.lens.value,
            severity=_coerce_severity(item.get("severity")),
            summary=str(item.get("summary", "")).strip(),
            challenge=str(item.get("challenge", self.perspective.challenge)).strip(),
            evidence_refs=[str(ref) for ref in item.get("evidence_refs", []) if str(ref)],
            scope_item=_optional_text(item.get("scope_item")),
        )


@dataclass
class HeuristicReviewModel:
    """Deterministic review backend driven by the evidence framework.

    Each perspective lens maps to a challenge routine that interrogates the
    target's claims, evidence, scope, and action. No external model is needed,
    which keeps swarm review reproducible and testable.
    """

    def review(self, request: ReviewRequest) -> dict[str, Any]:
        """Challenge the target according to the perspective's lens."""
        handler = _LENS_HANDLERS.get(request.perspective.lens, _lens_grounding)
        findings = handler(request)
        max_severity = max((finding["severity"] for finding in findings), default=None)
        approved, confidence = _verdict_from_severity(max_severity)
        rationale = self._rationale(request.perspective, findings, approved)
        return {
            "approved": approved,
            "confidence": confidence,
            "rationale": rationale,
            "findings": findings,
        }

    def _rationale(
        self,
        perspective: ReviewPerspective,
        findings: list[dict[str, Any]],
        approved: bool,
    ) -> str:
        """Summarize the heuristic outcome for the verdict."""
        if not findings:
            return f"{perspective.name}: observation withstands the {perspective.lens.value} challenge."
        verb = "passes with notes" if approved else "rejects"
        summaries = "; ".join(finding["summary"] for finding in findings)
        return f"{perspective.name} {verb}: {summaries}"


def _lens_grounding(request: ReviewRequest) -> list[dict[str, Any]]:
    """Challenge claims that are not grounded in evidence."""
    findings: list[dict[str, Any]] = []
    for claim in request.target.claims:
        if not claim.is_grounded:
            findings.append(
                _finding(
                    ReviewSeverity.BLOCKER,
                    f"Claim '{claim.subject}' is not grounded in any evidence",
                    request.perspective.challenge,
                )
            )
    if request.target.claims and not request.target.evidence:
        findings.append(
            _finding(
                ReviewSeverity.BLOCKER,
                "Target asserts claims but carries no evidence at all",
                request.perspective.challenge,
            )
        )
    return findings


def _lens_freshness(request: ReviewRequest) -> list[dict[str, Any]]:
    """Challenge stale or undated evidence."""
    target = request.target
    window = target.recency_window_seconds
    if not target.evidence:
        return [
            _finding(
                ReviewSeverity.MAJOR,
                "No evidence is present, so freshness cannot be established",
                request.perspective.challenge,
            )
        ]
    findings: list[dict[str, Any]] = []
    for entry in target.evidence:
        if entry.data_timestamp is None:
            findings.append(
                _finding(
                    ReviewSeverity.MAJOR,
                    f"Evidence for '{entry.scope_item}' has no timestamp; freshness is unverifiable",
                    request.perspective.challenge,
                    refs=[entry.reference],
                    scope_item=entry.scope_item,
                )
            )
        elif window is not None and entry.is_stale(window):
            findings.append(
                _finding(
                    ReviewSeverity.BLOCKER,
                    f"Stale evidence for '{entry.scope_item}': {entry.age_seconds:.0f}s old (limit {window}s)",
                    request.perspective.challenge,
                    refs=[entry.reference],
                    scope_item=entry.scope_item,
                )
            )
    return findings


def _lens_strength(request: ReviewRequest) -> list[dict[str, Any]]:
    """Challenge evidence that is weaker than the required level."""
    target = request.target
    required = request.perspective.min_evidence_level
    if not target.evidence:
        return [
            _finding(
                ReviewSeverity.BLOCKER,
                f"No evidence reaches the required level '{required.value}'",
                request.perspective.challenge,
            )
        ]
    findings: list[dict[str, Any]] = []
    for entry in target.evidence:
        if entry.level < required:
            findings.append(
                _finding(
                    ReviewSeverity.MAJOR,
                    f"Evidence for '{entry.scope_item}' is {entry.level.value}, below required {required.value}",
                    request.perspective.challenge,
                    refs=[entry.reference],
                    scope_item=entry.scope_item,
                )
            )
    return findings


def _lens_coverage(request: ReviewRequest) -> list[dict[str, Any]]:
    """Challenge declared scope items that lack supporting evidence."""
    target = request.target
    if not target.scope_items:
        return []
    covered = {entry.scope_item for entry in target.evidence}
    findings: list[dict[str, Any]] = []
    for item in target.scope_items:
        if item not in covered:
            findings.append(
                _finding(
                    ReviewSeverity.MAJOR,
                    f"Scope item '{item}' has no supporting evidence",
                    request.perspective.challenge,
                    scope_item=item,
                )
            )
    return findings


_SECURITY_PATTERNS: tuple[tuple[str, ReviewSeverity, str], ...] = (
    (r"\brm\s+-[a-z]*r[a-z]*f\b", ReviewSeverity.BLOCKER, "Recursive forced deletion"),
    (r"\bsudo\s+rm\b", ReviewSeverity.BLOCKER, "Privileged deletion"),
    (r"\bgit\s+reset\s+--hard\b", ReviewSeverity.BLOCKER, "Git hard reset"),
    (r"\bgit\s+push\b.*--force\b", ReviewSeverity.BLOCKER, "Force push"),
    (r"\bdrop\s+table\b", ReviewSeverity.BLOCKER, "Destructive SQL"),
    (r"\bmkfs\b", ReviewSeverity.BLOCKER, "Filesystem format"),
    (r"\bdd\s+if=", ReviewSeverity.BLOCKER, "Raw disk write"),
    (r":\(\)\s*\{", ReviewSeverity.BLOCKER, "Fork bomb"),
    (r"\b(?:shutdown|reboot)\b", ReviewSeverity.MAJOR, "Host power state change"),
    (r"\bchmod\s+-R?\s*777\b", ReviewSeverity.MAJOR, "World-writable permissions"),
    (r"\bcurl\b.+\|\s*(?:sh|bash)\b", ReviewSeverity.MAJOR, "Piping remote script to shell"),
    (r"\bgit\s+push\b", ReviewSeverity.MINOR, "Git push to remote"),
)


def _lens_security(request: ReviewRequest) -> list[dict[str, Any]]:
    """Challenge risky or destructive operations in the action or observation."""
    text = f"{request.target.action}\n{request.target.observation}"
    findings: list[dict[str, Any]] = []
    for pattern, severity, description in _SECURITY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            findings.append(
                _finding(
                    severity,
                    f"Potentially unsafe operation detected: {description}",
                    request.perspective.challenge,
                )
            )
    return findings


def _lens_adversarial(request: ReviewRequest) -> list[dict[str, Any]]:
    """Challenge overconfidence, weak proof, and easy consensus."""
    target = request.target
    findings: list[dict[str, Any]] = []
    strong = any(
        entry.level in {EvidenceLevel.FUNCTIONAL, EvidenceLevel.COMPARATIVE} for entry in target.evidence
    )
    if target.claims and not strong:
        findings.append(
            _finding(
                ReviewSeverity.MAJOR,
                "Claims rest on weak evidence; demand functional or comparative proof",
                request.perspective.challenge,
            )
        )
    if request.prior_verdicts and all(verdict.approved for verdict in request.prior_verdicts):
        findings.append(
            _finding(
                ReviewSeverity.MINOR,
                f"{len(request.prior_verdicts)} prior reviewers approved unanimously; insist on independent confirmation",
                request.perspective.challenge,
            )
        )
    if not findings:
        findings.append(
            _finding(
                ReviewSeverity.INFO,
                "No contradiction found, but absence of evidence is not evidence of absence",
                request.perspective.challenge,
            )
        )
    return findings


_LENS_HANDLERS = {
    ReviewLens.GROUNDING: _lens_grounding,
    ReviewLens.FRESHNESS: _lens_freshness,
    ReviewLens.STRENGTH: _lens_strength,
    ReviewLens.COVERAGE: _lens_coverage,
    ReviewLens.SECURITY: _lens_security,
    ReviewLens.ADVERSARIAL: _lens_adversarial,
}


def _finding(
    severity: ReviewSeverity,
    summary: str,
    challenge: str,
    *,
    refs: list[str] | None = None,
    scope_item: str | None = None,
) -> dict[str, Any]:
    """Build a raw finding payload for the heuristic model."""
    return {
        "severity": severity,
        "summary": summary,
        "challenge": challenge,
        "evidence_refs": refs or [],
        "scope_item": scope_item,
    }


def _verdict_from_severity(max_severity: ReviewSeverity | None) -> tuple[bool, float]:
    """Map the most severe finding to an approval decision and confidence."""
    if max_severity is None:
        return True, 0.9
    if max_severity == ReviewSeverity.BLOCKER:
        return False, 0.2
    if max_severity == ReviewSeverity.MAJOR:
        return False, 0.4
    if max_severity == ReviewSeverity.MINOR:
        return True, 0.6
    return True, 0.75


def _coerce_severity(value: Any) -> ReviewSeverity:
    """Parse a string or enum into a :class:`ReviewSeverity`."""
    if isinstance(value, ReviewSeverity):
        return value
    if isinstance(value, str):
        try:
            return ReviewSeverity(value.lower())
        except ValueError:
            return ReviewSeverity.MINOR
    return ReviewSeverity.MINOR


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a numeric confidence into the unit interval."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, number))


def _optional_text(value: Any) -> str | None:
    """Normalize an optional text field."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
