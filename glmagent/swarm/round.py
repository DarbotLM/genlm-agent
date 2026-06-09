"""Parallel review round: many reviewers challenge one target concurrently."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from glmagent.runtime import to_json_compatible
from glmagent.swarm.reviewer import ReviewerAgent
from glmagent.swarm.verdict import ReviewFinding, ReviewVerdict


@dataclass
class ReviewPolicy:
    """Aggregation and execution policy for review rounds and chains."""

    approval_threshold: float = 1.0
    block_on_blocker: bool = True
    stop_on_blocker: bool = False
    max_workers: int = 8

    def approves(self, approvals: int, total: int, blockers: list[ReviewFinding]) -> bool:
        """Decide whether a round's verdicts amount to approval."""
        if total == 0:
            return True
        if self.block_on_blocker and blockers:
            return False
        return (approvals / total) >= self.approval_threshold


@dataclass
class RoundResult:
    """Aggregate outcome of one parallel review round."""

    round_index: int
    name: str
    verdicts: list[ReviewVerdict]
    approved: bool
    blockers: list[ReviewFinding] = field(default_factory=list)
    dissenting: list[str] = field(default_factory=list)
    consensus_confidence: float = 0.0

    @property
    def all_findings(self) -> list[ReviewFinding]:
        """Every finding raised across the round."""
        return [finding for verdict in self.verdicts for finding in verdict.findings]

    def to_dict(self) -> dict[str, Any]:
        """Render the round result into JSON-safe data."""
        return to_json_compatible(self)


@dataclass
class ReviewRound:
    """A set of reviewers that challenge the same target in parallel."""

    reviewers: list[ReviewerAgent]
    policy: ReviewPolicy = field(default_factory=ReviewPolicy)
    name: str = "review-round"

    def run(
        self,
        target: Any,
        *,
        prior_verdicts: list[ReviewVerdict] | None = None,
        round_index: int = 0,
    ) -> RoundResult:
        """Run every reviewer against the target and aggregate the verdicts."""
        prior = list(prior_verdicts or [])
        verdicts = self._dispatch(target, prior, round_index)
        return self._aggregate(verdicts, round_index)

    def _dispatch(
        self,
        target: Any,
        prior: list[ReviewVerdict],
        round_index: int,
    ) -> list[ReviewVerdict]:
        """Execute reviewers, preserving reviewer order in the results."""
        workers = max(1, min(self.policy.max_workers, len(self.reviewers)))
        if workers == 1 or len(self.reviewers) <= 1:
            return [
                reviewer.review(target, prior_verdicts=prior, round_index=round_index)
                for reviewer in self.reviewers
            ]

        results: list[ReviewVerdict | None] = [None] * len(self.reviewers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    reviewer.review,
                    target,
                    prior_verdicts=prior,
                    round_index=round_index,
                ): index
                for index, reviewer in enumerate(self.reviewers)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return [verdict for verdict in results if verdict is not None]

    def _aggregate(self, verdicts: list[ReviewVerdict], round_index: int) -> RoundResult:
        """Fold per-reviewer verdicts into a single round result."""
        blockers = [finding for verdict in verdicts for finding in verdict.blockers]
        approvals = sum(1 for verdict in verdicts if verdict.approved)
        dissenting = [verdict.perspective for verdict in verdicts if not verdict.approved]
        approved = self.policy.approves(approvals, len(verdicts), blockers)
        confidence = (
            sum(verdict.confidence for verdict in verdicts) / len(verdicts) if verdicts else 0.0
        )
        return RoundResult(
            round_index=round_index,
            name=self.name,
            verdicts=verdicts,
            approved=approved,
            blockers=blockers,
            dissenting=dissenting,
            consensus_confidence=confidence,
        )
