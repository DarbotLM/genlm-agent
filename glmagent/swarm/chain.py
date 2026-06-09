"""Serial review chain and the top-level observational reviewer orchestrator.

A :class:`ReviewChain` runs review rounds in series, accumulating verdicts so
that later rounds can challenge the conclusions of earlier ones. Combined with
the parallel :class:`ReviewRound`, this yields the headline capability: run a
swarm of uniquely challenging reviewers *in series in a parallel chain*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from glmagent.runtime import to_json_compatible
from glmagent.swarm.perspective import ReviewPerspective, default_review_panel
from glmagent.swarm.reviewer import HeuristicReviewModel, ReviewerAgent, ReviewModel
from glmagent.swarm.round import ReviewPolicy, ReviewRound, RoundResult
from glmagent.swarm.verdict import ReviewFinding, ReviewTarget, ReviewVerdict


class ReviewMode(StrEnum):
    """Execution strategy for an observational review."""

    PARALLEL = "parallel"
    SERIES = "series"
    CHAIN = "chain"


@dataclass
class ChainResult:
    """Aggregate outcome of a serial chain of review rounds."""

    approved: bool
    rounds: list[RoundResult] = field(default_factory=list)
    verdicts: list[ReviewVerdict] = field(default_factory=list)
    blockers: list[ReviewFinding] = field(default_factory=list)
    consensus_confidence: float = 0.0
    stopped_early: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Render the chain result into JSON-safe data."""
        return to_json_compatible(self)


@dataclass
class ReviewChain:
    """An ordered series of review rounds with accumulating context."""

    rounds: list[ReviewRound]
    policy: ReviewPolicy = field(default_factory=ReviewPolicy)

    def run(self, target: ReviewTarget) -> ChainResult:
        """Run each round in series, threading prior verdicts forward."""
        accumulated: list[ReviewVerdict] = []
        round_results: list[RoundResult] = []
        stopped_early = False

        for index, review_round in enumerate(self.rounds):
            result = review_round.run(target, prior_verdicts=accumulated, round_index=index)
            round_results.append(result)
            accumulated = accumulated + result.verdicts
            if self.policy.stop_on_blocker and result.blockers:
                stopped_early = True
                break

        blockers = [finding for result in round_results for finding in result.blockers]
        approved = bool(round_results) and all(result.approved for result in round_results)
        if self.policy.block_on_blocker and blockers:
            approved = False
        confidence = (
            sum(verdict.confidence for verdict in accumulated) / len(accumulated) if accumulated else 0.0
        )
        return ChainResult(
            approved=approved,
            rounds=round_results,
            verdicts=accumulated,
            blockers=blockers,
            consensus_confidence=confidence,
            stopped_early=stopped_early,
        )


@dataclass
class ObservationalReviewer:
    """Orchestrate a swarm of challenging reviewers over an observation.

    The same panel of perspectives can be run as a single parallel round, as a
    serial chain of single-reviewer rounds, or as a serial chain of parallel
    rounds (``series in a parallel chain``).
    """

    perspectives: list[ReviewPerspective]
    model: ReviewModel = field(default_factory=HeuristicReviewModel)
    policy: ReviewPolicy = field(default_factory=ReviewPolicy)

    @classmethod
    def with_default_panel(
        cls,
        model: ReviewModel | None = None,
        policy: ReviewPolicy | None = None,
    ) -> ObservationalReviewer:
        """Build a reviewer with the curated default panel of perspectives."""
        return cls(
            perspectives=default_review_panel(),
            model=model or HeuristicReviewModel(),
            policy=policy or ReviewPolicy(),
        )

    def review(self, target: ReviewTarget, mode: ReviewMode = ReviewMode.PARALLEL) -> ChainResult:
        """Review a target using the requested execution mode."""
        if mode == ReviewMode.PARALLEL:
            return _wrap_round(self.review_parallel(target))
        if mode == ReviewMode.SERIES:
            return self.review_series(target)
        return self.review_chain(target)

    def review_parallel(self, target: ReviewTarget) -> RoundResult:
        """Run all perspectives concurrently in a single round."""
        return ReviewRound(self._reviewers(), policy=self.policy, name="parallel-round").run(target)

    def review_series(self, target: ReviewTarget) -> ChainResult:
        """Run each perspective in series so each sees the prior verdicts."""
        rounds = [
            ReviewRound([reviewer], policy=self.policy, name=f"series-{reviewer.perspective.name}")
            for reviewer in self._reviewers()
        ]
        return ReviewChain(rounds, policy=self.policy).run(target)

    def review_chain(
        self,
        target: ReviewTarget,
        groups: list[list[ReviewPerspective]] | None = None,
    ) -> ChainResult:
        """Run a serial chain of parallel rounds over grouped perspectives."""
        resolved_groups = groups or self._default_groups()
        rounds = [
            ReviewRound(
                [ReviewerAgent(perspective, self.model) for perspective in group],
                policy=self.policy,
                name=f"chain-round-{index}",
            )
            for index, group in enumerate(resolved_groups)
            if group
        ]
        return ReviewChain(rounds, policy=self.policy).run(target)

    def _reviewers(self) -> list[ReviewerAgent]:
        """Instantiate one reviewer agent per perspective."""
        return [ReviewerAgent(perspective, self.model) for perspective in self.perspectives]

    def _default_groups(self) -> list[list[ReviewPerspective]]:
        """Split the panel into two balanced parallel rounds."""
        midpoint = (len(self.perspectives) + 1) // 2
        first = self.perspectives[:midpoint]
        second = self.perspectives[midpoint:]
        return [group for group in (first, second) if group]


def _wrap_round(result: RoundResult) -> ChainResult:
    """Project a single round result into a chain result for a uniform return."""
    return ChainResult(
        approved=result.approved,
        rounds=[result],
        verdicts=result.verdicts,
        blockers=result.blockers,
        consensus_confidence=result.consensus_confidence,
    )
