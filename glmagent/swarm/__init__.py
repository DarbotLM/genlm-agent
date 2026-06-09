"""Observational swarm review for GenLM-Agent.

This package runs a swarm of reviewer agents, each with a uniquely challenging
perspective, against an observation or agent step. Reviewers can run in a single
parallel round, in a serial chain, or as a serial chain of parallel rounds.
"""

from glmagent.swarm.chain import (
    ChainResult,
    ObservationalReviewer,
    ReviewChain,
    ReviewMode,
)
from glmagent.swarm.perspective import (
    BUILTIN_PERSPECTIVES,
    ReviewLens,
    ReviewPerspective,
    default_review_panel,
    evidence_strength_auditor,
    freshness_auditor,
    red_team,
    scope_coverage_auditor,
    security_reviewer,
    skeptic,
)
from glmagent.swarm.reviewer import (
    HeuristicReviewModel,
    ReviewerAgent,
    ReviewModel,
    ReviewRequest,
)
from glmagent.swarm.round import ReviewPolicy, ReviewRound, RoundResult
from glmagent.swarm.verdict import (
    ReviewFinding,
    ReviewSeverity,
    ReviewTarget,
    ReviewVerdict,
)

__all__ = [
    "BUILTIN_PERSPECTIVES",
    "ChainResult",
    "HeuristicReviewModel",
    "ObservationalReviewer",
    "ReviewChain",
    "ReviewFinding",
    "ReviewLens",
    "ReviewMode",
    "ReviewModel",
    "ReviewPerspective",
    "ReviewPolicy",
    "ReviewRequest",
    "ReviewRound",
    "ReviewSeverity",
    "ReviewTarget",
    "ReviewVerdict",
    "ReviewerAgent",
    "RoundResult",
    "default_review_panel",
    "evidence_strength_auditor",
    "freshness_auditor",
    "red_team",
    "scope_coverage_auditor",
    "security_reviewer",
    "skeptic",
]
