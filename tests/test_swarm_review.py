from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from glmagent.swarm import (
    HeuristicReviewModel,
    ObservationalReviewer,
    ReviewMode,
    ReviewPolicy,
    ReviewRound,
    ReviewSeverity,
    ReviewTarget,
    ReviewerAgent,
    default_review_panel,
    evidence_strength_auditor,
    freshness_auditor,
    red_team,
    scope_coverage_auditor,
    security_reviewer,
    skeptic,
)
from glmagent.verification import Claim, Evidence, EvidenceLevel


def _evidence(
    *,
    scope_item: str = "repo",
    level: EvidenceLevel = EvidenceLevel.FUNCTIONAL,
    age_seconds: float | None = 0.0,
) -> Evidence:
    now = datetime.now(UTC)
    data_timestamp = None if age_seconds is None else now - timedelta(seconds=age_seconds)
    return Evidence(
        source="test",
        timestamp=now,
        data_timestamp=data_timestamp,
        level=level,
        content="observation content",
        scope_item=scope_item,
    )


def _clean_target() -> ReviewTarget:
    evidence = _evidence(level=EvidenceLevel.FUNCTIONAL, age_seconds=0.0)
    claim = Claim(
        subject="repo tests passed",
        predicate="status",
        evidence_refs=[evidence.reference],
        confidence=0.9,
    )
    return ReviewTarget(
        subject="verify repo",
        observation="repo tests passed\nall checks passed",
        action="pytest -q",
        claims=[claim],
        evidence=[evidence],
        scope_items=["repo"],
        recency_window_seconds=600,
    )


class ReviewSeverityTests(unittest.TestCase):
    def test_severity_orders_by_declaration(self):
        assert ReviewSeverity.BLOCKER > ReviewSeverity.MAJOR
        assert ReviewSeverity.MAJOR > ReviewSeverity.MINOR
        assert ReviewSeverity.MINOR > ReviewSeverity.INFO
        assert max(
            [ReviewSeverity.INFO, ReviewSeverity.BLOCKER, ReviewSeverity.MINOR]
        ) == ReviewSeverity.BLOCKER


class HeuristicLensTests(unittest.TestCase):
    def test_skeptic_blocks_ungrounded_claim(self):
        ungrounded = Claim(subject="it works", predicate="status", evidence_refs=[])
        target = ReviewTarget(subject="x", claims=[ungrounded], evidence=[_evidence()])
        verdict = ReviewerAgent(skeptic(), HeuristicReviewModel()).review(target)

        assert not verdict.approved
        assert verdict.has_blockers
        assert verdict.perspective == "skeptic"

    def test_freshness_blocks_stale_evidence(self):
        target = ReviewTarget(
            subject="x",
            evidence=[_evidence(age_seconds=1000.0)],
            recency_window_seconds=300,
        )
        verdict = ReviewerAgent(freshness_auditor(), HeuristicReviewModel()).review(target)

        assert not verdict.approved
        assert verdict.has_blockers

    def test_freshness_flags_missing_timestamp_as_major(self):
        target = ReviewTarget(subject="x", evidence=[_evidence(age_seconds=None)])
        verdict = ReviewerAgent(freshness_auditor(), HeuristicReviewModel()).review(target)

        assert not verdict.approved
        assert verdict.max_severity == ReviewSeverity.MAJOR

    def test_strength_auditor_rejects_weak_evidence(self):
        target = ReviewTarget(subject="x", evidence=[_evidence(level=EvidenceLevel.PROXY)])
        verdict = ReviewerAgent(evidence_strength_auditor(), HeuristicReviewModel()).review(target)

        assert not verdict.approved
        assert any("below required" in finding.summary for finding in verdict.findings)

    def test_coverage_auditor_flags_uncovered_scope(self):
        target = ReviewTarget(
            subject="x",
            evidence=[_evidence(scope_item="repo")],
            scope_items=["repo", "docs"],
        )
        verdict = ReviewerAgent(scope_coverage_auditor(), HeuristicReviewModel()).review(target)

        assert not verdict.approved
        assert any(finding.scope_item == "docs" for finding in verdict.findings)

    def test_security_reviewer_blocks_destructive_action(self):
        target = ReviewTarget(subject="x", action="rm -rf /tmp/data", evidence=[_evidence()])
        verdict = ReviewerAgent(security_reviewer(), HeuristicReviewModel()).review(target)

        assert not verdict.approved
        assert verdict.has_blockers

    def test_red_team_always_challenges(self):
        verdict = ReviewerAgent(red_team(), HeuristicReviewModel()).review(_clean_target())

        assert verdict.approved
        assert verdict.findings  # adversarial reviewer always raises at least an info challenge


class ReviewRoundTests(unittest.TestCase):
    def test_clean_target_passes_full_panel(self):
        reviewers = [ReviewerAgent(p, HeuristicReviewModel()) for p in default_review_panel()]
        result = ReviewRound(reviewers).run(_clean_target())

        assert result.approved
        assert len(result.verdicts) == len(default_review_panel())
        assert result.blockers == []

    def test_round_reports_dissent_and_blockers(self):
        target = ReviewTarget(
            subject="bad",
            action="sudo rm -rf /",
            claims=[Claim(subject="done", predicate="status", evidence_refs=[])],
            evidence=[_evidence(level=EvidenceLevel.PROXY, age_seconds=9000.0)],
            scope_items=["repo", "docs"],
            recency_window_seconds=300,
        )
        reviewers = [ReviewerAgent(p, HeuristicReviewModel()) for p in default_review_panel()]
        result = ReviewRound(reviewers).run(target)

        assert not result.approved
        assert result.blockers
        assert "skeptic" in result.dissenting
        assert "security-reviewer" in result.dissenting

    def test_parallel_and_sequential_dispatch_match(self):
        reviewers = [ReviewerAgent(p, HeuristicReviewModel()) for p in default_review_panel()]
        target = _clean_target()

        parallel = ReviewRound(reviewers, policy=ReviewPolicy(max_workers=8)).run(target)
        sequential = ReviewRound(reviewers, policy=ReviewPolicy(max_workers=1)).run(target)

        parallel_summary = [(v.perspective, v.approved, v.max_severity) for v in parallel.verdicts]
        sequential_summary = [(v.perspective, v.approved, v.max_severity) for v in sequential.verdicts]
        assert parallel_summary == sequential_summary


class ReviewChainTests(unittest.TestCase):
    def test_series_threads_prior_verdicts_to_red_team(self):
        reviewer = ObservationalReviewer.with_default_panel()
        result = reviewer.review_series(_clean_target())

        red_team_verdict = result.verdicts[-1]
        assert red_team_verdict.perspective == "red-team"
        assert any("unanimously" in finding.summary for finding in red_team_verdict.findings)

    def test_chain_runs_series_of_parallel_rounds(self):
        reviewer = ObservationalReviewer.with_default_panel()
        result = reviewer.review_chain(_clean_target())

        assert result.approved
        assert len(result.rounds) == 2
        assert len(result.verdicts) == len(default_review_panel())

    def test_review_mode_dispatch_returns_chain_result(self):
        reviewer = ObservationalReviewer.with_default_panel()

        parallel = reviewer.review(_clean_target(), mode=ReviewMode.PARALLEL)
        assert parallel.approved
        assert len(parallel.rounds) == 1

        payload = parallel.to_dict()
        assert payload["approved"] is True
        assert payload["rounds"][0]["verdicts"][0]["lens"]

    def test_stop_on_blocker_halts_chain(self):
        policy = ReviewPolicy(stop_on_blocker=True)
        reviewer = ObservationalReviewer(
            perspectives=[security_reviewer(), skeptic()],
            policy=policy,
        )
        target = ReviewTarget(subject="x", action="mkfs /dev/sda", evidence=[_evidence()])
        result = reviewer.review_series(target)

        assert not result.approved
        assert result.stopped_early
        assert len(result.rounds) == 1


class ReviewModelAdapterTests(unittest.TestCase):
    def test_language_model_payload_is_normalized(self):
        class StubReviewModel:
            def __init__(self):
                self.requests = []

            def review(self, request):
                self.requests.append(request)
                return {
                    "approved": False,
                    "confidence": 0.33,
                    "rationale": "looks wrong",
                    "findings": [{"severity": "blocker", "summary": "regression risk"}],
                }

        stub = StubReviewModel()
        verdict = ReviewerAgent(skeptic(), stub).review(_clean_target())

        assert not verdict.approved
        assert verdict.confidence == 0.33
        assert verdict.findings[0].severity == ReviewSeverity.BLOCKER
        assert verdict.findings[0].perspective == "skeptic"
        assert stub.requests[0].messages[0]["role"] == "system"

    def test_blocker_finding_overrides_model_approval(self):
        class OverconfidentModel:
            def review(self, request):
                return {
                    "approved": True,
                    "confidence": 0.99,
                    "findings": [{"severity": "blocker", "summary": "data loss"}],
                }

        verdict = ReviewerAgent(security_reviewer(), OverconfidentModel()).review(_clean_target())
        assert not verdict.approved


class ReviewTargetTests(unittest.TestCase):
    def test_from_observation_extracts_evidence(self):
        target = ReviewTarget.from_observation(
            "repo tests passed at 2999-01-01T00:00:00Z",
            scope_items=["repo"],
        )
        assert target.evidence
        assert target.scope_items == ["repo"]

    def test_from_step_reads_intent_scope(self):
        intent = SimpleNamespace(
            goal="verify repo",
            scope_items=["repo"],
            recency_window_seconds=600,
        )
        step = SimpleNamespace(
            observation="all checks passed",
            action="pytest -q",
            claims=[],
            evidence=[_evidence()],
            intent=intent,
        )
        target = ReviewTarget.from_step(step)

        assert target.subject == "verify repo"
        assert target.scope_items == ["repo"]
        assert target.recency_window_seconds == 600


if __name__ == "__main__":
    unittest.main()
