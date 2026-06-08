from __future__ import annotations

import unittest
from datetime import UTC

from glmagent.verification import (
    EvidenceLevel,
    classify_evidence_level,
    extract_evidence,
    extract_latest_timestamp,
)


class EvidenceTests(unittest.TestCase):
    def test_offset_timestamps_convert_to_utc(self):
        parsed = extract_latest_timestamp("ts=2026-01-09T12:00:00-0800")

        assert parsed is not None
        assert parsed.tzinfo == UTC
        assert parsed.isoformat() == "2026-01-09T20:00:00+00:00"

    def test_syslog_timestamps_are_parsed(self):
        parsed = extract_latest_timestamp("Jan  9 12:34:56 service started")

        assert parsed is not None
        assert parsed.hour == 12
        assert parsed.minute == 34
        assert parsed.second == 56

    def test_classifies_functional_and_comparative_evidence(self):
        functional = classify_evidence_level("repo tests passed at 2026-03-09T12:00:00Z\nall checks passed")
        comparative = classify_evidence_level("diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py")

        assert functional == EvidenceLevel.FUNCTIONAL
        assert comparative == EvidenceLevel.COMPARATIVE

    def test_extracts_multiple_scope_items_when_observation_mentions_them(self):
        evidence = extract_evidence(
            observation="api tests passed and worker tests passed at 2026-03-09T12:00:00Z",
            source="test",
            scope_items=["api", "worker"],
        )

        assert {item.scope_item for item in evidence} == {"api", "worker"}

    def test_functional_evidence_without_embedded_timestamp_is_fresh(self):
        evidence = extract_evidence(
            observation="repo tests passed\nall checks passed",
            source="test",
            scope_items=["repo"],
        )

        assert evidence[0].data_timestamp is not None
        assert evidence[0].level == EvidenceLevel.FUNCTIONAL


if __name__ == "__main__":
    unittest.main()
