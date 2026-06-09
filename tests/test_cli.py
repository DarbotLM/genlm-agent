from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime

from glmagent.run.run import main


class CliTests(unittest.TestCase):
    def test_verify_observation_command_outputs_json(self):
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            exit_code = main(
                [
                    "verify-observation",
                    "--text",
                    "repo tests passed at 2026-03-09T12:00:00Z",
                    "--scope",
                    "repo",
                ]
            )

        payload = json.loads(buffer.getvalue())
        assert exit_code == 0
        assert payload[0]["scope_item"] == "repo"
        assert payload[0]["level"] == "functional"


class ReviewCliTests(unittest.TestCase):
    def test_review_observation_command_outputs_json(self):
        buffer = io.StringIO()
        fresh = (
            datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        )

        with redirect_stdout(buffer):
            exit_code = main(
                [
                    "review-observation",
                    "--text",
                    f"repo tests passed at {fresh}\nall checks passed",
                    "--scope",
                    "repo",
                    "--mode",
                    "parallel",
                ]
            )

        payload = json.loads(buffer.getvalue())
        assert exit_code == 0
        assert payload["approved"] is True
        assert payload["rounds"]
        assert payload["rounds"][0]["verdicts"]

    def test_review_observation_flags_destructive_action(self):
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            exit_code = main(
                [
                    "review-observation",
                    "--text",
                    "cleaning up workspace",
                    "--action",
                    "rm -rf /tmp/data",
                    "--perspective",
                    "security-reviewer",
                ]
            )

        payload = json.loads(buffer.getvalue())
        assert exit_code == 1
        assert payload["approved"] is False
        assert payload["blockers"]


if __name__ == "__main__":
    unittest.main()

