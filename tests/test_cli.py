from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout

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


if __name__ == "__main__":
    unittest.main()
