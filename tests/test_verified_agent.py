from __future__ import annotations

import unittest
from datetime import UTC, datetime

from glmagent.agent import AgentConfig, ExecutionResult, VerifiedAgent
from glmagent.verification import EvidenceLevel


class QueueModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def query(self, messages):
        if not self.outputs:
            msg = "No more model outputs configured"
            raise AssertionError(msg)
        return self.outputs.pop(0)


class RecordingEnvironment:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[str] = []

    def execute(self, command, timeout=30.0):
        self.calls.append(command)
        if not self.responses:
            return ExecutionResult(command=command, stdout="")
        response = self.responses.pop(0)
        if isinstance(response, ExecutionResult):
            return response
        return ExecutionResult(command=command, stdout=str(response))


def fresh_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class VerifiedAgentTests(unittest.TestCase):
    def test_done_without_action_does_not_execute_environment(self):
        agent = VerifiedAgent(
            model=QueueModel(
                [
                    {
                        "thought": "Goal: finish safely",
                        "intent": {"goal": "finish safely", "evidence_required": "proxy"},
                        "action": "",
                        "done": True,
                    }
                ]
            ),
            env=RecordingEnvironment([]),
        )

        result = agent.step()

        assert result.done
        assert not result.verified
        assert "Completion requested" in result.verification_gaps[0]
        assert agent.env.calls == []

    def test_run_only_stops_on_verified_terminal_step(self):
        agent = VerifiedAgent(
            model=QueueModel(
                [
                    {
                        "thought": "Need a plan first",
                        "action": "",
                        "done": True,
                    },
                    {
                        "thought": "status: repo tests passed",
                        "intent": {
                            "goal": "verify the repo",
                            "evidence_required": "functional",
                            "scope_items": ["repo"],
                            "recency_window_seconds": 600,
                        },
                        "action": "pytest -q",
                        "done": True,
                    },
                ]
            ),
            env=RecordingEnvironment([f"repo tests passed at {fresh_timestamp()}\nall checks passed"]),
            config=AgentConfig(max_steps=2),
        )

        trajectory = agent.run("verify the repo")

        assert len(trajectory) == 2
        assert not trajectory[0].verified
        assert trajectory[1].verified
        assert agent.env.calls == ["pytest -q"]

    def test_history_records_action(self):
        agent = VerifiedAgent(
            model=QueueModel(
                [
                    {
                        "thought": "status: repo healthy",
                        "intent": {
                            "goal": "check the repo",
                            "evidence_required": "indicator",
                        },
                        "action": "ls",
                        "done": True,
                    }
                ]
            ),
            env=RecordingEnvironment(["status: running\nexit code: 0"]),
        )

        agent.run("check")

        assistant_messages = [entry["content"] for entry in agent.history if entry["role"] == "assistant"]
        assert any("Action: ls" in message for message in assistant_messages)

    def test_verified_items_reset_between_runs(self):
        agent = VerifiedAgent(
            model=QueueModel(
                [
                    {
                        "thought": "status: repo tests passed",
                        "intent": {
                            "goal": "verify the repo",
                            "evidence_required": "functional",
                            "scope_items": ["repo"],
                            "recency_window_seconds": 600,
                        },
                        "action": "pytest -q",
                        "done": True,
                    },
                    {
                        "thought": "Goal: stop",
                        "intent": {"goal": "stop", "evidence_required": "proxy"},
                        "action": "",
                        "done": True,
                    },
                ]
            ),
            env=RecordingEnvironment([f"repo tests passed at {fresh_timestamp()}\nall checks passed"]),
            config=AgentConfig(max_steps=1),
        )

        agent.set_scope(["repo"])
        agent.run("first")
        assert agent._verified_items == {"repo"}

        agent.config.max_steps = 1
        with self.assertLogs("glmagent.agent.base", level="WARNING"):
            agent.run("second")
        assert agent._verified_items == set()

    def test_risky_commands_are_detected(self):
        agent = VerifiedAgent(
            model=QueueModel([]),
            env=RecordingEnvironment([]),
        )

        review = agent._review_action("git reset --hard HEAD")

        assert review["risk"] == "critical"
        assert not review["proceed"]

    def test_claims_ground_to_functional_evidence(self):
        agent = VerifiedAgent(
            model=QueueModel(
                [
                    {
                        "thought": "status: repo tests passed",
                        "intent": {
                            "goal": "verify tests",
                            "evidence_required": EvidenceLevel.FUNCTIONAL,
                            "scope_items": ["repo"],
                            "recency_window_seconds": 600,
                        },
                        "action": "pytest -q",
                        "done": True,
                    }
                ]
            ),
            env=RecordingEnvironment([f"repo tests passed at {fresh_timestamp()}\nall checks passed"]),
        )

        result = agent.step()

        assert result.claims
        assert result.claims[0].is_grounded


if __name__ == "__main__":
    unittest.main()
