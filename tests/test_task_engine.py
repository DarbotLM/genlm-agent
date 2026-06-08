from __future__ import annotations

import unittest

from glmagent.agent import ExecutionResult
from glmagent.task_engine import (
    GenerativeLayer,
    RepositorySnapshot,
    TaskEngineAgent,
    TaskEngineConfig,
    TaskSpec,
)


class CapturingTaskModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.seen_messages: list[list[dict[str, str]]] = []

    def query(self, messages):
        self.seen_messages.append(messages)
        if not self.outputs:
            msg = "No more task outputs configured"
            raise AssertionError(msg)
        return self.outputs.pop(0)


class FakeTaskEnvironment:
    def __init__(self, snapshots, responses):
        self.snapshots = list(snapshots)
        self.responses = list(responses)
        self.snapshot_index = 0
        self.tasks: list[TaskSpec] = []

    def snapshot(self):
        index = min(self.snapshot_index, len(self.snapshots) - 1)
        return self.snapshots[index]

    def execute_task(self, task, timeout=30.0):
        self.tasks.append(task)
        if self.snapshot_index < len(self.snapshots) - 1:
            self.snapshot_index += 1
        response = self.responses.pop(0)
        if isinstance(response, ExecutionResult):
            return response
        return ExecutionResult(command=task.to_action_string(), stdout=str(response))


class TaskEngineTests(unittest.TestCase):
    def test_task_spec_rejects_write_without_files(self):
        task = TaskSpec.from_dict(
            {
                "kind": "write",
                "title": "write changes",
                "scope_items": ["repo"],
            }
        )

        gaps = task.validate(TaskEngineConfig().policy)

        assert "write tasks must target at least one file" in gaps

    def test_task_engine_includes_snapshot_in_model_context(self):
        model = CapturingTaskModel(
            [
                {
                    "task": {
                        "kind": "inspect",
                        "title": "inspect repo",
                        "command": "git status",
                        "scope_items": ["repo"],
                        "evidence_required": "indicator",
                    }
                }
            ]
        )
        env = FakeTaskEnvironment(
            snapshots=[
                RepositorySnapshot(root="S:/repo", branch="main", dirty_files=["app.py"]),
                RepositorySnapshot(root="S:/repo", branch="main", dirty_files=["app.py"]),
            ],
            responses=["status: ok\nexit code: 0"],
        )
        agent = TaskEngineAgent(model=model, env=env, config=TaskEngineConfig(max_steps=1))

        agent.step()

        final_message = model.seen_messages[0][-1]["content"]
        assert "Repository snapshot:" in final_message
        assert "dirty_files: app.py" in final_message

    def test_task_engine_completes_only_after_verified_task_scope(self):
        model = CapturingTaskModel(
            [
                {
                    "task": {
                        "kind": "test",
                        "title": "run repo tests",
                        "rationale": "verify the patch",
                        "command": "python -m unittest",
                        "scope_items": ["repo"],
                        "evidence_required": "functional",
                        "completion_claim": "repo tests passed",
                    }
                },
                {
                    "task": {
                        "kind": "complete",
                        "title": "finish",
                        "scope_items": ["repo"],
                        "completion_claim": "repo tests passed",
                    }
                },
            ]
        )
        env = FakeTaskEnvironment(
            snapshots=[
                RepositorySnapshot(root="S:/repo", branch="main", dirty_files=["app.py"]),
                RepositorySnapshot(root="S:/repo", branch="main", changed_files=["app.py"]),
            ],
            responses=["repo tests passed\nall checks passed"],
        )
        agent = TaskEngineAgent(model=model, env=env, config=TaskEngineConfig(max_steps=2))
        agent.begin_session(
            "fix the repo",
            history=[
                {"role": "system", "content": agent.config.policy.render()},
                {"role": "user", "content": "fix the repo"},
            ],
        )
        agent._decision_tree.current_layer = GenerativeLayer.VERIFY
        agent._decision_tree.history = [GenerativeLayer.VERIFY]

        first = agent.step()
        second = agent.step()
        trajectory = [first, second]

        assert len(trajectory) == 2
        assert trajectory[0].verified
        assert trajectory[1].done
        assert trajectory[1].verified
        assert trajectory[0].layer_before == GenerativeLayer.VERIFY
        assert trajectory[0].layer_after == GenerativeLayer.COMPLETE
        assert trajectory[1].layer_after == GenerativeLayer.COMPLETE
        assert len(env.tasks) == 1

    def test_layer_contract_blocks_patch_in_triage(self):
        model = CapturingTaskModel(
            [
                {
                    "task": {
                        "kind": "patch",
                        "title": "patch app.py",
                        "files": ["app.py"],
                        "patch": "*** Begin Patch",
                        "scope_items": ["app.py"],
                    }
                }
            ]
        )
        env = FakeTaskEnvironment(
            snapshots=[RepositorySnapshot(root="S:/repo"), RepositorySnapshot(root="S:/repo")],
            responses=["unused"],
        )
        agent = TaskEngineAgent(model=model, env=env, config=TaskEngineConfig(max_steps=1))

        result = agent.step()

        assert not result.verified
        assert "Task kind 'patch' is not allowed in layer 'triage'" in result.verification_gaps
        assert env.tasks == []

    def test_invalid_patch_task_is_rejected_before_execution(self):
        model = CapturingTaskModel(
            [
                {
                    "task": {
                        "kind": "patch",
                        "title": "patch app.py",
                        "files": ["app.py"],
                        "scope_items": ["app.py"],
                    }
                }
            ]
        )
        env = FakeTaskEnvironment(
            snapshots=[RepositorySnapshot(root="S:/repo"), RepositorySnapshot(root="S:/repo")],
            responses=["unused"],
        )
        agent = TaskEngineAgent(model=model, env=env, config=TaskEngineConfig(max_steps=1))

        result = agent.step()

        assert not result.verified
        assert "Patch tasks must include a patch payload" in result.verification_gaps
        assert env.tasks == []


if __name__ == "__main__":
    unittest.main()
