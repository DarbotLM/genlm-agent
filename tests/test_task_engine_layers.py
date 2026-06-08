from __future__ import annotations

import unittest
from types import SimpleNamespace

from glmagent.task_engine import (
    AdaptiveFlashcardDeck,
    GenerativeLayer,
    SemanticDecisionTree,
    TaskKind,
)


class TaskEngineLayerTests(unittest.TestCase):
    def test_semantic_tree_enforces_and_advances_layers(self):
        tree = SemanticDecisionTree()

        assert tree.current_layer == GenerativeLayer.TRIAGE
        assert tree.validate(TaskKind.PATCH) == ["Task kind 'patch' is not allowed in layer 'triage'"]

        tree.advance(TaskKind.SEARCH, verified=True, done=False)
        assert tree.current_layer == GenerativeLayer.LOCALIZE

        tree.advance(TaskKind.READ, verified=True, done=False)
        assert tree.current_layer == GenerativeLayer.EDIT

        tree.advance(TaskKind.PATCH, verified=True, done=False)
        assert tree.current_layer == GenerativeLayer.VERIFY

        tree.advance(TaskKind.TEST, verified=True, done=False)
        assert tree.current_layer == GenerativeLayer.COMPLETE

    def test_flashcards_store_verified_facts_and_failures(self):
        deck = AdaptiveFlashcardDeck()
        verified_result = SimpleNamespace(
            intent=SimpleNamespace(scope_items=["repo"]),
            evidence=[SimpleNamespace(reference="ref://repo/test")],
            task=SimpleNamespace(title="repo tests passed"),
            thought="repo tests passed",
            verification_gaps=[],
        )
        failed_result = SimpleNamespace(
            intent=SimpleNamespace(scope_items=["repo"]),
            evidence=[],
            task=SimpleNamespace(title="bad patch"),
            thought="bad patch",
            verification_gaps=["Patch tasks must include a patch payload"],
        )

        deck.remember_verified(verified_result, GenerativeLayer.VERIFY, step_index=1)
        deck.remember_gap(failed_result, GenerativeLayer.EDIT, step_index=2)

        rendered = deck.render(GenerativeLayer.VERIFY, limit=4)

        assert "latest verified fact about repo" in rendered
        assert "Patch tasks must include a patch payload" in rendered


if __name__ == "__main__":
    unittest.main()
