"""Generative Layer Management and flashcard memory for task engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from glmagent.task_engine.spec import TaskKind


class GenerativeLayer(StrEnum):
    """Finite semantic layers for coding-agent task routing."""

    TRIAGE = "triage"
    LOCALIZE = "localize"
    EDIT = "edit"
    VERIFY = "verify"
    COMPLETE = "complete"


@dataclass
class SemanticDecisionTree:
    """Finite-state controller for narrow task-engine agents."""

    current_layer: GenerativeLayer = GenerativeLayer.TRIAGE
    history: list[GenerativeLayer] = field(default_factory=lambda: [GenerativeLayer.TRIAGE])

    def allowed_tasks(self) -> tuple[TaskKind, ...]:
        """Return allowed task kinds for the current layer."""
        layer_map = {
            GenerativeLayer.TRIAGE: (
                TaskKind.INSPECT,
                TaskKind.SEARCH,
                TaskKind.READ,
                TaskKind.TEST,
                TaskKind.VERIFY,
            ),
            GenerativeLayer.LOCALIZE: (
                TaskKind.SEARCH,
                TaskKind.READ,
                TaskKind.VERIFY,
            ),
            GenerativeLayer.EDIT: (
                TaskKind.READ,
                TaskKind.WRITE,
                TaskKind.PATCH,
                TaskKind.RUN,
            ),
            GenerativeLayer.VERIFY: (
                TaskKind.READ,
                TaskKind.RUN,
                TaskKind.TEST,
                TaskKind.VERIFY,
            ),
            GenerativeLayer.COMPLETE: (
                TaskKind.COMPLETE,
                TaskKind.READ,
                TaskKind.TEST,
                TaskKind.VERIFY,
            ),
        }
        return layer_map[self.current_layer]

    def validate(self, task_kind: TaskKind) -> list[str]:
        """Check whether a task kind is allowed in the current layer."""
        if task_kind in self.allowed_tasks():
            return []
        return [f"Task kind '{task_kind.value}' is not allowed in layer '{self.current_layer.value}'"]

    def advance(
        self,
        task_kind: TaskKind,
        *,
        verified: bool,
        done: bool,
    ) -> GenerativeLayer:
        """Advance the finite-state controller after a task result."""
        if done and verified:
            next_layer = GenerativeLayer.COMPLETE
        elif self.current_layer == GenerativeLayer.TRIAGE:
            next_layer = (
                GenerativeLayer.LOCALIZE
                if verified
                and task_kind in {TaskKind.INSPECT, TaskKind.SEARCH, TaskKind.READ, TaskKind.TEST, TaskKind.VERIFY}
                else GenerativeLayer.TRIAGE
            )
        elif self.current_layer == GenerativeLayer.LOCALIZE:
            next_layer = (
                GenerativeLayer.EDIT
                if verified and task_kind in {TaskKind.SEARCH, TaskKind.READ, TaskKind.VERIFY}
                else GenerativeLayer.TRIAGE
            )
        elif self.current_layer == GenerativeLayer.EDIT:
            next_layer = (
                GenerativeLayer.VERIFY
                if verified and task_kind in {TaskKind.WRITE, TaskKind.PATCH, TaskKind.RUN}
                else GenerativeLayer.LOCALIZE
            )
        elif self.current_layer == GenerativeLayer.VERIFY:
            next_layer = (
                GenerativeLayer.COMPLETE
                if verified and task_kind in {TaskKind.RUN, TaskKind.TEST, TaskKind.VERIFY}
                else GenerativeLayer.EDIT
            )
        else:
            next_layer = GenerativeLayer.COMPLETE if verified else GenerativeLayer.VERIFY

        self.current_layer = next_layer
        self.history.append(next_layer)
        return next_layer

    def render(self) -> str:
        """Render the layer controller into a compact prompt contract."""
        return "\n".join(
            [
                "Generative Layer Management:",
                f"current_layer: {self.current_layer.value}",
                "allowed_tasks: " + ", ".join(task.value for task in self.allowed_tasks()),
                "Progression heuristic: triage -> localize -> edit -> verify -> complete.",
                "If a task fails verification, regress to the previous problem-solving layer.",
            ]
        )


@dataclass
class AdaptiveFlashcard:
    """Compact verified memory card for small-model prompting."""

    prompt: str
    answer: str
    layer: GenerativeLayer
    scope_items: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    card_type: str = "fact"
    strength: float = 1.0
    hits: int = 1
    misses: int = 0
    last_step: int = 0


@dataclass
class AdaptiveFlashcardDeck:
    """Adaptive memory tuned for constrained coding agents."""

    cards: list[AdaptiveFlashcard] = field(default_factory=list)

    def remember_verified(self, result: Any, layer: GenerativeLayer, step_index: int) -> None:
        """Store or reinforce a verified fact card."""
        scope_items = list(getattr(getattr(result, "intent", None), "scope_items", []) or ["global"])
        evidence_refs = [entry.reference for entry in getattr(result, "evidence", [])]
        task = getattr(result, "task", None)
        answer = getattr(task, "title", "") or getattr(result, "thought", "Verified result")

        for scope in scope_items:
            prompt = f"What is the latest verified fact about {scope}?"
            self._upsert(
                prompt=prompt,
                answer=answer,
                layer=layer,
                scope_items=[scope],
                evidence_refs=evidence_refs,
                card_type="fact",
                success=True,
                step_index=step_index,
            )

    def remember_gap(self, result: Any, layer: GenerativeLayer, step_index: int) -> None:
        """Store or reinforce an anti-pattern card from a failed step."""
        gaps = list(getattr(result, "verification_gaps", []))
        if not gaps:
            return
        prompt = f"What failure pattern should be avoided in {layer.value}?"
        answer = gaps[0]
        scope_items = list(getattr(getattr(result, "intent", None), "scope_items", []) or [])
        self._upsert(
            prompt=prompt,
            answer=answer,
            layer=layer,
            scope_items=scope_items,
            evidence_refs=[],
            card_type="anti-pattern",
            success=False,
            step_index=step_index,
        )

    def select(
        self,
        layer: GenerativeLayer,
        limit: int = 6,
    ) -> list[AdaptiveFlashcard]:
        """Return the most relevant flashcards for the current layer."""
        ranked = sorted(
            self.cards,
            key=lambda card: (
                card.layer == layer,
                card.card_type == "fact",
                card.strength,
                card.last_step,
            ),
            reverse=True,
        )
        return ranked[:limit]

    def render(self, layer: GenerativeLayer, limit: int = 6) -> str:
        """Render the selected flashcards into prompt-friendly text."""
        selected = self.select(layer, limit=limit)
        if not selected:
            return "Adaptive flashcards: (none)"

        lines = ["Adaptive flashcards:"]
        for index, card in enumerate(selected, start=1):
            lines.append(
                f"{index}. [{card.card_type}|{card.layer.value}|strength={card.strength:.1f}] "
                f"Q: {card.prompt} A: {card.answer}"
            )
        return "\n".join(lines)

    def _upsert(
        self,
        *,
        prompt: str,
        answer: str,
        layer: GenerativeLayer,
        scope_items: list[str],
        evidence_refs: list[str],
        card_type: str,
        success: bool,
        step_index: int,
    ) -> None:
        for card in self.cards:
            if card.prompt == prompt and card.card_type == card_type:
                card.answer = answer
                card.layer = layer
                card.scope_items = list(scope_items)
                card.evidence_refs = list(evidence_refs)
                card.last_step = step_index
                if success:
                    card.hits += 1
                    card.strength += 1.0
                else:
                    card.misses += 1
                    card.strength = max(0.5, card.strength - 0.25)
                return

        self.cards.append(
            AdaptiveFlashcard(
                prompt=prompt,
                answer=answer,
                layer=layer,
                scope_items=list(scope_items),
                evidence_refs=list(evidence_refs),
                card_type=card_type,
                strength=1.0 if success else 0.75,
                hits=1 if success else 0,
                misses=0 if success else 1,
                last_step=step_index,
            )
        )
