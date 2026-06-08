"""Base agent with verification built into the execution loop.

This is the foundation of GenLM-Agent's verification-first architecture.
Every step passes through four gates before the agent can proceed:

1. Intent Gate - declare what success looks like before acting
2. Action Gate - assess risk and prerequisites before executing
3. Observation Gate - verify evidence freshness and depth after executing
4. Verdict Gate - validate that all claims are grounded in evidence

The verification framework is not a wrapper or afterthought - it is
structural. The agent loop cannot skip these gates.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from glmagent.runtime import RuntimeAgentBase, SessionEventType, SessionPresenter
from glmagent.verification.evidence import (
    Claim,
    Evidence,
    EvidenceLevel,
    SuccessCriteria,
    extract_evidence,
)

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Structured output from an environment action."""

    command: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    cwd: str | None = None
    timed_out: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def combined_output(self) -> str:
        """Render the execution result into a text observation."""
        parts: list[str] = []
        if self.stdout:
            parts.append(self.stdout.rstrip())
        if self.stderr:
            parts.append(f"STDERR:\n{self.stderr.rstrip()}")
        parts.append(f"EXIT CODE: {self.exit_code}")
        if self.cwd:
            parts.append(f"CWD: {self.cwd}")
        if self.timed_out:
            parts.append("TIMED OUT: true")
        return "\n".join(parts)


class Environment(Protocol):
    """Protocol for execution environments."""

    def execute(
        self,
        command: str,
        timeout: float = 30.0,
    ) -> ExecutionResult | str: ...


class Model(Protocol):
    """Protocol for language model backends."""

    def query(self, messages: list[dict[str, Any]]) -> dict[str, Any]: ...


@dataclass
class StepResult:
    """Result of a single agent step, including verification metadata."""

    thought: str = ""
    action: str = ""
    observation: str = ""
    execution_time: float = 0.0
    done: bool = False

    # Verification metadata - populated by the gates
    intent: SuccessCriteria | None = None
    action_review: dict[str, Any] = field(default_factory=dict)
    execution: ExecutionResult | None = None
    evidence: list[Evidence] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    verification_gaps: list[str] = field(default_factory=list)
    verified: bool = False


@dataclass
class AgentConfig:
    """Agent configuration with verification settings."""

    max_steps: int = 30
    execution_timeout_seconds: float = 30.0
    max_staleness_seconds: int = 300
    min_evidence_level: EvidenceLevel = EvidenceLevel.BEHAVIORAL
    require_scope_completion: bool = True
    require_intent: bool = True
    block_on_critical_risk: bool = True


class VerifiedAgent(RuntimeAgentBase):
    """Agent with verification gates built into the execution loop.

    Unlike frameworks where verification is an optional hook, this agent
    structurally cannot skip verification. The step() method enforces all
    four gates and `run()` only accepts a terminal result once it is verified.
    """

    def __init__(
        self,
        model: Model,
        env: Environment,
        config: AgentConfig | None = None,
        presenter: SessionPresenter | None = None,
    ):
        super().__init__(presenter=presenter)
        self.model = model
        self.env = env
        self.config = config or AgentConfig()

    agent_type = "verified"

    def step(self) -> StepResult:
        """Execute one step with all four verification gates."""
        result = StepResult()
        model_output = self._normalize_model_output(self.model.query(self.history))

        # Gate 1: Intent
        result.thought = model_output["thought"]
        result.action = model_output["action"]
        result.done = model_output["done"]
        result.intent = self._extract_intent(model_output)
        self._merge_scope_from_intent(result.intent)

        if self.config.require_intent and result.intent is None:
            result.observation = "INTENT REQUIRED: declare success criteria before acting."
            result.verification_gaps.append("No success criteria declared before action")
            self._record_step(result)
            return result

        # Gate 2: Action review
        result.action_review = self._review_action(result.action)
        if result.action and not result.action_review.get("proceed", True):
            result.observation = "ACTION BLOCKED: " + ", ".join(result.action_review.get("concerns", []))
            self._record_step(result)
            return result

        if not result.action:
            if result.done:
                result.observation = "COMPLETION REQUESTED WITHOUT ACTION: provide fresh evidence before stopping."
                result.verification_gaps.append("Completion requested without executing a verifiable action")
            else:
                result.observation = "NO ACTION PROPOSED: planning is allowed, but no evidence was gathered."
                result.verification_gaps.append("No action proposed")
            self._record_step(result)
            return result

        # Execute
        t0 = time.perf_counter()
        execution = self._normalize_execution_result(
            command=result.action,
            raw=self.env.execute(result.action, timeout=self.config.execution_timeout_seconds),
        )
        result.execution_time = time.perf_counter() - t0
        result.execution = execution
        result.observation = execution.combined_output

        # Gate 3: Observation verification
        result.evidence = self._collect_evidence(result.observation)
        result.verification_gaps.extend(self._check_evidence_quality(result.evidence, result.intent))

        # Gate 4: Verdict validation
        result.claims = self._extract_claims(result.thought, result.evidence)
        ungrounded = [claim for claim in result.claims if not claim.is_grounded]
        if ungrounded:
            result.verification_gaps.append("Ungrounded claims: " + ", ".join(claim.subject for claim in ungrounded))

        self._mark_verified_scope_items(result.evidence, result.intent)

        if result.verification_gaps:
            result.observation += "\nVERIFICATION GAPS: " + "; ".join(result.verification_gaps)

        result.verified = not result.verification_gaps
        self._record_step(result)
        return result

    def run(self, problem: str) -> list[StepResult]:
        """Run the agent until it produces a verified terminal state."""
        self.begin_session(problem, [{"role": "user", "content": problem}])

        for _ in range(self.config.max_steps):
            result = self.step()
            if result.done and result.verified:
                break

        self.finish_session()

        if self._scope_items:
            unverified = [item for item in self._scope_items if item not in self._verified_items]
            if unverified:
                logger.warning(
                    "Run complete with unverified scope items: %s (coverage: %d/%d)",
                    unverified,
                    len(self._verified_items),
                    len(self._scope_items),
                )
        return self.trajectory

    def _normalize_model_output(self, raw: Any) -> dict[str, Any]:
        """Validate the model response shape early."""
        if not isinstance(raw, dict):
            msg = f"Model.query() must return a dict, got {type(raw).__name__}"
            raise TypeError(msg)

        return {
            "thought": str(raw.get("thought", "")).strip(),
            "action": str(raw.get("action", "")).strip(),
            "done": bool(raw.get("done", False)),
            "intent": raw.get("intent") or raw.get("success_criteria"),
        }

    def _normalize_execution_result(
        self,
        command: str,
        raw: ExecutionResult | str,
    ) -> ExecutionResult:
        """Accept both structured and legacy string environment outputs."""
        if isinstance(raw, ExecutionResult):
            if not raw.command:
                raw.command = command
            return raw

        return ExecutionResult(command=command, stdout=str(raw))

    def _extract_intent(self, model_output: dict[str, Any]) -> SuccessCriteria | None:
        """Extract structured success criteria from model output or thought text."""
        intent_payload = model_output.get("intent")
        if isinstance(intent_payload, dict):
            goal = str(intent_payload.get("goal", "")).strip()
            if not goal:
                return None
            evidence_required = self._coerce_evidence_level(
                intent_payload.get("evidence_required"),
                default=self.config.min_evidence_level,
            )
            recency_window = intent_payload.get("recency_window_seconds")
            if recency_window is None:
                recency_window = self.config.max_staleness_seconds
            scope_items = self._normalize_scope_items(intent_payload.get("scope_items", []))
            failure_indicators = [
                str(item).strip() for item in intent_payload.get("failure_indicators", []) if str(item).strip()
            ]
            return SuccessCriteria(
                goal=goal,
                evidence_required=evidence_required,
                recency_window_seconds=int(recency_window) if recency_window is not None else None,
                scope_items=scope_items,
                failure_indicators=failure_indicators,
            )

        thought = model_output["thought"]
        if not thought:
            return None

        goal_match = re.search(r"\bgoal\s*:\s*(.+)", thought, re.IGNORECASE)
        if not goal_match:
            return None

        evidence_match = re.search(
            r"\bevidence\s*:\s*(proxy|indicator|behavioral|functional|comparative)",
            thought,
            re.IGNORECASE,
        )
        scope_match = re.search(r"\bscope\s*:\s*(.+)", thought, re.IGNORECASE)
        recency_match = re.search(r"\brecency\s*:\s*(\d+)", thought, re.IGNORECASE)
        failures_match = re.search(r"\bfailure\s*:\s*(.+)", thought, re.IGNORECASE)

        return SuccessCriteria(
            goal=goal_match.group(1).strip(),
            evidence_required=self._coerce_evidence_level(
                evidence_match.group(1) if evidence_match else None,
                default=self.config.min_evidence_level,
            ),
            recency_window_seconds=(
                int(recency_match.group(1)) if recency_match else self.config.max_staleness_seconds
            ),
            scope_items=(self._normalize_scope_items(scope_match.group(1).split(",")) if scope_match else []),
            failure_indicators=(
                [item.strip() for item in failures_match.group(1).split(",") if item.strip()] if failures_match else []
            ),
        )

    def _merge_scope_from_intent(self, intent: SuccessCriteria | None) -> None:
        """Adopt intent scope items as the active verification scope."""
        if intent is None or not intent.scope_items:
            return

        intent_scope = self._normalize_scope_items(intent.scope_items)
        if intent_scope != self._scope_items:
            self._scope_items = intent_scope
            self._verified_items.clear()

    def _review_action(self, action: str) -> dict[str, Any]:
        """Assess risk of a proposed action."""
        if not action:
            return {"proceed": True, "concerns": [], "risk": "none"}

        concerns: list[str] = []
        risk = "low"
        risk_patterns = [
            (
                r"\bssh\b.*\b(?:rm|systemctl\s+(?:stop|restart)|reboot|shutdown)\b",
                "critical",
                "Remote destructive command",
            ),
            (r"\bsudo\s+rm\b", "critical", "Privileged deletion"),
            (r"\brm\s+-[^\n]*r[^\n]*f\b", "critical", "Recursive forced deletion"),
            (r"\bgit\s+reset\s+--hard\b", "critical", "Git hard reset"),
            (r"\bgit\s+clean\b.*\b-f", "critical", "Git clean"),
            (r"\bdocker\s+rm\b.*\b-f\b", "high", "Forced container removal"),
            (r"\bscp\b.*\b/usr/bin/\b", "high", "Binary deployment"),
            (r"\bgit\s+push\b", "medium", "Git push"),
            (r"\bsystemctl\s+restart\b", "medium", "Service restart"),
        ]

        risk_order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        for pattern, level, description in risk_patterns:
            if re.search(pattern, action, re.IGNORECASE):
                concerns.append(f"[{level.upper()}] {description}")
                if risk_order[level] > risk_order[risk]:
                    risk = level

        proceed = not (self.config.block_on_critical_risk and risk == "critical")
        return {"proceed": proceed, "concerns": concerns, "risk": risk}

    def _collect_evidence(self, observation: str) -> list[Evidence]:
        """Extract structured evidence from an observation."""
        scope_items = self._scope_items or ["unknown"]
        return extract_evidence(
            observation,
            source="command_output",
            scope_items=scope_items,
        )

    def _check_evidence_quality(
        self,
        evidence: list[Evidence],
        intent: SuccessCriteria | None,
    ) -> list[str]:
        """Check evidence meets minimum quality requirements."""
        if not evidence:
            return ["No evidence collected from observation"]

        gaps: list[str] = []
        required_level = intent.evidence_required if intent is not None else self.config.min_evidence_level
        max_staleness_seconds = (
            intent.recency_window_seconds if intent is not None else self.config.max_staleness_seconds
        )

        for item in self._scope_items:
            matching = [entry for entry in evidence if entry.scope_item == item]
            if not matching:
                gaps.append(f"Scope item not covered: {item}")

        for entry in evidence:
            if entry.age_seconds is None and required_level >= EvidenceLevel.BEHAVIORAL:
                gaps.append(f"Evidence for '{entry.scope_item}' has no timestamp; freshness cannot be verified")
            elif (
                max_staleness_seconds is not None
                and entry.age_seconds is not None
                and entry.is_stale(max_staleness_seconds)
            ):
                gaps.append(
                    f"Stale evidence for '{entry.scope_item}': "
                    f"data is {entry.age_seconds:.0f}s old "
                    f"(limit: {max_staleness_seconds}s)"
                )

            if entry.level < required_level:
                gaps.append(
                    f"Insufficient evidence level for '{entry.scope_item}': "
                    f"{entry.level.value} < {required_level.value}"
                )

        return gaps

    def _mark_verified_scope_items(
        self,
        evidence: list[Evidence],
        intent: SuccessCriteria | None,
    ) -> None:
        """Track scope items that were verified by the current step."""
        required_level = intent.evidence_required if intent is not None else self.config.min_evidence_level
        max_staleness_seconds = (
            intent.recency_window_seconds if intent is not None else self.config.max_staleness_seconds
        )

        for entry in evidence:
            if entry.scope_item not in self._scope_items:
                continue
            if entry.level < required_level:
                continue
            if (
                max_staleness_seconds is not None
                and required_level >= EvidenceLevel.BEHAVIORAL
                and entry.age_seconds is None
            ):
                continue
            if (
                max_staleness_seconds is not None
                and entry.age_seconds is not None
                and entry.is_stale(max_staleness_seconds)
            ):
                continue
            self._verified_items.add(entry.scope_item)

    def _extract_claims(self, thought: str, evidence: list[Evidence]) -> list[Claim]:
        """Extract status claims from the model thought and ground them to evidence."""
        if not thought:
            return []

        claims: list[Claim] = []
        seen: set[str] = set()
        claim_patterns = [
            r"(?:^|\n)(?:status|claim)\s*:\s*(.+)",
            r"(?:^|\n)(?:done|fixed|resolved)\s*:\s*(.+)",
            r"(?:^|\n)[\-\*]\s*(.+?\b(?:fixed|resolved|working|passed|healthy|stable)\b.+)",
        ]

        for pattern in claim_patterns:
            for match in re.finditer(pattern, thought, re.IGNORECASE):
                subject = match.group(1).strip()
                if not subject or subject in seen:
                    continue
                seen.add(subject)

                refs = self._ground_claim(subject, evidence)
                claims.append(
                    Claim(
                        subject=subject,
                        predicate="status",
                        evidence_refs=refs,
                        confidence=0.9 if refs else 0.1,
                    )
                )

        return claims

    def _ground_claim(self, subject: str, evidence: list[Evidence]) -> list[str]:
        """Associate a claim with evidence references from the current step."""
        subject_lower = subject.lower()
        matching_refs = [
            entry.reference
            for entry in evidence
            if entry.scope_item != "unknown" and entry.scope_item.lower() in subject_lower
        ]
        if matching_refs:
            return matching_refs

        if any(token in subject_lower for token in ("test", "build", "lint", "diff", "patch", "change")):
            typed_refs = [
                entry.reference
                for entry in evidence
                if entry.level in {EvidenceLevel.FUNCTIONAL, EvidenceLevel.COMPARATIVE}
            ]
            if typed_refs:
                return typed_refs

        return [entry.reference for entry in evidence] if evidence else []

    def _record_step(self, result: StepResult) -> None:
        """Append the step to the conversational history and trajectory."""
        assistant_lines = [f"Thought: {result.thought or '(empty)'}"]
        if result.intent is not None:
            assistant_lines.append(f"Goal: {result.intent.goal}")
            assistant_lines.append(f"Evidence: {result.intent.evidence_required.value}")
            if result.intent.scope_items:
                assistant_lines.append("Scope: " + ", ".join(result.intent.scope_items))
        if result.action:
            assistant_lines.append(f"Action: {result.action}")
        if result.done:
            assistant_lines.append("Done: true")

        self.history.append({"role": "assistant", "content": "\n".join(assistant_lines)})
        self.history.append({"role": "user", "content": result.observation})
        self.trajectory.append(result)
        self._publish_session(SessionEventType.STEP_RECORDED)

    def _coerce_evidence_level(
        self,
        raw: Any,
        default: EvidenceLevel,
    ) -> EvidenceLevel:
        """Parse a string or enum value into an EvidenceLevel."""
        if isinstance(raw, EvidenceLevel):
            return raw
        if isinstance(raw, str):
            try:
                return EvidenceLevel(raw.lower())
            except ValueError:
                return default
        return default

    def _normalize_scope_items(self, items: Any) -> list[str]:
        """Normalize scope collections into de-duplicated strings."""
        if isinstance(items, str):
            candidates = [items]
        else:
            candidates = list(items or [])
        normalized: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            text = str(item).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized
