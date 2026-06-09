"""Portable runtime, session, and manifest contracts for GenLM-Agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum, StrEnum
from typing import Any, Protocol
from uuid import uuid4

from glmagent import __version__


class FrontendKind(StrEnum):
    """Frontend surfaces that can consume session snapshots."""

    CLI = "cli"
    TUI = "tui"
    WEB = "web"
    API = "api"


class SessionEventType(StrEnum):
    """Lifecycle events published by the runtime."""

    STARTED = "started"
    STEP_RECORDED = "step_recorded"
    FINISHED = "finished"


@dataclass
class CapabilityDescriptor:
    """A portable runtime capability."""

    name: str
    description: str
    enabled: bool = True


@dataclass
class IntegrationContract:
    """A JSON-friendly contract exposed to external frontends or adapters."""

    name: str
    kind: str
    description: str
    available: bool = True
    schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentManifest:
    """Static description of an agent runtime surface."""

    name: str
    version: str
    agent_type: str
    description: str
    frontends: list[FrontendKind] = field(default_factory=list)
    capabilities: list[CapabilityDescriptor] = field(default_factory=list)
    integrations: list[IntegrationContract] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render the manifest into JSON-safe data."""
        return to_json_compatible(self)


@dataclass
class SessionStepView:
    """Frontend-safe projection of a single runtime step."""

    index: int
    thought: str
    action: str
    observation: str
    done: bool
    verified: bool
    verification_gaps: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    scope_items: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionSnapshot:
    """Cross-platform projection of the runtime state."""

    session_id: str
    agent_type: str
    objective: str
    status: str
    scope_items: list[str] = field(default_factory=list)
    verified_items: list[str] = field(default_factory=list)
    steps: list[SessionStepView] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    frontends: list[FrontendKind] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render the snapshot into JSON-safe data."""
        return to_json_compatible(self)


@dataclass
class PublishedSessionEvent:
    """One emitted runtime event."""

    event: SessionEventType
    snapshot: SessionSnapshot


class SessionPresenter(Protocol):
    """Consumer of runtime session snapshots."""

    def publish(self, snapshot: SessionSnapshot, event: SessionEventType) -> None: ...


class NullSessionPresenter:
    """Presenter that discards all session events."""

    def publish(self, snapshot: SessionSnapshot, event: SessionEventType) -> None:
        return None


class CollectingSessionPresenter:
    """Presenter used for tests and in-memory embedding."""

    def __init__(self) -> None:
        self.events: list[PublishedSessionEvent] = []

    def publish(self, snapshot: SessionSnapshot, event: SessionEventType) -> None:
        self.events.append(PublishedSessionEvent(event=event, snapshot=snapshot))


def build_agent_manifest(agent_type: str = "verified") -> AgentManifest:
    """Build a portable manifest for a runtime surface."""
    frontends = [
        FrontendKind.CLI,
        FrontendKind.TUI,
        FrontendKind.WEB,
        FrontendKind.API,
    ]
    shared_capabilities = [
        CapabilityDescriptor(
            name="verification",
            description="Evidence-gated execution loop with grounded claims.",
        ),
        CapabilityDescriptor(
            name="session_snapshots",
            description="JSON-safe session state for CLI, web, TUI, and API frontends.",
        ),
        CapabilityDescriptor(
            name="frontend_abstraction",
            description="Presentation is decoupled through a snapshot publisher contract.",
        ),
        CapabilityDescriptor(
            name="model_adapter",
            description="Language-model backends plug in through a query(messages) adapter.",
        ),
        CapabilityDescriptor(
            name="environment_adapter",
            description="Execution backends plug in through environment protocols.",
        ),
    ]
    shared_integrations = [
        IntegrationContract(
            name="session-snapshot",
            kind="contract",
            description="Stable JSON projection for frontends and external controllers.",
            schema={
                "type": "object",
                "required": ["session_id", "agent_type", "status", "steps"],
            },
        ),
        IntegrationContract(
            name="verify-observation",
            kind="tool",
            description="Typed evidence extraction for raw observations.",
            schema={
                "type": "object",
                "required": ["text_or_file", "scope_items", "source"],
            },
        ),
        IntegrationContract(
            name="skill-catalog",
            kind="extension",
            description="Adapter slot for Claude-style skills or Copilot-style tool catalogs.",
            available=False,
            schema={
                "type": "object",
                "required": ["name", "description", "entrypoint"],
            },
        ),
        IntegrationContract(
            name="swarm-orchestrator",
            kind="extension",
            description="Adapter slot for AG2-like multi-agent orchestration.",
            available=False,
            schema={
                "type": "object",
                "required": ["agent_id", "role", "handoff_policy"],
            },
        ),
        IntegrationContract(
            name="observational-review",
            kind="tool",
            description="Swarm of challenging reviewer perspectives over an observation, run in parallel rounds or serial chains.",
            schema={
                "type": "object",
                "required": ["perspectives", "mode", "target"],
            },
        ),
    ]

    if agent_type == "task-engine":
        return AgentManifest(
            name="GenLM-Agent",
            version=__version__,
            agent_type="task-engine",
            description=(
                "Structured repository-task runtime with verification gates and portable session/front-end contracts."
            ),
            frontends=frontends,
            capabilities=shared_capabilities
            + [
                CapabilityDescriptor(
                    name="task_engine",
                    description="Narrow JSON task contract for repository workflows.",
                ),
                CapabilityDescriptor(
                    name="repository_context",
                    description="Repository snapshots are part of the execution context.",
                ),
            ],
            integrations=shared_integrations
            + [
                IntegrationContract(
                    name="repository-snapshot",
                    kind="contract",
                    description="Structured repository state injected into model messages.",
                    schema={
                        "type": "object",
                        "required": ["root", "dirty_files", "changed_files"],
                    },
                )
            ],
            metadata={
                "schema": "genlm-agent/manifest/v1",
                "open_contracts": ["session_snapshot", "agent_manifest", "task_spec"],
            },
        )

    return AgentManifest(
        name="GenLM-Agent",
        version=__version__,
        agent_type="verified",
        description=(
            "Verification-first runtime for LLM agents and coding workflows with portable session/front-end contracts."
        ),
        frontends=frontends,
        capabilities=shared_capabilities,
        integrations=shared_integrations,
        metadata={
            "schema": "genlm-agent/manifest/v1",
            "open_contracts": ["session_snapshot", "agent_manifest"],
        },
    )


class RuntimeAgentBase:
    """Shared state and projection layer for all GenLM-Agent runtimes."""

    agent_type = "verified"

    def __init__(self, presenter: SessionPresenter | None = None):
        self.presenter = presenter or NullSessionPresenter()
        self.history: list[dict[str, Any]] = []
        self.trajectory: list[Any] = []
        self._scope_items: list[str] = []
        self._verified_items: set[str] = set()
        self._session_id = uuid4().hex
        self._objective = ""
        self._status = "idle"

    @property
    def session_id(self) -> str:
        """Unique identifier for the current runtime session."""
        return self._session_id

    @property
    def objective(self) -> str:
        """Objective passed to the current run."""
        return self._objective

    @property
    def status(self) -> str:
        """Lifecycle status for the current session."""
        return self._status

    def set_presenter(self, presenter: SessionPresenter | None) -> None:
        """Replace the active presenter."""
        self.presenter = presenter or NullSessionPresenter()

    def set_scope(self, items: list[str]) -> None:
        """Declare the items that must all be verified."""
        self._scope_items = self._normalize_scope_items(items)
        self._verified_items.clear()

    def begin_session(
        self,
        objective: str,
        history: list[dict[str, Any]] | None = None,
    ) -> None:
        """Reset the runtime state for a new run."""
        self._session_id = uuid4().hex
        self._objective = objective
        self._status = "running"
        self.history = list(history or [])
        self.trajectory = []
        self._verified_items.clear()
        self._publish_session(SessionEventType.STARTED)

    def finish_session(self, status: str | None = None) -> None:
        """Mark the current session as finished."""
        self._status = status or self._default_session_status()
        self._publish_session(SessionEventType.FINISHED)

    def session_snapshot(self) -> SessionSnapshot:
        """Build a portable snapshot of the current runtime state."""
        manifest = self.describe_manifest()
        return SessionSnapshot(
            session_id=self._session_id,
            agent_type=self.agent_type,
            objective=self._objective,
            status=self._status,
            scope_items=list(self._scope_items),
            verified_items=sorted(self._verified_items),
            steps=[self._build_step_view(index + 1, step) for index, step in enumerate(self.trajectory)],
            history=[dict(entry) for entry in self.history],
            frontends=list(manifest.frontends),
            capabilities=[capability.name for capability in manifest.capabilities if capability.enabled],
            metadata={
                "schema": "genlm-agent/session/v1",
                **self.session_metadata(),
            },
        )

    def describe_manifest(self) -> AgentManifest:
        """Describe the runtime surface for embedding and UI layers."""
        manifest = build_agent_manifest(self.agent_type)
        manifest.metadata.update(self.manifest_metadata())
        return manifest

    def session_metadata(self) -> dict[str, Any]:
        """Attach runtime-specific metadata to the session snapshot."""
        return {}

    def manifest_metadata(self) -> dict[str, Any]:
        """Attach runtime-specific metadata to the agent manifest."""
        metadata: dict[str, Any] = {}
        config = getattr(self, "config", None)
        if config is not None:
            metadata["config"] = to_json_compatible(config)
        if self._scope_items:
            metadata["scope_items"] = list(self._scope_items)
        return metadata

    def _publish_session(self, event: SessionEventType) -> None:
        """Send the current session state to the configured presenter."""
        self.presenter.publish(self.session_snapshot(), event)

    def _default_session_status(self) -> str:
        """Compute the terminal status from the current trajectory."""
        if not self.trajectory:
            return "idle"
        last = self.trajectory[-1]
        if getattr(last, "done", False) and getattr(last, "verified", False):
            return "verified"
        return "incomplete"

    def _build_step_view(self, index: int, result: Any) -> SessionStepView:
        """Project one runtime result into a frontend-safe step view."""
        intent = getattr(result, "intent", None)
        scope_items = list(getattr(intent, "scope_items", []) or [])
        evidence_refs = [entry.reference for entry in getattr(result, "evidence", [])]
        return SessionStepView(
            index=index,
            thought=getattr(result, "thought", ""),
            action=getattr(result, "action", ""),
            observation=getattr(result, "observation", ""),
            done=bool(getattr(result, "done", False)),
            verified=bool(getattr(result, "verified", False)),
            verification_gaps=list(getattr(result, "verification_gaps", [])),
            evidence_refs=evidence_refs,
            scope_items=scope_items,
            metadata=self._step_metadata(result),
        )

    def _step_metadata(self, result: Any) -> dict[str, Any]:
        """Collect auxiliary metadata for a step view."""
        metadata: dict[str, Any] = {
            "execution_time": getattr(result, "execution_time", 0.0),
            "action_review": to_json_compatible(getattr(result, "action_review", {})),
        }
        for field_name in (
            "intent",
            "execution",
            "claims",
            "task",
            "snapshot_before",
            "snapshot_after",
        ):
            value = getattr(result, field_name, None)
            if value is not None:
                metadata[field_name] = to_json_compatible(value)
        return metadata

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


def to_json_compatible(value: Any) -> Any:
    """Convert runtime objects into JSON-safe primitives."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return to_json_compatible(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(item) for item in value]
    if isinstance(value, set):
        return [to_json_compatible(item) for item in sorted(value, key=str)]
    return value


__all__ = [
    "AgentManifest",
    "CapabilityDescriptor",
    "CollectingSessionPresenter",
    "FrontendKind",
    "IntegrationContract",
    "PublishedSessionEvent",
    "RuntimeAgentBase",
    "SessionEventType",
    "SessionPresenter",
    "SessionSnapshot",
    "SessionStepView",
    "build_agent_manifest",
    "to_json_compatible",
]
