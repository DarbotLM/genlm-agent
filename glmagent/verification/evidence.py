"""Evidence types and extraction for verification framework.

This module defines the core data types shared between GenLM-Agent and SMAX
for evidence-based verification, plus utilities for extracting evidence
from command output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class EvidenceLevel(StrEnum):
    """Hierarchy of verification strength.

    PROXY: Process exists, file exists, exit code 0
    INDICATOR: Service status shows "active", no errors in last N lines
    BEHAVIORAL: Recent activity observed (logs within time window)
    FUNCTIONAL: Desired outcome confirmed (proof completed, test passed)
    COMPARATIVE: Before/after comparison confirms change took effect
    """

    PROXY = "proxy"
    INDICATOR = "indicator"
    BEHAVIORAL = "behavioral"
    FUNCTIONAL = "functional"
    COMPARATIVE = "comparative"

    def __ge__(self, other: EvidenceLevel) -> bool:
        order = list(EvidenceLevel)
        return order.index(self) >= order.index(other)

    def __gt__(self, other: EvidenceLevel) -> bool:
        order = list(EvidenceLevel)
        return order.index(self) > order.index(other)

    def __le__(self, other: EvidenceLevel) -> bool:
        order = list(EvidenceLevel)
        return order.index(self) <= order.index(other)

    def __lt__(self, other: EvidenceLevel) -> bool:
        order = list(EvidenceLevel)
        return order.index(self) < order.index(other)


@dataclass
class SuccessCriteria:
    """Declared before action execution - what does success look like?"""

    goal: str
    evidence_required: EvidenceLevel = EvidenceLevel.BEHAVIORAL
    recency_window_seconds: int | None = 300
    scope_items: list[str] = field(default_factory=list)
    failure_indicators: list[str] = field(default_factory=list)


@dataclass
class Evidence:
    """Collected after action execution - what did we actually observe?"""

    source: str
    timestamp: datetime
    data_timestamp: datetime | None
    level: EvidenceLevel
    content: str
    scope_item: str

    @property
    def age_seconds(self) -> float | None:
        if self.data_timestamp is None:
            return None
        return (self.timestamp - self.data_timestamp).total_seconds()

    def is_stale(self, max_age_seconds: int) -> bool:
        age = self.age_seconds
        if age is None:
            return True
        return age > max_age_seconds

    @property
    def reference(self) -> str:
        """Stable reference for claim grounding."""
        timestamp = self.data_timestamp.astimezone(UTC).isoformat() if self.data_timestamp is not None else "na"
        return f"{self.source}:{self.scope_item}:{self.level.value}:{timestamp}"


@dataclass
class Claim:
    """A status assertion the agent wants to make."""

    subject: str
    predicate: str
    evidence_refs: list[str] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def is_grounded(self) -> bool:
        return len(self.evidence_refs) > 0


TIMESTAMP_PATTERNS = [
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)",
    r"(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})",
    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)",
    r"\b(\d{10})\b",
]

TIMESTAMP_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
]


def _try_parse_timestamp(raw: str) -> datetime | None:
    """Parse a timestamp into UTC."""
    if re.fullmatch(r"\d{10}", raw):
        try:
            return datetime.fromtimestamp(int(raw), tz=UTC)
        except (ValueError, OSError, OverflowError):
            return None

    if re.fullmatch(r"\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}", raw):
        try:
            current_year = datetime.now(UTC).year
            parsed = datetime.strptime(
                f"{current_year} {raw}",
                "%Y %b %d %H:%M:%S",
            )
            return parsed.replace(tzinfo=UTC)
        except ValueError:
            return None

    for fmt in TIMESTAMP_FORMATS:
        try:
            parsed = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


def extract_latest_timestamp(text: str) -> datetime | None:
    """Extract the most recent timestamp from text."""
    timestamps: list[datetime] = []
    for pattern in TIMESTAMP_PATTERNS:
        for match in re.finditer(pattern, text):
            parsed = _try_parse_timestamp(match.group(1))
            if parsed is not None:
                timestamps.append(parsed)
    return max(timestamps) if timestamps else None


def classify_evidence_level(observation: str) -> EvidenceLevel:
    """Classify the evidence level based on observation content."""
    comparative_indicators = [
        r"^diff --git\b",
        r"^\+\+\+ .+$",
        r"^--- .+$",
        r"\bbefore\b.+\bafter\b",
        r"\bchanged files?\b",
        r"\bpatch(?:ed)?\b",
    ]
    for pattern in comparative_indicators:
        if re.search(pattern, observation, re.IGNORECASE | re.MULTILINE):
            return EvidenceLevel.COMPARATIVE

    functional_indicators = [
        r"\btests?\b.*\bpassed\b",
        r"\bpytest\b.*\bpassed\b",
        r"\bbuild\b.*\b(?:passed|succeeded|successful)\b",
        r"\blint\b.*\b(?:passed|clean)\b",
        r"\btypecheck\b.*\bpassed\b",
        r"\ball checks passed\b",
        r"\bproof\s+(?:completed|finished|submitted)\b",
    ]
    for pattern in functional_indicators:
        if re.search(pattern, observation, re.IGNORECASE):
            return EvidenceLevel.FUNCTIONAL

    behavioral_indicators = [
        r"\d+\s*ips",
        r"new\s+(?:peak|block|height)",
        r"connected\s+to\s+\d+\s+peers",
        r"synced\s+to\s+height",
        r"processed\s+\d+\s+requests?",
    ]
    for pattern in behavioral_indicators:
        if re.search(pattern, observation, re.IGNORECASE):
            return EvidenceLevel.BEHAVIORAL

    indicator_patterns = [
        r"active\s*\(running\)",
        r"status:\s*(?:active|running|ok)",
        r"(?:no|0)\s+errors?",
        r"exit code:\s*0",
    ]
    for pattern in indicator_patterns:
        if re.search(pattern, observation, re.IGNORECASE):
            return EvidenceLevel.INDICATOR

    return EvidenceLevel.PROXY


def extract_evidence(
    observation: str,
    source: str = "command_output",
    scope_item: str = "unknown",
    scope_items: list[str] | None = None,
) -> list[Evidence]:
    """Extract structured evidence from command observation."""
    now = datetime.now(UTC)
    latest_ts = extract_latest_timestamp(observation)
    level = classify_evidence_level(observation)
    if latest_ts is None and level in {EvidenceLevel.FUNCTIONAL, EvidenceLevel.COMPARATIVE}:
        latest_ts = now

    requested_scope_items = _normalize_scope_items(scope_items or [scope_item])
    if not requested_scope_items:
        requested_scope_items = ["unknown"]

    mentioned_scope_items = [
        item
        for item in requested_scope_items
        if item != "unknown" and re.search(re.escape(item), observation, re.IGNORECASE)
    ]

    evidence_scope_items = mentioned_scope_items
    if not evidence_scope_items:
        evidence_scope_items = requested_scope_items if len(requested_scope_items) == 1 else ["unknown"]

    return [
        Evidence(
            source=source,
            timestamp=now,
            data_timestamp=latest_ts,
            level=level,
            content=observation[:1000],
            scope_item=item,
        )
        for item in evidence_scope_items
    ]


def _normalize_scope_items(items: list[str]) -> list[str]:
    """Normalize scope items into unique strings."""
    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized
