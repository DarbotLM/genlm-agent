"""CLI entrypoint for GenLM-Agent."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from glmagent import __version__
from glmagent.swarm import (
    BUILTIN_PERSPECTIVES,
    ObservationalReviewer,
    ReviewMode,
    ReviewPolicy,
    ReviewTarget,
    default_review_panel,
)
from glmagent.verification import extract_evidence


def build_parser() -> argparse.ArgumentParser:
    """Build the GenLM-Agent CLI parser."""
    parser = argparse.ArgumentParser(
        prog="glmagent",
        description="GenLM-Agent verification-first coding-agent toolkit.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    verify_parser = subparsers.add_parser(
        "verify-observation",
        help="Extract typed evidence from an observation string or file.",
    )
    verify_parser.add_argument(
        "--text",
        help="Observation text to verify.",
    )
    verify_parser.add_argument(
        "--file",
        type=Path,
        help="Path to a file containing the observation to verify.",
    )
    verify_parser.add_argument(
        "--scope",
        action="append",
        default=[],
        help="Scope item to match in the observation. Repeatable.",
    )
    verify_parser.add_argument(
        "--source",
        default="cli",
        help="Evidence source label.",
    )
    verify_parser.set_defaults(handler=_handle_verify_observation)

    review_parser = subparsers.add_parser(
        "review-observation",
        help="Run a swarm of challenging reviewers over an observation.",
    )
    review_parser.add_argument(
        "--text",
        help="Observation text to review.",
    )
    review_parser.add_argument(
        "--file",
        type=Path,
        help="Path to a file containing the observation to review.",
    )
    review_parser.add_argument(
        "--scope",
        action="append",
        default=[],
        help="Scope item to match in the observation. Repeatable.",
    )
    review_parser.add_argument(
        "--action",
        default="",
        help="Proposed action the observation resulted from, for safety review.",
    )
    review_parser.add_argument(
        "--perspective",
        action="append",
        default=[],
        choices=sorted(BUILTIN_PERSPECTIVES),
        help="Reviewer perspective to include. Repeatable. Defaults to the full panel.",
    )
    review_parser.add_argument(
        "--mode",
        choices=[mode.value for mode in ReviewMode],
        default=ReviewMode.PARALLEL.value,
        help="Execution mode: parallel round, series chain, or chained parallel rounds.",
    )
    review_parser.add_argument(
        "--recency",
        type=int,
        default=300,
        help="Recency window in seconds for freshness review.",
    )
    review_parser.add_argument(
        "--source",
        default="cli",
        help="Evidence source label.",
    )
    review_parser.set_defaults(handler=_handle_review_observation)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args)


def _handle_verify_observation(args: argparse.Namespace) -> int:
    """Handle the verify-observation subcommand."""
    observation = _read_observation(args)
    evidence = extract_evidence(
        observation=observation,
        source=args.source,
        scope_items=args.scope or ["unknown"],
    )
    print(json.dumps([_serialize_evidence(item) for item in evidence], indent=2))
    return 0


def _handle_review_observation(args: argparse.Namespace) -> int:
    """Handle the review-observation subcommand."""
    observation = _read_observation(args)
    target = ReviewTarget.from_observation(
        observation,
        action=args.action,
        scope_items=args.scope or None,
        source=args.source,
        recency_window_seconds=args.recency,
    )
    if args.perspective:
        perspectives = [BUILTIN_PERSPECTIVES[name]() for name in args.perspective]
    else:
        perspectives = default_review_panel()
    reviewer = ObservationalReviewer(perspectives=perspectives, policy=ReviewPolicy())
    result = reviewer.review(target, mode=ReviewMode(args.mode))
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.approved else 1


def _read_observation(args: argparse.Namespace) -> str:
    """Read observation text from args."""
    if args.text:
        return args.text
    if args.file:
        return args.file.read_text(encoding="utf-8")
    msg = "Either --text or --file is required."
    raise SystemExit(msg)


def _serialize_evidence(item: object) -> dict[str, object]:
    """Convert dataclass evidence into JSON-safe primitives."""
    payload = asdict(item)
    for key in ("timestamp", "data_timestamp"):
        value = payload.get(key)
        if value is not None:
            payload[key] = value.isoformat()
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
