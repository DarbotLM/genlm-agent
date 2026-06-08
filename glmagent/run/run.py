"""CLI entrypoint for GenLM-Agent."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from glmagent import __version__
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
