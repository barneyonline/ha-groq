#!/usr/bin/env python3
"""Benchmark Groq TTS local free-tier guard accounting."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom_components.groq.api import GroqApiClient, SpeechRequest  # noqa: E402

DEFAULT_HISTORY = 50_000
DEFAULT_ITERATIONS = 10_000


def _large_limits() -> dict[str, int]:
    """Return limits that keep the benchmark focused on accounting overhead."""
    return {
        "requests_per_minute": 1_000_000,
        "requests_per_day": 1_000_000,
        "tokens_per_minute": 1_000_000,
        "tokens_per_day": 1_000_000,
    }


def build_client(history_size: int) -> tuple[GroqApiClient, SpeechRequest]:
    """Return a client and TTS request with populated local usage history."""
    client = GroqApiClient(
        object(),  # type: ignore[arg-type]
        api_key="benchmark-key",
    )
    request = SpeechRequest(
        text="benchmark text",
        model="canopylabs/orpheus-v1-english",
        voice="tara",
    )
    client._free_tier_limits = lambda model: _large_limits()  # type: ignore[method-assign]
    now = 100_000.0
    start = now - max(0, history_size - 1)
    for offset in range(history_size):
        client._record_local_tts_usage(request, 1, now=start + offset)
    return client, request


def run_benchmark(history_size: int, iterations: int) -> float:
    """Return average guard-check duration in microseconds."""
    client, request = build_client(history_size)
    now = 100_000.0
    start = time.perf_counter()
    for _ in range(iterations):
        client._check_local_tts_free_tier_limit(request, now=now)
    elapsed = time.perf_counter() - start
    return elapsed * 1_000_000 / iterations


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Groq TTS local free-tier guard accounting."
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=DEFAULT_HISTORY,
        help="Number of historical local TTS requests to seed.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Number of guard checks to time.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and print a compact result."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    average_us = run_benchmark(args.history_size, args.iterations)
    print(
        "tts_free_tier_guard_avg_us="
        f"{average_us:.3f} history_size={args.history_size} "
        f"iterations={args.iterations}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
