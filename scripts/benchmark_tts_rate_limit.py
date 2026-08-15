#!/usr/bin/env python3
"""Benchmark Groq TTS local free-tier guard accounting."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom_components.groq.api import (  # noqa: E402
    RATE_LIMIT_DAY_SECONDS,
    GroqApiClient,
    SpeechRequest,
)

SCENARIO_STEADY_STATE = "steady-state"
SCENARIO_EXPIRED_PRUNE = "expired-prune"
DEFAULT_STEADY_HISTORY = 100
DEFAULT_STEADY_ITERATIONS = 10_000
DEFAULT_PRUNE_HISTORY = 50_000
DEFAULT_PRUNE_ITERATIONS = 20
BENCHMARK_NOW = 100_000.0


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """One benchmark scenario result and its retained usage state."""

    scenario: str
    average_us: float
    seeded_history: int
    retained_minute_requests: int
    retained_daily_requests: int
    iterations: int


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
    start = BENCHMARK_NOW - max(0, history_size - 1)
    for offset in range(history_size):
        client._record_local_tts_usage(request, 1, now=start + offset)
    return client, request


def _retained_history(client: GroqApiClient, request: SpeechRequest) -> tuple[int, int]:
    """Return retained minute and daily request counts."""
    state = client._tts_usage_state(request)
    return len(state.minute_request_timestamps), len(state.request_timestamps)


def run_steady_benchmark(history_size: int, iterations: int) -> BenchmarkResult:
    """Benchmark steady-state guard checks against active usage history."""
    client, request = build_client(history_size)
    start = time.perf_counter()
    for _ in range(iterations):
        client._check_local_tts_free_tier_limit(request, now=BENCHMARK_NOW)
    elapsed = time.perf_counter() - start
    retained_minute, retained_daily = _retained_history(client, request)
    return BenchmarkResult(
        scenario=SCENARIO_STEADY_STATE,
        average_us=elapsed * 1_000_000 / iterations,
        seeded_history=history_size,
        retained_minute_requests=retained_minute,
        retained_daily_requests=retained_daily,
        iterations=iterations,
    )


def run_benchmark(history_size: int, iterations: int) -> float:
    """Return steady-state average guard-check duration for compatibility."""
    return run_steady_benchmark(history_size, iterations).average_us


def _build_expired_client(
    history_size: int,
) -> tuple[GroqApiClient, SpeechRequest]:
    """Return a client whose usage history is entirely expired."""
    client, request = build_client(0)
    state = client._tts_usage_state(request)
    expired_at = BENCHMARK_NOW - RATE_LIMIT_DAY_SECONDS - 1
    for _ in range(history_size):
        state.request_timestamps.append(expired_at)
        state.token_timestamps.append((expired_at, 1))
        state.minute_request_timestamps.append(expired_at)
        state.minute_token_timestamps.append((expired_at, 1))
    state.daily_token_total = history_size
    state.minute_token_total = history_size
    return client, request


def run_prune_benchmark(history_size: int, iterations: int) -> BenchmarkResult:
    """Benchmark a guard check that must prune expired usage history."""
    elapsed = 0.0
    retained_minute = 0
    retained_daily = 0
    for _ in range(iterations):
        client, request = _build_expired_client(history_size)
        start = time.perf_counter()
        client._check_local_tts_free_tier_limit(request, now=BENCHMARK_NOW)
        elapsed += time.perf_counter() - start
        retained_minute, retained_daily = _retained_history(client, request)
    return BenchmarkResult(
        scenario=SCENARIO_EXPIRED_PRUNE,
        average_us=elapsed * 1_000_000 / iterations,
        seeded_history=history_size,
        retained_minute_requests=retained_minute,
        retained_daily_requests=retained_daily,
        iterations=iterations,
    )


def _non_negative_int(value: str) -> int:
    """Return a non-negative integer for argparse."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_int(value: str) -> int:
    """Return a positive integer for argparse."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Groq TTS local free-tier guard accounting."
    )
    parser.add_argument(
        "--scenario",
        choices=(SCENARIO_STEADY_STATE, SCENARIO_EXPIRED_PRUNE),
        default=SCENARIO_STEADY_STATE,
        help="Benchmark active-history checks or one-shot expired-history pruning.",
    )
    parser.add_argument(
        "--history-size",
        type=_non_negative_int,
        help="Number of historical local TTS requests to seed.",
    )
    parser.add_argument(
        "--iterations",
        type=_positive_int,
        help="Number of guard checks to time.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and print a compact result."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.scenario == SCENARIO_EXPIRED_PRUNE:
        history_size = (
            args.history_size
            if args.history_size is not None
            else DEFAULT_PRUNE_HISTORY
        )
        iterations = (
            args.iterations if args.iterations is not None else DEFAULT_PRUNE_ITERATIONS
        )
        result = run_prune_benchmark(history_size, iterations)
    else:
        history_size = (
            args.history_size
            if args.history_size is not None
            else DEFAULT_STEADY_HISTORY
        )
        iterations = (
            args.iterations
            if args.iterations is not None
            else DEFAULT_STEADY_ITERATIONS
        )
        result = run_steady_benchmark(history_size, iterations)
    print(
        "tts_free_tier_guard_avg_us="
        f"{result.average_us:.3f} scenario={result.scenario} "
        f"seeded_history={result.seeded_history} "
        f"retained_minute_requests={result.retained_minute_requests} "
        f"retained_daily_requests={result.retained_daily_requests} "
        f"iterations={result.iterations}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
