from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "scripts" / "benchmark_tts_rate_limit.py"
    spec = importlib.util.spec_from_file_location(
        "benchmark_tts_rate_limit", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_module()


def test_steady_benchmark_reports_retained_history() -> None:
    result = benchmark.run_steady_benchmark(history_size=100, iterations=5)

    assert result.scenario == benchmark.SCENARIO_STEADY_STATE
    assert result.seeded_history == 100
    assert result.retained_minute_requests == 60
    assert result.retained_daily_requests == 100
    assert result.iterations == 5
    assert result.average_us >= 0


def test_prune_benchmark_reports_expired_history_removed() -> None:
    result = benchmark.run_prune_benchmark(history_size=10, iterations=2)

    assert result.scenario == benchmark.SCENARIO_EXPIRED_PRUNE
    assert result.seeded_history == 10
    assert result.retained_minute_requests == 0
    assert result.retained_daily_requests == 0
    assert result.iterations == 2
    assert result.average_us >= 0


def test_main_reports_scenario_and_retained_history(capsys) -> None:
    exit_code = benchmark.main(["--history-size", "10", "--iterations", "2"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "scenario=steady-state" in output
    assert "seeded_history=10" in output
    assert "retained_minute_requests=10" in output
    assert "retained_daily_requests=10" in output


@pytest.mark.parametrize(
    "args",
    (["--history-size", "-1"], ["--iterations", "0"]),
)
def test_parse_args_rejects_invalid_sizes(args: list[str]) -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args(args)
