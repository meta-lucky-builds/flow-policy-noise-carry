"""Keeps the published claims tied to the raw episodes. Fails if the README drifts from the data."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

import analyze_results as ar  # noqa: E402


@pytest.fixture(scope="module")
def summary():
    return ar.summarize()


@pytest.fixture(scope="module")
def readme():
    return (ROOT / "README.md").read_text(encoding="utf-8")


def test_stored_summary_is_current(summary):
    stored = json.loads((ar.RESULTS / "summary.json").read_text())
    assert stored == json.loads(json.dumps(summary))


def test_run_is_complete_and_paired(summary):
    i = summary["integrity"]
    assert i["jobs"] == 80 and i["episodes"] == 1600 and i["start_state_mismatches"] == 0


def test_raw_cells_agree_with_the_run_output(summary):
    run = json.loads((ar.RESULTS / "quick_results.json").read_text())
    assert run["verdict"] == "NO-IMPROVEMENT-SHOWN"
    for method, value in run["success"].items():
        assert summary["success_pct"][method]["overall"] == pytest.approx(value * 100, abs=0.01)
    assert summary["effects"]["same_L10_r90 - fresh_L10"]["overall"]["diff_pts"] == pytest.approx(run["primary_same_minus_fresh"]["diff"] * 100)
    assert summary["effects"]["shift_L10_r90 - fresh_L10"]["overall"]["diff_pts"] == pytest.approx(run["shift_minus_fresh"]["diff"] * 100)


@pytest.mark.parametrize("key,condition,text", [
    ("fresh_L10 - native_L50", "clean", "+22.5"),
    ("fresh_L10 - native_L50", "delay4", "−14.0"),
    ("shift_L10_r90 - fresh_L10", "overall", "+1.0"),
    ("same_L10_r90 - fresh_L10", "overall", "−3.25"),
])
def test_readme_effects_match_the_data(summary, readme, key, condition, text):
    value = summary["effects"][key][condition]["diff_pts"]
    assert f"{value:+g}".replace("-", "−") in {text, text.rstrip("0").rstrip(".")}
    assert text in readme


def test_readme_intervals_and_rates_match_the_data(summary, readme):
    low, high = summary["effects"]["shift_L10_r90 - fresh_L10"]["overall"]["ci95_tasks_and_episodes_pts"]
    assert f"−{abs(low)} to +{high}" in readme
    low, high = summary["effects"]["fresh_L10 - native_L50"]["clean"]["ci95_tasks_and_episodes_pts"]
    assert f"+{low} to +{high}" in readme
    s = summary["success_pct"]
    for value in (s["fresh_L10"]["clean"], s["native_L50"]["clean"], s["fresh_L10"]["delay4"], s["native_L50"]["delay4"]):
        assert f"{value}%" in readme
    keep = summary["stage0"]["keep_same_observation"]["same"]
    assert f"{keep['0.0'] * 100:.1f}%" in readme and f"{keep['0.95'] * 100:.1f}%" in readme


def test_clean_replanning_wins_on_every_task(summary):
    e = summary["effects"]["fresh_L10 - native_L50"]["clean"]
    assert e["tasks_up"] == 10 and e["tasks_down"] == 0 and (e["a_only"], e["b_only"]) == (50, 5)


def test_exact_mcnemar_known_values():
    assert ar.exact_mcnemar(0, 0) == 1.0
    assert ar.exact_mcnemar(1, 1) == 1.0
    assert ar.exact_mcnemar(5, 0) == pytest.approx(2 / 32)
