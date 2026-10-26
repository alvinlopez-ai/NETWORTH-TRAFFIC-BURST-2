import math

import pytest

from detector import analyze_count, calculate_threshold, calculate_z_score


def test_threshold_calculation():
    assert calculate_threshold(50, 2) == pytest.approx(50 + 2 * math.sqrt(50))


def test_normal_count_is_not_flagged():
    result = analyze_count(55, 50, 2)
    assert result["detected"] is False
    assert result["status"] == "NORMAL TRAFFIC"


def test_count_above_threshold_is_flagged():
    result = analyze_count(70, 50, 2)
    assert result["detected"] is True
    assert 0 <= result["p_value"] <= 1


def test_z_score_calculation_and_zero_rate():
    assert calculate_z_score(60, 50) == pytest.approx(10 / math.sqrt(50))
    assert calculate_z_score(0, 0) == 0
    assert math.isinf(calculate_z_score(1, 0))


def test_small_expected_rate_and_strict_threshold_rule():
    assert analyze_count(0, 0.1, 2)["detected"] is False
    assert analyze_count(1, 0.1, 2)["detected"] is True
    assert analyze_count(0, 0, 2)["p_value"] == 1


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        calculate_threshold(-1, 2)
    with pytest.raises(ValueError):
        analyze_count(-1, 10, 2)
