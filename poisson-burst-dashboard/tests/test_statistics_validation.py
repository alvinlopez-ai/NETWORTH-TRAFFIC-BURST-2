import numpy as np

from statistics_validation import chi_square_goodness_of_fit, exponential_interarrival_ks


def test_validation_functions_handle_empty_or_too_small_data():
    assert chi_square_goodness_of_fit([])["available"] is False
    assert exponential_interarrival_ks([])["available"] is False


def test_chi_square_uses_fitted_poisson_model():
    counts = np.random.default_rng(10).poisson(20, 500)
    result = chi_square_goodness_of_fit(counts)
    assert result["available"] is True
    assert result["degrees_of_freedom"] > 0
    assert 0 <= result["p_value"] <= 1


def test_exponential_ks_uses_parametric_bootstrap():
    intervals = np.random.default_rng(10).exponential(0.2, 80)
    result = exponential_interarrival_ks(intervals, bootstrap_samples=49, seed=10)
    assert result["available"] is True
    assert result["bootstrap_samples"] == 49
    assert 0 <= result["p_value"] <= 1
