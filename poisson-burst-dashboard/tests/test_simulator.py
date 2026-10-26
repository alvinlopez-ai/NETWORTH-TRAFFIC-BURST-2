import numpy as np

from simulator import TrafficSimulator, simulate_counts


def test_simulation_is_reproducible():
    first = simulate_counts(50, 100, seed=7)
    second = simulate_counts(50, 100, seed=7)
    np.testing.assert_array_equal(first, second)


def test_manual_burst_is_labelled_and_exceeds_normal_baseline():
    simulator = TrafficSimulator(seed=7)
    result = simulator.sample_window(50, 5, 1, manual_burst=True)
    assert result["event_type"] == "Injected Burst"
    assert result["injected"] is True
    assert result["observed_count"] >= 50


def test_normal_traffic_is_not_labelled_as_injected():
    result = TrafficSimulator(seed=7).sample_window(50, 5, 1)
    assert result["event_type"] == "Normal Simulation"
    assert result["injected"] is False
