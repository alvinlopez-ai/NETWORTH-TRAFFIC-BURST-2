"""Streamlit dashboard for Poisson network-traffic burst detection."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import streamlit as st

from detector import analyze_count
from event_log import events_frame, events_to_csv
from simulator import SCENARIOS, TrafficSimulator
from statistics_validation import chi_square_goodness_of_fit, exponential_interarrival_ks
from visualizations import interarrival_figure, poisson_figure, traffic_figure


st.set_page_config(
    page_title="Poisson Burst Monitor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: 'Bahnschrift', 'Segoe UI', sans-serif; }
    .stApp { background: radial-gradient(ellipse at 76% 0%, #19312f 0%, #101820 34%, #0c1218 75%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #131e25 0%, #10171e 100%); border-right: 1px solid #26343c; }
    [data-testid="stMetric"] { background: linear-gradient(145deg, rgba(27,40,48,.94), rgba(18,28,35,.96)); border: 1px solid #2c3b43; border-radius: 7px; padding: 14px 16px; }
    [data-testid="stMetricLabel"] { color: #aab9bd; }
    [data-testid="stMetricValue"] { font-family: 'Consolas', monospace; font-size: 1.45rem; }
    h1, h2, h3 { letter-spacing: 0 !important; }
    .eyebrow { color: #60d9c5; font-family: 'Consolas', monospace; font-size: .72rem; text-transform: uppercase; letter-spacing: 1.2px; }
    .monitor-alert { border-radius: 7px; padding: 18px 20px; margin: 8px 0 16px; border: 1px solid; }
    .monitor-alert.burst { background: linear-gradient(105deg, rgba(111,34,49,.48), rgba(37,24,31,.78)); border-color: #9c4552; }
    .monitor-alert.normal { background: linear-gradient(105deg, rgba(20,81,75,.36), rgba(20,37,40,.76)); border-color: #347e72; }
    .alert-title { font-family: 'Consolas', monospace; font-size: 1.06rem; font-weight: 500; margin-bottom: 6px; }
    .alert-copy { color: #c5d0d1; font-size: .91rem; }
    .muted-note { color: #91a1a6; font-size: .84rem; }
    div[data-testid="stPlotlyChart"], div[data-testid="stImage"] { border-radius: 7px; }
    button[kind="primary"] { background: #35bba9; color: #061412; border: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    defaults = {
        "events": [],
        "running": False,
        "paused": False,
        "manual_step": False,
        "pending_burst": False,
        "pending_normal": False,
        "simulator": TrafficSimulator(),
        "normal_intervals": [],
        "interval_config": None,
        "last_sample_at": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()

with st.sidebar:
    st.markdown('<span class="eyebrow">Experiment controls</span>', unsafe_allow_html=True)
    st.header("Traffic model")
    expected_count = st.slider("Expected packets per window (λ)", 10, 200, 50, 1)
    window_duration = st.select_slider("Window duration", options=[1, 5, 10, 30], value=5, format_func=lambda value: f"{value} sec")
    sensitivity = st.slider("Sensitivity factor (k)", 1.0, 4.0, 2.0, 0.1)
    scenario = st.selectbox("Traffic scenario", SCENARIOS)

    st.markdown("---")
    st.markdown('<span class="eyebrow">Run configuration</span>', unsafe_allow_html=True)
    speed = st.slider("Simulation speed (windows/second)", 0.2, 2.0, 1.0, 0.1)
    target_windows = st.number_input("Windows to simulate", min_value=10, max_value=1000, value=100, step=10)
    use_seed = st.checkbox("Use reproducible random seed", value=False)
    seed_value = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=42, step=1, disabled=not use_seed)
    if st.button("Apply seed", use_container_width=True):
        st.session_state.simulator = TrafficSimulator(int(seed_value) if use_seed else None)
        st.toast("Random generator reseeded")

    st.markdown("---")
    st.markdown('<span class="eyebrow">Simulation controls</span>', unsafe_allow_html=True)
    first_controls = st.columns(2)
    if first_controls[0].button("Start", type="primary", use_container_width=True):
        st.session_state.running = True
        st.session_state.paused = False
    if first_controls[1].button("Pause", use_container_width=True, disabled=not st.session_state.running):
        st.session_state.running = False
        st.session_state.paused = True
    second_controls = st.columns(2)
    if second_controls[0].button("Resume", use_container_width=True, disabled=not st.session_state.paused):
        st.session_state.running = True
        st.session_state.paused = False
    if second_controls[1].button("Stop", use_container_width=True, disabled=not (st.session_state.running or st.session_state.paused)):
        st.session_state.running = False
        st.session_state.paused = False
    third_controls = st.columns(2)
    if third_controls[0].button("Generate normal", use_container_width=True):
        st.session_state.manual_step = True
        st.session_state.pending_burst = False
        st.session_state.pending_normal = True
    if third_controls[1].button("Inject burst", use_container_width=True):
        st.session_state.manual_step = True
        st.session_state.pending_burst = True
        st.session_state.pending_normal = False
    if st.button("Reset session", use_container_width=True):
        st.session_state.events = []
        st.session_state.normal_intervals = []
        st.session_state.running = False
        st.session_state.paused = False
        st.session_state.manual_step = False
        st.session_state.pending_burst = False
        st.session_state.pending_normal = False
        st.session_state.last_sample_at = None
        st.session_state.simulator = TrafficSimulator(int(seed_value) if use_seed else None)

    status = "RUNNING" if st.session_state.running else "PAUSED" if st.session_state.paused else "STOPPED"
    status_color = "#60d9c5" if status == "RUNNING" else "#f2b84b" if status == "PAUSED" else "#91a1a6"
    st.markdown(f'<p class="eyebrow" style="margin-top:16px;color:{status_color}">● {status} · SIMULATION MODE</p>', unsafe_allow_html=True)
    st.caption("Synthetic packet arrivals only. Live capture is not enabled.")

current_interval_config = (scenario, expected_count, window_duration)
if st.session_state.interval_config != current_interval_config:
    st.session_state.normal_intervals = []
    st.session_state.interval_config = current_interval_config


st.markdown('<span class="eyebrow">B.Tech AI & Data Science · Probability and Statistics</span>', unsafe_allow_html=True)
header_left, header_right = st.columns([4, 1])
with header_left:
    st.title("Communication Network Traffic Burst Detection")
    st.caption("Poisson arrival modelling · statistical anomaly detection · reproducible simulation")
with header_right:
    st.markdown("<div style='text-align:right;padding-top:24px'><span class='eyebrow'>MODE</span><br><b>SIMULATION</b></div>", unsafe_allow_html=True)


def render_monitor() -> None:
    should_sample = st.session_state.running or st.session_state.manual_step
    if should_sample and len(st.session_state.events) < int(target_windows):
        inject = bool(st.session_state.pending_burst)
        sample_scenario = "Normal traffic" if st.session_state.pending_normal else scenario
        sample = st.session_state.simulator.sample_window(
            expected_count=float(expected_count),
            duration_seconds=float(window_duration),
            window_number=len(st.session_state.events) + 1,
            scenario=sample_scenario,
            manual_burst=inject,
        )
        stats = analyze_count(sample["observed_count"], float(expected_count), float(sensitivity))
        event = {
            "window_number": len(st.session_state.events) + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "observed_count": stats["observed_count"],
            "expected_count": stats["expected_count"],
            "sensitivity_factor": stats["sensitivity_factor"],
            "window_duration_seconds": int(window_duration),
            "threshold": stats["threshold"],
            "z_score": stats["z_score"],
            "p_value": stats["p_value"],
            "status": stats["status"],
            "event_type": sample["event_type"],
            "scenario": sample_scenario,
        }
        st.session_state.events.append(event)
        if sample["event_type"] == "Normal Simulation" and sample_scenario == "Normal traffic":
            st.session_state.normal_intervals.extend(sample["interarrival_times"])
        st.session_state.last_sample_at = event["timestamp"]
        st.session_state.manual_step = False
        st.session_state.pending_burst = False
        st.session_state.pending_normal = False
        if len(st.session_state.events) >= int(target_windows):
            st.session_state.running = False
            st.session_state.paused = False
    elif should_sample:
        st.session_state.running = False
        st.session_state.paused = False
        st.session_state.manual_step = False
        st.session_state.pending_burst = False
        st.session_state.pending_normal = False

    live_status = "RUNNING" if st.session_state.running else "PAUSED" if st.session_state.paused else "STOPPED"
    st.markdown(f'<span class="eyebrow">● {live_status} · AUTO-REFRESH {speed:.1f} WINDOWS/SEC</span>', unsafe_allow_html=True)

    event_history = st.session_state.events
    current_frame = events_frame(event_history)
    latest = event_history[-1] if event_history else None
    detected_count = sum(event["status"] == "BURST DETECTED" for event in event_history)
    total_count = len(event_history)
    flagged_pct = 100 * detected_count / total_count if total_count else 0.0
    threshold_now = expected_count + sensitivity * math.sqrt(expected_count)

    metric_rows = st.columns(4)
    metric_rows[0].metric("Expected λ", f"{expected_count:g} packets")
    metric_rows[1].metric("Observed count", str(latest["observed_count"]) if latest else "—")
    metric_rows[2].metric("Std. deviation √λ", f"{math.sqrt(expected_count):.2f}")
    metric_rows[3].metric("Upper threshold", f"{threshold_now:.2f}")
    metric_rows = st.columns(4)
    metric_rows[0].metric("Latest z-score", f"{latest['z_score']:.2f}" if latest else "—")
    metric_rows[1].metric("Windows monitored", str(total_count))
    metric_rows[2].metric("Bursts detected", str(detected_count))
    metric_rows[3].metric("Windows flagged", f"{flagged_pct:.1f}%")

    if latest:
        burst = latest["status"] == "BURST DETECTED"
        tone = "burst" if burst else "normal"
        color = "#ff7886" if burst else "#62d8c4"
        explanation = (
            f"The observed count {latest['observed_count']} exceeded the strict upper threshold {latest['threshold']:.2f} for λ = {latest['expected_count']:.1f} and k = {latest['sensitivity_factor']:.1f}."
            if burst
            else f"The observed count {latest['observed_count']} is at or below the upper threshold {latest['threshold']:.2f} for λ = {latest['expected_count']:.1f}."
        )
        st.markdown(
            f'<div class="monitor-alert {tone}"><div class="alert-title" style="color:{color}">{latest["status"]}</div>'
            f'<div class="alert-copy">{explanation}<br>z = {latest["z_score"]:.3f} &nbsp; · &nbsp; P(X ≥ x) = {latest["p_value"]:.4g} &nbsp; · &nbsp; source: {latest["event_type"]}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("NORMAL TRAFFIC · Waiting for the first simulated observation.")

    st.markdown('<span class="muted-note">A detected burst is a statistical anomaly under the configured model, not proof of a cyberattack.</span>', unsafe_allow_html=True)
    figure = traffic_figure(current_frame)
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    lower_left, lower_right = st.columns([1.25, 1])
    with lower_left:
        figure = poisson_figure(expected_count, latest["observed_count"] if latest else None)
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    with lower_right:
        figure = interarrival_figure(st.session_state.normal_intervals)
        st.pyplot(figure, width="stretch")
        plt.close(figure)

    download_col, run_info_col = st.columns([1, 4])
    with download_col:
        st.download_button("Export event log CSV", data=events_to_csv(event_history), file_name="poisson_traffic_events.csv", mime="text/csv", use_container_width=True)
    with run_info_col:
        st.caption(f"Window: {window_duration} seconds · Scenario: {scenario} · Detection rule: count > λ + k√λ")
    st.subheader("Event log")
    if current_frame.empty:
        st.caption("No event rows yet. Start the simulation or generate a sample from the control panel.")
    else:
        st.dataframe(current_frame.iloc[::-1].head(100), use_container_width=True, hide_index=True)

refresh_interval = 1 / speed
@st.fragment(run_every=refresh_interval)
def live_monitor() -> None:
    render_monitor()


live_monitor()

validation_tab, math_tab = st.tabs(["Model validation", "How to read the model"])
with validation_tab:
    st.subheader("Poisson count goodness-of-fit")
    clean_events = [
        event for event in st.session_state.events
        if event["scenario"] == "Normal traffic"
        and event["event_type"] == "Normal Simulation"
        and event["expected_count"] == expected_count
        and event["window_duration_seconds"] == window_duration
    ]
    clean_counts = [int(event["observed_count"]) for event in clean_events]
    alpha = st.select_slider("Significance level (α)", options=[0.01, 0.05, 0.10], value=0.05, format_func=lambda value: f"{value:.2f}")
    validation = chi_square_goodness_of_fit(clean_counts, alpha=float(alpha))
    if not validation["available"]:
        st.info(validation["reason"])
    else:
        results = st.columns(4)
        results[0].metric("χ² statistic", f"{validation['statistic']:.3f}")
        results[1].metric("Degrees of freedom", str(validation["degrees_of_freedom"]))
        results[2].metric("p-value", f"{validation['p_value']:.4f}")
        results[3].metric("Clean samples", str(validation["sample_size"]))
        st.write(f"**λ estimated from sample:** {validation['lambda_estimate']:.3f} packets/window · **α:** {validation['alpha']:.2f} · {validation['categories']} pooled categories.")
        st.write(validation["interpretation"])
    st.markdown("The null hypothesis is that clean counts follow a Poisson distribution with λ estimated by the sample mean. Adjacent count categories are pooled until expected frequencies are at least 5; one degree of freedom is removed for estimating λ. Injected bursts, changing load, and calm/congested scenarios are excluded.")

    st.divider()
    st.subheader("Exponential inter-arrival check")
    ks_result = exponential_interarrival_ks(st.session_state.normal_intervals, seed=2026)
    if not ks_result["available"]:
        st.info(ks_result["reason"])
    else:
        results = st.columns(4)
        results[0].metric("KS statistic", f"{ks_result['statistic']:.4f}")
        results[1].metric("Bootstrap p-value", f"{ks_result['p_value']:.4f}")
        results[2].metric("Estimated rate", f"{ks_result['estimated_rate_per_second']:.3f}/sec")
        results[3].metric("Inter-arrivals", str(ks_result["sample_size"]))
        st.write(ks_result["interpretation"])
    st.markdown("The null hypothesis is that normal-traffic waiting times are exponential. These waiting times are independently simulated under the selected normal rate, rather than reconstructed from the count chart. The displayed p-value uses a parametric bootstrap: each sample is generated under the fitted rate and refitted before computing its KS statistic, accounting for parameter estimation. This diagnoses the assumed simulator model; it is not evidence about real network traffic.")

with math_tab:
    st.subheader("Detection rule")
    st.latex(r"P(X=x)=e^{-\lambda}\frac{\lambda^x}{x!}, \quad E[X]=\operatorname{Var}(X)=\lambda")
    st.latex(r"\sigma=\sqrt{\lambda}, \quad T_{upper}=\lambda+k\sqrt{\lambda}")
    st.latex(r"z=\frac{x-\lambda}{\sqrt{\lambda}}, \quad P(X\geq x)=\operatorname{sf}(x-1;\lambda)")
    st.write("λ is the expected packet count in one selected-duration window. For a rate r packets/second, λ = r × window duration. The decision is strict: a count is flagged only when x > T_upper. The upper-tail p-value is computed from the exact Poisson survival function, not a normal approximation.")
    st.write("Injected bursts are extra arrivals added after the ordinary Poisson sample. They never update the configured baseline λ. Changing-load and two-state scenarios deliberately violate the fixed-rate assumption and are kept out of the goodness-of-fit sample.")
    st.warning("Statistical anomaly ≠ confirmed attack. Bursts may also reflect legitimate workload changes, batching, or a poor baseline choice.")
