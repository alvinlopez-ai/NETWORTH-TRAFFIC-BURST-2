# Communication Network Traffic Burst Detection Using Poisson Distribution

A local Streamlit dashboard that simulates packet arrivals in fixed-duration windows, applies an exact Poisson upper-tail calculation, and flags counts above a configurable threshold. It is intended for a mathematics mini project and classroom demonstration; it does not require router access or a network connection after dependencies are installed.

## Project structure

```text
poisson-burst-dashboard/
|-- app.py
|-- detector.py
|-- event_log.py
|-- simulator.py
|-- statistics_validation.py
|-- visualizations.py
|-- requirements.txt
|-- README.md
|-- .streamlit/config.toml
`-- tests/
    |-- test_detector.py
    |-- test_event_log.py
    |-- test_simulator.py
    `-- test_statistics_validation.py
```

## Features

- Live, automatically refreshing simulation with start, pause, resume, stop, and reset controls.
- λ from 10 to 200 packets/window, selectable 1/5/10/30-second windows, sensitivity k from 1 to 4, adjustable update speed, and a configurable run length.
- Normal traffic, one-off and repeated burst scenarios, gradually changing load, and a two-state calm/congested process.
- Manual burst injection and manual normal samples, separately labelled in the event log.
- Fixed configured baseline: injected burst counts never increase λ.
- Observed-count chart with mean, upper/lower guides, and burst markers; Poisson mass and inter-arrival-time visualizations.
- Exact SciPy Poisson survival probability, z-score, burst alert, event history, and CSV download.
- Chi-square Poisson goodness-of-fit diagnostic with pooled expected frequencies and parameter-adjusted degrees of freedom.
- Exponential inter-arrival KS diagnostic calibrated with a parametric bootstrap that refits the rate.
- Optional reproducible random seed.

Live packet capture is not included. The dashboard always runs in simulation mode; no Scapy, administrator privileges, network interface selection, or network access is needed.

## Mathematical background

For a fixed observation window, the Poisson model assumes a constant arrival rate and independent packet arrivals. If the expected count is λ, then

$$P(X=x)=e^{-\lambda}\frac{\lambda^x}{x!},\quad E[X]=\operatorname{Var}(X)=\lambda,\quad \sigma=\sqrt{\lambda}.$$

The adjustable upper limit is $T=\lambda+k\sqrt{\lambda}$. An observation is classified as a burst only when $x>T$ (strict inequality). The standardized deviation is $z=(x-\lambda)/\sqrt{\lambda}$. The reported exact upper-tail probability is $P(X\geq x)=\operatorname{sf}(x-1;\lambda)$ from SciPy.

A low upper-tail probability indicates that a count would be unusual if this Poisson model and its λ were correct. It is not the probability that an attack occurred. For λ = 0, the z-score is defined as 0 for x = 0 and positive infinity for x > 0 to avoid division by zero.

The count validation estimates λ using clean samples with the current λ and window duration. Adjacent count categories are pooled until their expected count is at least 5. With one fitted parameter, degrees of freedom are pooled-category-count − 1 − 1. The test is omitted when there are too few observations or categories. The inter-arrival test uses a separate sample of positive exponential waiting times simulated under the selected normal rate, not intervals reconstructed from the count chart. Its parametric bootstrap refits the rate in every replicate because a standard one-sample KS p-value is not valid as-is when the exponential rate is estimated. This checks the simulator's waiting-time assumption, not real network timestamps.

## Installation

Use Python 3.10 or later. Open a terminal in this project directory and create an isolated environment:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, either run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` with appropriate permission or use Command Prompt:

```bat
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Once installed, the application itself works offline.

## Run

```powershell
streamlit run app.py
```

Streamlit prints a local address, normally `http://localhost:8501`, and opens the dashboard in your browser. Stop it with Ctrl+C in the terminal.

Run automated tests from the project directory:

```powershell
python -m pytest -q
```

Expected test result: `14 passed` (test runtime is normally a few seconds). Tests cover threshold and z-score calculations, strict detection behavior, λ = 0 and small λ, seeded repeatability, burst labeling, CSV and empty-history handling, and both validation procedures. Actual results depend on installed package versions.

## Dashboard usage

1. Set the expected count λ, window duration, sensitivity k, simulation speed, and maximum number of windows in the sidebar.
2. Choose a scenario. For repeatable results, select the seed option, enter a seed, and press **Apply seed**.
3. Press **Start** for automatic updates. Pause/resume/stop work without deleting history; **Reset session** clears the event and inter-arrival histories.
4. Press **Generate normal** for one baseline sample or **Inject burst** to add a labelled burst to the next observation. During a running simulation, these are queued into the next refresh.
5. Read the latest alert, thresholds, z-score, exact upper-tail probability, and graph. Download the complete event log as CSV.
6. Use the **Model validation** tab after collecting clean normal samples. Burst and nonstationary scenario rows are intentionally excluded from the Poisson goodness-of-fit sample.

The dashboard refresh interval is the reciprocal of the selected windows-per-second value. It performs one sample per refresh and keeps all event history in Streamlit session state.

## Sample experiments

- **Baseline:** λ = 50, k = 2, normal traffic, seed = 42. Run 100 windows and inspect the count variation around 50.
- **Sensitivity comparison:** hold λ and seed fixed, compare k = 1 with k = 4, and note that the threshold and number of alerts change while each count and its p-value do not.
- **Injected anomaly:** run normal traffic, inject a burst, and confirm the event type changes to “Injected Burst” and the baseline remains 50.
- **Repeated spikes:** select “Repeated bursts” and run at least 40 windows to see periodic labelled spikes.
- **Model mismatch:** choose changing load or calm/congested traffic. Alerts may rise because the fixed-rate Poisson assumption is intentionally violated; these samples are not used in the goodness-of-fit test.
- **Validation:** collect at least 20 clean normal windows, open Model validation, and interpret p-values as evidence relative to the model, not a pass/fail certificate.

## College demonstration procedure

1. Introduce λ as the expected packets per window, then show σ = √λ and the upper threshold λ + k√λ.
2. Set λ = 50 and k = 2, choose normal traffic and seed 42, then start. Point out that normal random variation does not imply an anomaly.
3. Inject a burst and compare the observed count, threshold, z-score, exact P(X ≥ x), red chart marker, and event type.
4. Change k to 1 and then 4 to demonstrate sensitivity without retraining or changing the baseline.
5. Export the CSV and show the labelled audit trail. Collect clean windows and show how goodness-of-fit and bootstrap KS diagnostics address different assumptions.
6. Close by distinguishing a statistical anomaly from a confirmed cyberattack and explaining how changing load can violate the stationary Poisson model.

## How this connects statistics and computer science

The project turns a discrete probability model into an operational monitoring rule. It demonstrates random-variable modelling, expectation and variance, exact tail probabilities, standardization, goodness-of-fit testing, parameter estimation, reproducible simulation, event-stream processing, and stateful visualization. These ideas are useful in computer systems when modelling arrivals, queues, service load, and unusual resource demand.

## Expected outputs

The dashboard shows eight current metrics, a continuously updating count plot, the Poisson probability mass function, inter-arrival diagnostics, a latest-window alert, and a timestamped event table. The CSV contains window number, timestamp, observed/expected count, sensitivity k, window duration, threshold, z-score, upper-tail p-value, status, event type, and scenario. Validation panels show test statistics and p-values when sufficient eligible data is available.

## Limitations and future enhancements

- The Poisson model assumes a constant rate and independent arrivals; real network traffic can be bursty, dependent, seasonal, and nonstationary.
- Simulation is not packet capture and does not establish malicious intent.
- Threshold exceedance is a configurable screening rule, not a significance-controlled multiple-testing procedure. Repeated windows can produce false alerts.
- Goodness-of-fit conclusions depend on sample size, grouping, and model assumptions. A non-significant p-value does not prove the model.
- The inter-arrival check uses simulated clean-normal arrival intervals; it is not a test of real network data.
- Live capture, protocol breakdown, labelled precision/recall evaluation, and baseline adaptation can be added as separately authorized future work.

## Troubleshooting

- **`streamlit` is not recognized:** activate the virtual environment, then run `python -m streamlit run app.py`.
- **`ModuleNotFoundError`:** confirm the environment is activated and rerun `python -m pip install -r requirements.txt`.
- **Dashboard does not refresh:** keep the Streamlit terminal process running; press Start and check the simulation status in the sidebar.
- **No burst appears:** set k lower, use Inject burst, or select Sudden burst / Repeated bursts and run enough windows.
- **Validation says there are too few samples:** goodness-of-fit needs at least 20 eligible Normal traffic rows; changing and injected scenarios are excluded by design.
- **Port 8501 is occupied:** run `python -m streamlit run app.py --server.port 8502`.
- **PowerShell activation is blocked:** use Command Prompt activation or adjust the current-user execution policy as described above.
