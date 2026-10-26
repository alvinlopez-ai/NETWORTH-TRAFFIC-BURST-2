"""Event-log schema and CSV serialization."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

EVENT_COLUMNS = [
    "window_number",
    "timestamp",
    "observed_count",
    "expected_count",
    "sensitivity_factor",
    "window_duration_seconds",
    "threshold",
    "z_score",
    "p_value",
    "status",
    "event_type",
    "scenario",
]


def events_frame(events: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """Build a consistently ordered event table, including for empty history."""
    return pd.DataFrame(list(events), columns=EVENT_COLUMNS)


def events_to_csv(events: Iterable[Mapping[str, object]]) -> bytes:
    """Serialize events for a browser download."""
    return events_frame(events).to_csv(index=False).encode("utf-8")
