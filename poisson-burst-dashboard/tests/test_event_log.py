from event_log import EVENT_COLUMNS, events_frame, events_to_csv


def test_empty_event_log_has_headers_and_csv():
    frame = events_frame([])
    assert list(frame.columns) == EVENT_COLUMNS
    csv_text = events_to_csv([]).decode("utf-8")
    assert "window_number" in csv_text
    assert csv_text.endswith("\n")


def test_event_csv_contains_rows():
    event = {
        "window_number": 1,
        "timestamp": "2026-10-13T10:00:00+00:00",
        "observed_count": 52,
        "expected_count": 50,
        "threshold": 64.14,
        "z_score": 0.28,
        "p_value": 0.4,
        "status": "NORMAL TRAFFIC",
        "event_type": "Normal Simulation",
        "scenario": "Normal traffic",
    }
    csv_text = events_to_csv([event]).decode("utf-8")
    assert "Normal Simulation" in csv_text
    assert "52" in csv_text
