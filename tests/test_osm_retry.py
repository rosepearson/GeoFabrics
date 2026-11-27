import logging
import urllib.error
from unittest.mock import Mock, call

import pytest

from geofabrics import processor


def test_overpass_retries_and_preserves_query_options(monkeypatch, caplog):
    overpass = Mock()
    overpass.query.side_effect = [
        urllib.error.URLError("temporary outage"),
        Exception("[overpass] could not fetch or interpret status of the endpoint"),
        "result",
    ]
    factory = Mock(return_value=overpass)
    sleep = Mock()
    monkeypatch.setattr(processor.OSMPythonTools.overpass, "Overpass", factory)
    monkeypatch.setattr(processor.time, "sleep", sleep)
    monkeypatch.setattr(processor.random, "uniform", lambda start, end: 0.5)

    with caplog.at_level(logging.WARNING):
        result = processor._query_overpass("query", logging.getLogger(__name__), "2020")

    assert result == "result"
    assert factory.call_count == 3
    assert overpass.query.call_args_list == [
        call("query", timeout=60, date="2020"),
        call("query", timeout=60, date="2020"),
        call("query", timeout=60, date="2020"),
    ]
    assert sleep.call_args_list == [call(2.5), call(4.5)]
    assert len(caplog.records) == 2
    assert all(record.exc_info for record in caplog.records)


def test_overpass_exhaustion_preserves_cause(monkeypatch):
    http_error = urllib.error.HTTPError(
        "https://example.invalid", 502, "bad gateway", {}, None
    )
    error = Exception("The requested data could not be downloaded.", http_error)
    overpass = Mock()
    overpass.query.side_effect = error
    sleep = Mock()
    monkeypatch.setattr(
        processor.OSMPythonTools.overpass, "Overpass", Mock(return_value=overpass)
    )
    monkeypatch.setattr(processor.time, "sleep", sleep)
    monkeypatch.setattr(processor.random, "uniform", lambda start, end: 0)

    with pytest.raises(ConnectionError, match="3 attempts") as caught:
        processor._query_overpass("query", logging.getLogger(__name__))

    assert caught.value.__cause__ is error
    assert overpass.query.call_count == 3
    assert sleep.call_args_list == [call(2), call(4)]
    overpass.query.assert_called_with("query", timeout=60)


@pytest.mark.parametrize(
    "error",
    [
        ValueError("invalid input"),
        Exception(),
        Exception(
            "The requested data could not be downloaded.",
            urllib.error.HTTPError(
                "https://example.invalid", 400, "bad request", {}, None
            ),
        ),
    ],
)
def test_overpass_does_not_retry_permanent_errors(monkeypatch, error):
    overpass = Mock()
    overpass.query.side_effect = error
    sleep = Mock()
    monkeypatch.setattr(
        processor.OSMPythonTools.overpass, "Overpass", Mock(return_value=overpass)
    )
    monkeypatch.setattr(processor.time, "sleep", sleep)

    with pytest.raises(type(error)) as caught:
        processor._query_overpass("query", logging.getLogger(__name__))

    assert caught.value is error
    assert overpass.query.call_count == 1
    sleep.assert_not_called()
