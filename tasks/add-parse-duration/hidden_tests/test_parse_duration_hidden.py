import pytest

from src.durations import format_duration, parse_duration


@pytest.mark.parametrize(
    "text, seconds",
    [("45s", 45), ("2h", 7200), ("1h30m", 5400), ("1h2m3s", 3723), ("90m", 5400), ("  10m ", 600)],
)
def test_valid(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "1x", "30m1h", "-5s", "h", "1h 30m", "1.5h"])
def test_invalid(text):
    with pytest.raises(ValueError):
        parse_duration(text)


def test_round_trip():
    for seconds in (1, 59, 61, 3600, 3723, 86399):
        assert parse_duration(format_duration(seconds)) == seconds
