import re

_DURATION = re.compile(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?")


def parse_duration(text: str) -> int:
    """Parse a duration like '1h30m' into seconds."""
    match = _DURATION.fullmatch(text.strip())
    if not match or not any(match.groups()):
        raise ValueError(f"invalid duration: {text!r}")
    hours, minutes, seconds = (int(g or 0) for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def format_duration(seconds: int) -> str:
    """Format seconds as e.g. '1h30m' (inverse of parse_duration)."""
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    parts = [f"{hours}h" if hours else "", f"{minutes}m" if minutes else "", f"{secs}s" if secs else ""]
    return "".join(parts) or "0s"
