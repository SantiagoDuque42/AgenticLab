def parse_duration(text: str) -> int:
    """Parse a duration like '1h30m' into seconds."""
    raise NotImplementedError


def format_duration(seconds: int) -> str:
    """Format seconds as e.g. '1h30m' (inverse of parse_duration)."""
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    parts = [f"{hours}h" if hours else "", f"{minutes}m" if minutes else "", f"{secs}s" if secs else ""]
    return "".join(parts) or "0s"
