DEFAULTS = {"base_url": "https://api.example.com", "timeout_s": 30, "retries": 2}


def load_config(overrides: dict | None = None) -> dict:
    """Merge user overrides into the defaults."""
    config = dict(DEFAULTS)
    config.update(overrides or {})
    return config
