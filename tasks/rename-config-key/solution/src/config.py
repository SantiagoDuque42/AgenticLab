DEFAULTS = {"base_url": "https://api.example.com", "timeout_s": 30, "retries": 2}


def load_config(overrides: dict | None = None) -> dict:
    """Merge user overrides into the defaults."""
    overrides = dict(overrides or {})
    legacy_timeout = overrides.pop("timeout", None)
    if legacy_timeout is not None and "timeout_s" not in overrides:
        overrides["timeout_s"] = legacy_timeout
    config = dict(DEFAULTS)
    config.update(overrides)
    return config
