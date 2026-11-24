from src.config import load_config


def show_settings(overrides: dict | None = None) -> str:
    config = load_config(overrides)
    return f"timeout={config['timeout']} retries={config['retries']}"
