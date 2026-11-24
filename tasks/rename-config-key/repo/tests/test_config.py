from src.config import load_config


def test_defaults():
    assert load_config()["retries"] == 2
