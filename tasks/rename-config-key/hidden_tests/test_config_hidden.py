from src.cli import show_settings
from src.client import Client
from src.config import load_config


def test_default_timeout():
    assert load_config()["timeout_s"] == 30


def test_new_key_override():
    assert load_config({"timeout_s": 5})["timeout_s"] == 5


def test_old_key_still_works():
    config = load_config({"timeout": 7})
    assert config["timeout_s"] == 7


def test_new_key_wins_over_old():
    assert load_config({"timeout": 7, "timeout_s": 9})["timeout_s"] == 9


def test_client_and_cli():
    assert Client().timeout == 30
    assert Client(load_config({"timeout": 4})).timeout == 4
    assert show_settings() == "timeout=30 retries=2"
