from src.config import load_config


class Client:
    def __init__(self, config: dict | None = None) -> None:
        config = config or load_config()
        self.base_url = config["base_url"]
        self.timeout = config["timeout_s"]
        self.retries = config["retries"]

    def describe(self) -> str:
        return f"{self.base_url} (timeout {self.timeout}s, {self.retries} retries)"
