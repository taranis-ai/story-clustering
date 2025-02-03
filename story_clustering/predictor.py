from abc import ABC, abstractmethod
import requests


class Predictor(ABC):
    model_name: str

    def __init__(self):
        pass

    @abstractmethod
    def predict(self, text: str) -> dict[str, str]:
        pass

    @property
    def modelinfo(self) -> dict[str, str]:
        api_url = f"https://huggingface.co/api/models/{self.model_name}"
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()
