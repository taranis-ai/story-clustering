import importlib
import sys
import socket
import pytest

from story_clustering.config import Config
import taranis_base_bot.misc as bot_misc


def _block_network(*args, **kwargs):
    raise OSError("Network access is disabled for this test")


def _payload_from_schema(schema: dict[str, dict]) -> dict[str, object]:
    sample_by_type = {
        "str": "offline startup test text",
        "int": 1,
        "float": 1.0,
        "bool": True,
        "list": ["offline startup test text"],
        "dict": {"value": "offline startup test text"},
    }
    return {key: sample_by_type[key_schema["type"]] for key, key_schema in schema.items() if key_schema.get("required", True)}


@pytest.mark.asyncio
async def test_app_starts_and_serves_without_network(monkeypatch):
    class FakeModel:
        async def predict(self, **kwargs):
            return {"status": "ok"}

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(socket, "create_connection", _block_network)
    monkeypatch.setattr(socket, "getaddrinfo", _block_network)
    monkeypatch.setattr(socket.socket, "connect", _block_network)
    monkeypatch.setattr(socket.socket, "connect_ex", _block_network)

    monkeypatch.setattr(bot_misc, "get_model", lambda config: FakeModel())

    sys.modules.pop("app", None)
    app_module = importlib.import_module("app")
    quart_app = app_module.app

    client = quart_app.test_client()

    health_response = await client.get("/health")
    assert health_response.status_code == 200

    payload = _payload_from_schema(Config.PAYLOAD_SCHEMA)

    inference_response = await client.post("/", json=payload)
    assert inference_response.status_code == 200

    response_json = await inference_response.get_json()
    assert isinstance(response_json, dict)
    assert response_json
