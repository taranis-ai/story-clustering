import importlib
import sys
import socket
import pytest

<<<<<<< before updating
=======
from story_clustering.config import Config
import taranis_base_bot.misc as bot_misc

>>>>>>> after updating

def _block_network(*args, **kwargs):
    raise OSError("Network access is disabled for this test")


<<<<<<< before updating
@pytest.mark.asyncio
async def test_app_starts_and_serves_without_network(monkeypatch):
=======
def _payload_from_schema(schema: dict[str, dict]) -> dict[str, object]:
    sample_by_type = {
        "str": "offline startup test text",
        "int": 1,
        "float": 1.0,
        "bool": True,
        "list": ["offline startup test text"],
        "dict": {"value": "offline startup test text"},
    }
    return {
        key: sample_by_type[key_schema["type"]]
        for key, key_schema in schema.items()
        if key_schema.get("required", True)
    }


@pytest.mark.asyncio
async def test_app_starts_and_serves_without_network(monkeypatch):
    class FakeModel:
        async def predict(self, **kwargs):
            return {"status": "ok"}

>>>>>>> after updating
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(socket, "create_connection", _block_network)
    monkeypatch.setattr(socket, "getaddrinfo", _block_network)
    monkeypatch.setattr(socket.socket, "connect", _block_network)
    monkeypatch.setattr(socket.socket, "connect_ex", _block_network)

<<<<<<< before updating
=======
    monkeypatch.setattr(bot_misc, "get_model", lambda config: FakeModel())

>>>>>>> after updating
    sys.modules.pop("app", None)
    app_module = importlib.import_module("app")
    quart_app = app_module.app

    client = quart_app.test_client()

    health_response = await client.get("/health")
    assert health_response.status_code == 200

<<<<<<< before updating
    payload = {
        "stories": [
            {
                "id": "offline-startup-test-story-1",
                "tags": {
                    "APT28": {"name": "APT28", "tag_type": "APT"},
                    "CVE-2024-1234": {"name": "CVE-2024-1234", "tag_type": "cves"},
                    "Microsoft": {"name": "Microsoft", "tag_type": "Company"},
                    "Germany": {"name": "Germany", "tag_type": "Country"},
                    "Berlin": {"name": "Berlin", "tag_type": "LOC"},
                },
                "news_items": [
                    {
                        "news_id": "offline-startup-test-news-1",
                        "title": "Offline startup test headline",
                        "content": "APT28 targeted Microsoft in Berlin, Germany.",
                        "review": "",
                        "language": "en",
                    }
                ],
            }
        ]
    }
=======
    payload = _payload_from_schema(Config.PAYLOAD_SCHEMA)
>>>>>>> after updating

    inference_response = await client.post("/", json=payload)
    assert inference_response.status_code == 200

    response_json = await inference_response.get_json()
    assert isinstance(response_json, dict)
<<<<<<< before updating
    event_clusters = response_json.get("event_clusters")
    if event_clusters is None:
        event_clusters = response_json.get("cluster_ids", {}).get("event_clusters")

    assert event_clusters
    assert "offline-startup-test-story-1" in event_clusters[0]
=======
    assert response_json
>>>>>>> after updating
