import importlib
import sys
import socket
import pytest


def _block_network(*args, **kwargs):
    raise OSError("Network access is disabled for this test")


@pytest.mark.asyncio
async def test_app_starts_and_serves_without_network(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(socket, "create_connection", _block_network)
    monkeypatch.setattr(socket, "getaddrinfo", _block_network)
    monkeypatch.setattr(socket.socket, "connect", _block_network)
    monkeypatch.setattr(socket.socket, "connect_ex", _block_network)

    sys.modules.pop("app", None)
    app_module = importlib.import_module("app")
    quart_app = app_module.app

    client = quart_app.test_client()

    health_response = await client.get("/health")
    assert health_response.status_code == 200

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

    inference_response = await client.post("/", json=payload)
    assert inference_response.status_code == 200

    response_json = await inference_response.get_json()
    assert isinstance(response_json, dict)
    event_clusters = response_json.get("event_clusters")
    if event_clusters is None:
        event_clusters = response_json.get("cluster_ids", {}).get("event_clusters")

    assert event_clusters
    assert "offline-startup-test-story-1" in event_clusters[0]
