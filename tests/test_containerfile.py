from pathlib import Path


def test_container_uses_asgi_interface_for_quart():
    containerfile = Path(__file__).resolve().parents[1] / "Containerfile"
    contents = containerfile.read_text()

    assert "ENV GRANIAN_INTERFACE=asgi" in contents
    assert "ENV GRANIAN_INTERFACE=wsgi" not in contents
    assert "ENV GRANIAN_BLOCKING_THREADS=1" in contents
