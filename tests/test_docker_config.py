"""Container/import readiness without trained weights."""

from pathlib import Path

from app import runtime_config
from app.reasoning import parse_intent


def test_taxonomy_config_is_loadable() -> None:
    assert runtime_config.TAXONOMY_PATH.exists()
    assert len(runtime_config.CLASS_NAMES) == 7
    assert runtime_config.IMGSZ == 480


def test_dockerfile_copies_config() -> None:
    text = Path("Dockerfile").read_text(encoding="utf-8")
    assert "COPY config ./config" in text
    assert "COPY app ./app" in text
