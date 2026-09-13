from pathlib import Path
from types import SimpleNamespace

import evaluate


class _FakeBox:
    map = 0.5
    map50 = 0.7
    map75 = 0.4
    maps = [0.1, 0.2]
    mp = 0.6
    mr = 0.55


class _FakeMetrics:
    box = _FakeBox()
    names = {0: "fastener_hardware", 1: "hand_tool"}
    fitness = 0.5


class _FakeModel:
    def val(self, **kwargs):
        assert kwargs["split"] == "val"
        assert kwargs["imgsz"] == 416
        return _FakeMetrics()


def test_evaluate_main_writes_json(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    data = tmp_path / "data.yaml"
    weights.write_text("x", encoding="utf-8")
    data.write_text("path: .\n", encoding="utf-8")
    output = tmp_path / "out" / "metrics.json"

    def factory(_path: str):
        return _FakeModel()

    result = evaluate.main(
        [
            "--weights",
            str(weights),
            "--data",
            str(data),
            "--imgsz",
            "416",
            "--split",
            "val",
            "--output",
            str(output),
        ],
        model_factory=factory,
    )
    assert output.exists()
    assert result["imgsz"] == 416
    assert "416" in result["note"]
    assert result["map50"] == 0.7
