import json
from datetime import datetime
from pathlib import Path

from src.context import RunContext
from src.provenance import build_run_provenance, write_provenance


def test_write_provenance_serializes_run_payload(tmp_path: Path):
    ctx = RunContext(path=Path("sample.tif"), image=None, meta={"reader": "test"})  # type: ignore[arg-type]
    ctx.metrics["cell_count"] = 3
    ctx.summary_row = {"filename": "sample.tif", "cell_count": 3}
    ctx.artifacts["object_table"] = Path("/tmp/sample_objects.csv")

    payload = build_run_provenance(
        args={"input_dir": "input"},
        resolved_config={"backend": "fake", "model_spec": {"model_label": "fake_builtin:demo"}},
        contexts=[ctx],
        run_started_at=datetime(2026, 1, 1, 12, 0, 0),
        run_finished_at=datetime(2026, 1, 1, 12, 1, 0),
        results_csv_path=tmp_path / "results.csv",
        study_statistics={"requested_mode": "auto", "warnings": ["fell back"]},
        model_spec={"model_label": "fake_builtin:demo", "source": "builtin"},
        spatial_analysis={"mode": "rigorous", "simulation_count": 8},
    )
    written = write_provenance(tmp_path / "provenance.json", payload)

    data = json.loads(written.read_text(encoding="utf-8"))
    assert data["resolved_config"]["backend"] == "fake"
    assert data["model_spec"]["model_label"] == "fake_builtin:demo"
    assert data["images"][0]["summary_row"]["cell_count"] == 3
    assert data["study_statistics"]["requested_mode"] == "auto"
    assert data["spatial_analysis"]["mode"] == "rigorous"
