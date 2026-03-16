from pathlib import Path

from src import config


def test_load_config_falls_back_to_builtin_defaults(monkeypatch):
    monkeypatch.setattr(
        config,
        "candidate_config_paths",
        lambda: [Path("/definitely/missing/retinal-phenotyper-config.yaml")],
    )

    loaded, resolved = config.load_config()

    assert resolved is None
    assert loaded["microns_per_pixel"] == config.DEFAULT_CONFIG["microns_per_pixel"]
    assert loaded["cell_detection"]["model_type"] == config.DEFAULT_CONFIG["cell_detection"]["model_type"]
