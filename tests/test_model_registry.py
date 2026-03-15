from pathlib import Path

import pytest

from src.model_registry import resolve_model_spec
from src.models import CellposeSegmenter, build_segmenter


def test_resolve_builtin_cellpose_model():
    spec = resolve_model_spec(
        backend=None,
        model_type="cyto",
        cellpose_model=None,
        stardist_weights=None,
        sam_checkpoint=None,
        model_alias=None,
    )

    assert spec.backend == "cellpose"
    assert spec.source == "builtin"
    assert spec.builtin_name == "cyto"


def test_resolve_custom_cellpose_model_from_explicit_flag(tmp_path: Path):
    checkpoint = tmp_path / "rgc_model"
    checkpoint.write_text("model", encoding="utf-8")

    spec = resolve_model_spec(
        backend="cellpose",
        model_type="cyto",
        cellpose_model=str(checkpoint),
        stardist_weights=None,
        sam_checkpoint=None,
        model_alias="RGC v1",
    )

    assert spec.source == "custom"
    assert spec.asset_path == str(checkpoint.resolve())
    assert spec.alias == "RGC v1"


def test_resolve_legacy_custom_cellpose_path_from_model_type(tmp_path: Path):
    checkpoint = tmp_path / "legacy_model"
    checkpoint.write_text("legacy", encoding="utf-8")

    spec = resolve_model_spec(
        backend=None,
        model_type=str(checkpoint),
        cellpose_model=None,
        stardist_weights=None,
        sam_checkpoint=None,
        model_alias=None,
    )

    assert spec.source == "legacy_custom"
    assert spec.asset_path == str(checkpoint.resolve())


def test_resolve_custom_stardist_weights(tmp_path: Path):
    weights = tmp_path / "weights.h5"
    weights.write_text("weights", encoding="utf-8")

    spec = resolve_model_spec(
        backend=None,
        model_type=None,
        cellpose_model=None,
        stardist_weights=str(weights),
        sam_checkpoint=None,
        model_alias=None,
    )

    assert spec.backend == "stardist"
    assert spec.source == "custom"


def test_resolve_sam_requires_checkpoint(tmp_path: Path):
    with pytest.raises(ValueError):
        resolve_model_spec(
            backend="sam",
            model_type=None,
            cellpose_model=None,
            stardist_weights=None,
            sam_checkpoint=None,
            model_alias=None,
        )

    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_text("ckpt", encoding="utf-8")
    spec = resolve_model_spec(
        backend="sam",
        model_type=None,
        cellpose_model=None,
        stardist_weights=None,
        sam_checkpoint=str(checkpoint),
        model_alias=None,
    )
    assert spec.backend == "sam"
    assert spec.asset_path == str(checkpoint.resolve())


def test_resolve_model_spec_rejects_conflicting_custom_flags(tmp_path: Path):
    cellpose = tmp_path / "cellpose_model"
    stardist = tmp_path / "weights.h5"
    cellpose.write_text("c", encoding="utf-8")
    stardist.write_text("s", encoding="utf-8")

    with pytest.raises(ValueError):
        resolve_model_spec(
            backend=None,
            model_type=None,
            cellpose_model=str(cellpose),
            stardist_weights=str(stardist),
            sam_checkpoint=None,
            model_alias=None,
        )


def test_resolve_model_spec_rejects_missing_asset_path():
    with pytest.raises(FileNotFoundError):
        resolve_model_spec(
            backend="cellpose",
            model_type=None,
            cellpose_model="/tmp/does-not-exist",
            stardist_weights=None,
            sam_checkpoint=None,
            model_alias=None,
        )


def test_backend_inference_from_custom_flag(tmp_path: Path):
    checkpoint = tmp_path / "custom_cp"
    checkpoint.write_text("x", encoding="utf-8")

    spec = resolve_model_spec(
        backend=None,
        model_type=None,
        cellpose_model=str(checkpoint),
        stardist_weights=None,
        sam_checkpoint=None,
        model_alias="alias",
    )

    assert spec.backend == "cellpose"
    assert spec.display_label == "alias"
    assert spec.trust_mode == "trusted_local_only"


def test_build_segmenter_uses_model_spec_for_cellpose():
    spec = resolve_model_spec(
        backend="cellpose",
        model_type="cyto",
        cellpose_model=None,
        stardist_weights=None,
        sam_checkpoint=None,
        model_alias=None,
    )

    segmenter = build_segmenter(spec, diameter=20.0, use_gpu=False)

    assert isinstance(segmenter, CellposeSegmenter)
    assert segmenter.model_spec == spec


def test_build_segmenter_dispatches_optional_backends(monkeypatch, tmp_path: Path):
    weights = tmp_path / "weights.h5"
    weights.write_text("weights", encoding="utf-8")
    sam = tmp_path / "sam.pth"
    sam.write_text("ckpt", encoding="utf-8")
    calls: list[tuple[str, str]] = []

    class FakeStarDist:
        def __init__(self, model_spec):
            calls.append(("stardist", model_spec.asset_path or "builtin"))

    class FakeSAM:
        def __init__(self, model_spec, device):
            calls.append(("sam", device))

    monkeypatch.setattr("src.models.StarDistSegmenter", FakeStarDist)
    monkeypatch.setattr("src.models.SAMSegmenter", FakeSAM)

    stardist_spec = resolve_model_spec(
        backend="stardist",
        model_type=None,
        cellpose_model=None,
        stardist_weights=str(weights),
        sam_checkpoint=None,
        model_alias=None,
    )
    sam_spec = resolve_model_spec(
        backend="sam",
        model_type=None,
        cellpose_model=None,
        stardist_weights=None,
        sam_checkpoint=str(sam),
        model_alias=None,
    )

    build_segmenter(stardist_spec, diameter=None, use_gpu=False)
    build_segmenter(sam_spec, diameter=None, use_gpu=False)

    assert ("stardist", str(weights.resolve())) in calls
    assert ("sam", "cpu") in calls
