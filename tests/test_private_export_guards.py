from __future__ import annotations

from pathlib import Path

import pytest

from src.advisor_packet import assert_public_roi_benchmark_export_allowed, is_private_roi_benchmark_dir


def test_private_roi_benchmark_dir_is_detected(tmp_path: Path):
    private_path = tmp_path / "test_subjects" / "private" / "lane" / "benchmark"
    private_path.mkdir(parents=True)

    assert is_private_roi_benchmark_dir(private_path) is True


def test_private_roi_benchmark_export_requires_explicit_flag(tmp_path: Path):
    private_path = tmp_path / "test_subjects" / "private" / "lane" / "benchmark"
    private_path.mkdir(parents=True)

    with pytest.raises(ValueError, match="allow-private-roi-benchmark-export"):
        assert_public_roi_benchmark_export_allowed(private_path)

    assert_public_roi_benchmark_export_allowed(private_path, allow_private_roi_benchmark_export=True)
