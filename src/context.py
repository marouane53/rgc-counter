from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class RunContext:
    path: Path
    image: np.ndarray
    meta: dict[str, Any]

    gray: np.ndarray | None = None
    qc_mask: np.ndarray | None = None
    labels: np.ndarray | None = None

    object_table: pd.DataFrame | None = None
    region_table: pd.DataFrame | None = None
    study_table: pd.DataFrame | None = None

    artifacts: dict[str, Path] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    seg_info: dict[str, Any] = field(default_factory=dict)
    summary_row: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
