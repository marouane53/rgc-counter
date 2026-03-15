from __future__ import annotations

from src.ui_napari.dock_widget import RGCCounterDockWidget


def napari_experimental_provide_dock_widget():
    return RGCCounterDockWidget


__all__ = ["RGCCounterDockWidget", "napari_experimental_provide_dock_widget"]
