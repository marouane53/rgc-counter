from __future__ import annotations

import napari

from src.ui_napari import RGCCounterDockWidget


def main() -> None:
    viewer = napari.Viewer(title="RGC Counter")
    widget = RGCCounterDockWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right", name="RGC Counter")
    napari.run()


if __name__ == "__main__":
    main()
