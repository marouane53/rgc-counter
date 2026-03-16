from __future__ import annotations

import napari

from src.ui_napari import RGCCounterDockWidget


def main() -> None:
    viewer = napari.Viewer(title="retinal-phenotyper")
    widget = RGCCounterDockWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right", name="retinal-phenotyper")
    napari.run()


if __name__ == "__main__":
    main()
