from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.edits import default_edit_log_path, save_edit_log
from src.run_service import RuntimeOptions, build_runtime, export_context, run_array, summarize_context
from src.ui_napari.helpers import (
    format_xy_text,
    landmarks_from_points,
    parse_optional_float,
    parse_optional_int,
    parse_xy_text,
)


class RGCCounterDockWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._last_ctx = None
        self._last_runtime = None
        self._pending_edits: list[dict[str, object]] = []

        self.setWindowTitle("retinal-phenotyper")
        self._build_ui()
        self._bind_viewer_events()
        self.refresh_layer_choices()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        self.image_layer_combo = QComboBox()
        self.landmark_layer_combo = QComboBox()
        refresh_button = QPushButton("Refresh Layers")
        refresh_button.clicked.connect(self.refresh_layer_choices)

        layer_row = QHBoxLayout()
        layer_row.addWidget(self.image_layer_combo)
        layer_row.addWidget(refresh_button)
        root.addLayout(layer_row)

        config_box = QGroupBox("Run Options")
        form = QFormLayout(config_box)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["cellpose", "blob_watershed", "stardist", "sam"])
        self.modality_combo = QComboBox()
        self.modality_combo.addItems(["flatmount", "oct", "vis_octf", "lightsheet"])
        self.projection_combo = QComboBox()
        self.projection_combo.addItems(["max", "mean", "sum"])
        self.focus_mode_combo = QComboBox()
        self.focus_mode_combo.addItems(["none", "qc", "auto"])
        self.onh_mode_combo = QComboBox()
        self.onh_mode_combo.addItems(["cli", "sidecar", "auto_hole", "auto_combined"])
        self.phenotype_engine_combo = QComboBox()
        self.phenotype_engine_combo.addItems(["legacy", "v2"])
        self.region_schema_combo = QComboBox()
        self.region_schema_combo.addItems(["mouse_flatmount_v1", "rat_flatmount_v1"])

        self.use_gpu_checkbox = QCheckBox("Use GPU if available")
        self.use_gpu_checkbox.setChecked(True)
        self.apply_clahe_checkbox = QCheckBox("Apply CLAHE")
        self.marker_metrics_checkbox = QCheckBox("Marker metrics")
        self.interaction_metrics_checkbox = QCheckBox("Interaction metrics")
        self.spatial_stats_checkbox = QCheckBox("Spatial stats")
        self.register_retina_checkbox = QCheckBox("Register retina")
        self.save_debug_checkbox = QCheckBox("Save debug overlay on export")
        self.save_debug_checkbox.setChecked(True)
        self.write_report_checkbox = QCheckBox("Write HTML report on export")
        self.write_report_checkbox.setChecked(True)
        self.write_object_table_checkbox = QCheckBox("Write object table on export")
        self.write_object_table_checkbox.setChecked(True)
        self.write_provenance_checkbox = QCheckBox("Write provenance on export")
        self.write_provenance_checkbox.setChecked(True)
        self.write_uncertainty_maps_checkbox = QCheckBox("Write uncertainty maps on export")
        self.write_qc_maps_checkbox = QCheckBox("Write QC maps on export")
        self.strict_schemas_checkbox = QCheckBox("Strict schemas on export")
        self.tiling_checkbox = QCheckBox("Enable tiling")

        self.diameter_edit = QLineEdit()
        self.diameter_edit.setPlaceholderText("Use config default")
        self.min_size_edit = QLineEdit()
        self.min_size_edit.setPlaceholderText("Use config default")
        self.max_size_edit = QLineEdit()
        self.max_size_edit.setPlaceholderText("Use config default")
        self.tile_size_edit = QLineEdit("1024")
        self.tile_overlap_edit = QLineEdit("128")
        self.channel_index_edit = QLineEdit()
        self.channel_index_edit.setPlaceholderText("Default 0")
        self.phenotype_config_edit = QLineEdit()
        self.phenotype_config_edit.setPlaceholderText("Optional YAML path")
        self.output_dir_edit = QLineEdit(str((Path.cwd() / "Outputs_napari").resolve()))
        self.retina_frame_path_edit = QLineEdit()
        self.retina_frame_path_edit.setPlaceholderText("Optional sidecar JSON")
        self.edit_log_path_edit = QLineEdit()
        self.edit_log_path_edit.setPlaceholderText("Optional review sidecar JSON")
        self.onh_xy_edit = QLineEdit()
        self.onh_xy_edit.setPlaceholderText("x, y")
        self.dorsal_xy_edit = QLineEdit()
        self.dorsal_xy_edit.setPlaceholderText("x, y")
        self.phenotype_relabel_edit = QLineEdit()
        self.phenotype_relabel_edit.setPlaceholderText("Phenotype label")

        self.capture_landmarks_button = QPushButton("Use 2-Point Landmarks")
        self.capture_landmarks_button.clicked.connect(self.capture_landmarks_from_layer)
        self.delete_selected_button = QPushButton("Delete Selected")
        self.delete_selected_button.clicked.connect(self.delete_selected_objects)
        self.merge_selected_button = QPushButton("Merge Selected")
        self.merge_selected_button.clicked.connect(self.merge_selected_objects)
        self.relabel_selected_button = QPushButton("Relabel Selected")
        self.relabel_selected_button.clicked.connect(self.relabel_selected_objects)
        self.save_edits_button = QPushButton("Save Edits")
        self.save_edits_button.clicked.connect(self.save_pending_edits)

        form.addRow("Image layer", self.image_layer_combo)
        form.addRow("Landmarks layer", self.landmark_layer_combo)
        form.addRow("", self.capture_landmarks_button)
        form.addRow("Backend", self.backend_combo)
        form.addRow("Modality", self.modality_combo)
        form.addRow("Projection", self.projection_combo)
        form.addRow("Focus mode", self.focus_mode_combo)
        form.addRow("ONH mode", self.onh_mode_combo)
        form.addRow("Phenotype engine", self.phenotype_engine_combo)
        form.addRow("Diameter", self.diameter_edit)
        form.addRow("Min size", self.min_size_edit)
        form.addRow("Max size", self.max_size_edit)
        form.addRow("Tile size", self.tile_size_edit)
        form.addRow("Tile overlap", self.tile_overlap_edit)
        form.addRow("Channel index", self.channel_index_edit)
        form.addRow("Phenotype config", self.phenotype_config_edit)
        form.addRow("Region schema", self.region_schema_combo)
        form.addRow("Retina frame", self.retina_frame_path_edit)
        form.addRow("Edit log", self.edit_log_path_edit)
        form.addRow("ONH xy", self.onh_xy_edit)
        form.addRow("Dorsal xy", self.dorsal_xy_edit)
        form.addRow("Relabel phenotype", self.phenotype_relabel_edit)
        form.addRow("Output dir", self.output_dir_edit)
        form.addRow("", self.use_gpu_checkbox)
        form.addRow("", self.apply_clahe_checkbox)
        form.addRow("", self.marker_metrics_checkbox)
        form.addRow("", self.interaction_metrics_checkbox)
        form.addRow("", self.spatial_stats_checkbox)
        form.addRow("", self.register_retina_checkbox)
        form.addRow("", self.tiling_checkbox)
        form.addRow("", self.save_debug_checkbox)
        form.addRow("", self.write_report_checkbox)
        form.addRow("", self.write_object_table_checkbox)
        form.addRow("", self.write_provenance_checkbox)
        form.addRow("", self.write_uncertainty_maps_checkbox)
        form.addRow("", self.write_qc_maps_checkbox)
        form.addRow("", self.strict_schemas_checkbox)
        root.addWidget(config_box)

        button_row = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_pipeline)
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.export_latest_run)
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.export_button)
        root.addLayout(button_row)

        review_row = QHBoxLayout()
        review_row.addWidget(self.delete_selected_button)
        review_row.addWidget(self.merge_selected_button)
        review_row.addWidget(self.relabel_selected_button)
        review_row.addWidget(self.save_edits_button)
        root.addLayout(review_row)

        self.status_label = QLabel("Ready.")
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlaceholderText("Run summary appears here.")
        root.addWidget(self.status_label)
        root.addWidget(self.summary)

    def _bind_viewer_events(self) -> None:
        self.viewer.layers.events.inserted.connect(lambda event=None: self.refresh_layer_choices())
        self.viewer.layers.events.removed.connect(lambda event=None: self.refresh_layer_choices())
        self.viewer.layers.events.reordered.connect(lambda event=None: self.refresh_layer_choices())

    def _layer_names(self, layer_type: str) -> list[str]:
        names: list[str] = []
        for layer in self.viewer.layers:
            if getattr(layer, "_type_string", "") == layer_type:
                names.append(layer.name)
        return names

    def refresh_layer_choices(self) -> None:
        image_names = self._layer_names("image")
        point_names = self._layer_names("points")

        self.image_layer_combo.clear()
        self.image_layer_combo.addItems(image_names)

        self.landmark_layer_combo.clear()
        self.landmark_layer_combo.addItem("")
        self.landmark_layer_combo.addItems(point_names)

    def _get_layer(self, name: str):
        for layer in self.viewer.layers:
            if layer.name == name:
                return layer
        return None

    def _get_source_path(self, image_layer) -> str:
        source = getattr(image_layer, "source", None)
        source_path = getattr(source, "path", None)
        if source_path:
            return str(source_path)
        return f"{image_layer.name}.tif"

    def _collect_options(self) -> RuntimeOptions:
        retina_frame_path = self.retina_frame_path_edit.text().strip() or None
        edit_log_path = self.edit_log_path_edit.text().strip() or None
        onh_xy = parse_xy_text(self.onh_xy_edit.text()) if self.onh_xy_edit.text().strip() else None
        dorsal_xy = parse_xy_text(self.dorsal_xy_edit.text()) if self.dorsal_xy_edit.text().strip() else None

        return RuntimeOptions(
            backend=self.backend_combo.currentText(),
            modality=self.modality_combo.currentText(),
            modality_projection=self.projection_combo.currentText(),
            modality_channel_index=parse_optional_int(self.channel_index_edit.text()) if self.channel_index_edit.text().strip() else 0,
            diameter=parse_optional_float(self.diameter_edit.text()),
            min_size=parse_optional_int(self.min_size_edit.text()),
            max_size=parse_optional_int(self.max_size_edit.text()),
            tiling=self.tiling_checkbox.isChecked(),
            tile_size=parse_optional_int(self.tile_size_edit.text()) or 1024,
            tile_overlap=parse_optional_int(self.tile_overlap_edit.text()) or 128,
            use_gpu=self.use_gpu_checkbox.isChecked(),
            apply_clahe=self.apply_clahe_checkbox.isChecked(),
            focus_mode=self.focus_mode_combo.currentText(),
            phenotype_config=self.phenotype_config_edit.text().strip() or None,
            phenotype_engine=self.phenotype_engine_combo.currentText(),
            marker_metrics=self.marker_metrics_checkbox.isChecked(),
            interaction_metrics=self.interaction_metrics_checkbox.isChecked(),
            spatial_stats=self.spatial_stats_checkbox.isChecked(),
            register_retina=self.register_retina_checkbox.isChecked(),
            region_schema=self.region_schema_combo.currentText(),
            onh_mode=self.onh_mode_combo.currentText(),
            onh_xy=onh_xy,
            dorsal_xy=dorsal_xy,
            retina_frame_path=retina_frame_path,
            apply_edits=edit_log_path,
            save_debug=self.save_debug_checkbox.isChecked(),
            write_html_report=self.write_report_checkbox.isChecked(),
            write_object_table=self.write_object_table_checkbox.isChecked(),
            write_provenance=self.write_provenance_checkbox.isChecked(),
            write_uncertainty_maps=self.write_uncertainty_maps_checkbox.isChecked(),
            write_qc_maps=self.write_qc_maps_checkbox.isChecked(),
            strict_schemas=self.strict_schemas_checkbox.isChecked(),
        )

    def capture_landmarks_from_layer(self) -> None:
        layer_name = self.landmark_layer_combo.currentText().strip()
        if not layer_name:
            self.status_label.setText("Choose a points layer with ONH then dorsal landmarks.")
            return
        layer = self._get_layer(layer_name)
        if layer is None:
            self.status_label.setText("Landmarks layer not found.")
            return
        try:
            landmarks = landmarks_from_points(layer.data)
        except Exception as exc:
            self.status_label.setText(str(exc))
            return
        self.onh_xy_edit.setText(format_xy_text(landmarks["onh_xy"]))
        self.dorsal_xy_edit.setText(format_xy_text(landmarks["dorsal_xy"]))
        self.status_label.setText("Loaded ONH and dorsal landmarks from points layer.")

    def _upsert_labels_layer(self, name: str, data: np.ndarray, *, opacity: float = 0.6) -> None:
        layer = self._get_layer(name)
        if layer is not None:
            layer.data = data
            layer.opacity = opacity
            return
        self.viewer.add_labels(data, name=name, opacity=opacity)

    def _upsert_points_layer(self, name: str, data: np.ndarray, features=None) -> None:
        layer = self._get_layer(name)
        if layer is not None:
            layer.data = data
            if features is not None:
                layer.features = features
            return
        self.viewer.add_points(data, name=name, size=6, face_color="cyan", edge_color="black", features=features)

    def _selected_object_ids(self) -> list[int]:
        layer = getattr(self.viewer.layers.selection, "active", None)
        if layer is None or getattr(layer, "_type_string", "") != "points":
            raise ValueError("Select points from the centroid layer first.")
        features = getattr(layer, "features", None)
        if features is None or "object_id" not in features:
            raise ValueError("Selected points layer does not contain object_id features.")
        selected = sorted(int(index) for index in getattr(layer, "selected_data", set()))
        if not selected:
            raise ValueError("Select one or more centroid points first.")
        return [int(features.iloc[index]["object_id"]) for index in selected]

    def _append_edit(self, edit: dict[str, object], message: str) -> None:
        self._pending_edits.append(edit)
        self.status_label.setText(message)
        self.summary.setPlainText("\n".join([summarize_context(self._last_ctx)] + ["", "Pending edits:"] + [str(item) for item in self._pending_edits]) if self._last_ctx is not None else "\n".join(str(item) for item in self._pending_edits))

    def delete_selected_objects(self) -> None:
        try:
            object_ids = self._selected_object_ids()
        except Exception as exc:
            self.status_label.setText(str(exc))
            return
        for object_id in object_ids:
            self._append_edit({"op": "delete_object", "object_id": int(object_id)}, f"Queued deletion for {len(object_ids)} object(s).")

    def merge_selected_objects(self) -> None:
        try:
            object_ids = self._selected_object_ids()
        except Exception as exc:
            self.status_label.setText(str(exc))
            return
        if len(object_ids) < 2:
            self.status_label.setText("Select at least two centroid points to merge.")
            return
        keep_id = object_ids[0]
        self._append_edit(
            {"op": "merge_objects", "keep_object_id": int(keep_id), "object_ids": [int(value) for value in object_ids]},
            f"Queued merge into object {keep_id}.",
        )

    def relabel_selected_objects(self) -> None:
        phenotype = self.phenotype_relabel_edit.text().strip()
        if not phenotype:
            self.status_label.setText("Enter a phenotype label first.")
            return
        try:
            object_ids = self._selected_object_ids()
        except Exception as exc:
            self.status_label.setText(str(exc))
            return
        for object_id in object_ids:
            self._append_edit(
                {"op": "relabel_phenotype", "object_id": int(object_id), "phenotype": phenotype},
                f"Queued phenotype relabel to '{phenotype}'.",
            )

    def save_pending_edits(self) -> None:
        if self._last_ctx is None:
            self.status_label.setText("Run an image before saving edits.")
            return
        edit_path_text = self.edit_log_path_edit.text().strip()
        edit_path = Path(edit_path_text) if edit_path_text else default_edit_log_path(self._last_ctx.path)
        edits = list(self._pending_edits)
        onh_text = self.onh_xy_edit.text().strip()
        dorsal_text = self.dorsal_xy_edit.text().strip()
        if onh_text and dorsal_text:
            edits.append(
                {
                    "op": "set_landmarks",
                    "onh_xy": list(parse_xy_text(onh_text)),
                    "dorsal_xy": list(parse_xy_text(dorsal_text)),
                }
            )
        document = {
            "image_id": self._last_ctx.path.name.rsplit(".", 1)[0],
            "edits": edits,
        }
        try:
            saved = save_edit_log(document, edit_path)
        except Exception as exc:
            self.status_label.setText(f"Failed to save edits: {exc}")
            return
        self.edit_log_path_edit.setText(str(saved))
        self.status_label.setText(f"Saved review edits to {saved}")

    def _display_context(self, image_layer_name: str, ctx) -> None:
        if ctx.labels is not None:
            self._upsert_labels_layer(f"{image_layer_name} [RGC labels]", ctx.labels, opacity=0.55)
        if ctx.qc_mask is not None:
            self._upsert_labels_layer(f"{image_layer_name} [Focus mask]", ctx.qc_mask.astype(np.uint8), opacity=0.18)
        if ctx.object_table is not None and not ctx.object_table.empty:
            points = ctx.object_table[["centroid_y_px", "centroid_x_px"]].to_numpy(dtype=float)
            features = ctx.object_table.copy()
            self._upsert_points_layer(f"{image_layer_name} [RGC centroids]", points, features=features)

    def run_pipeline(self) -> None:
        image_layer_name = self.image_layer_combo.currentText().strip()
        if not image_layer_name:
            self.status_label.setText("Choose an image layer first.")
            return

        image_layer = self._get_layer(image_layer_name)
        if image_layer is None:
            self.status_label.setText("Image layer not found.")
            return

        try:
            options = self._collect_options()
            runtime = build_runtime(options)
            ctx = run_array(
                runtime,
                image=np.asarray(image_layer.data),
                source_path=self._get_source_path(image_layer),
                meta={"reader": "napari", "layer_name": image_layer_name},
            )
        except Exception as exc:
            self.status_label.setText(f"Run failed: {exc}")
            self.summary.setPlainText(str(exc))
            return

        self._last_runtime = runtime
        self._last_ctx = ctx
        if not self.edit_log_path_edit.text().strip():
            self.edit_log_path_edit.setText(str(default_edit_log_path(ctx.path)))
        self._display_context(image_layer_name, ctx)
        self.status_label.setText("Run completed.")
        self.summary.setPlainText(summarize_context(ctx))

    def export_latest_run(self) -> None:
        if self._last_ctx is None or self._last_runtime is None:
            self.status_label.setText("Run an image before exporting.")
            return

        output_dir = self.output_dir_edit.text().strip()
        if not output_dir:
            self.status_label.setText("Choose an output directory.")
            return

        try:
            artifacts = export_context(self._last_runtime, self._last_ctx, output_dir)
        except Exception as exc:
            self.status_label.setText(f"Export failed: {exc}")
            self.summary.setPlainText(str(exc))
            return

        lines = [summarize_context(self._last_ctx), "", "Artifacts:"]
        lines.extend(f"- {key}: {value}" for key, value in sorted(artifacts.items()))
        self.summary.setPlainText("\n".join(lines))
        self.status_label.setText("Export completed.")
