from pathlib import Path

from src.manifest import load_manifest


def test_load_manifest_resolves_relative_paths(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                "sample_id,animal_id,eye,condition,genotype,timepoint_dpi,modality,stain_panel,path,label_path",
                "S1,A1,OD,treated,WT,7,flatmount,RBPMS,images/sample.tif,labels/sample_labels.tif",
            ]
        ),
        encoding="utf-8",
    )

    df = load_manifest(manifest)

    assert df.loc[0, "path"] == str((images_dir / "sample.tif").resolve())
    assert df.loc[0, "label_path"] == str((labels_dir / "sample_labels.tif").resolve())
