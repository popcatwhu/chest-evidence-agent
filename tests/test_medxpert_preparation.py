import json
import zipfile

from PIL import Image
import pytest

from scripts import prepare_medxpert as prep


def test_split_is_image_disjoint_and_selection_does_not_depend_on_labels(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(prep, "ROOT", tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    with zipfile.ZipFile(source / "images.zip", "w") as archive:
        for number in range(4):
            path = tmp_path / f"{number}.png"
            Image.new("RGB", (8, 8), (number * 40, 100, 50)).save(path)
            archive.write(path, f"images/{number}.png")
            rows.append(
                {
                    "id": f"MM-{number}",
                    "question": "Clinical vignette. What is the diagnosis? Answer Choices: (A) X (B) Y",
                    "medical_task": "Diagnosis",
                    "images": [path.name],
                    "options": {"A": "X", "B": "Y"},
                    "label": "A",
                }
            )
    (source / "source_revision.json").write_text(
        json.dumps({"revision": "test-revision"})
    )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "source": "https://example.org/source",
                "source_revision": "test-revision",
                "approved_ids": [r["id"] for r in rows],
            }
        )
    )
    metadata = source / "test.jsonl"
    metadata.write_text("\n".join(json.dumps(r) for r in rows))
    first = prep.prepare(source, audit, tmp_path / "data/first", dev=2, holdout=2)
    for row in rows:
        row["label"] = "B"
    metadata.write_text("\n".join(json.dumps(r) for r in rows))
    second = prep.prepare(source, audit, tmp_path / "data/second", dev=2, holdout=2)
    for split in ["development", "holdout"]:
        assert first["splits"][split]["case_ids"] == second["splits"][split]["case_ids"]
    assert not set(first["splits"]["development"]["pixel_sha256"]).intersection(
        first["splits"]["holdout"]["pixel_sha256"]
    )
    with pytest.raises(ValueError, match="frozen"):
        prep.prepare(source, audit, tmp_path / "data/first", dev=2, holdout=2)
