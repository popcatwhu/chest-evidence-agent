"""Freeze image-disjoint MedXpertQA chest-radiograph development and holdout sets.

Image modality approval is external to this script and never uses reference answers.
Question IDs cannot guarantee patient disjointness; this is a cross-dataset MCQ pilot.
"""

import argparse
import hashlib
import io
import json
import random
import re
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def pixel_hash(image):
    rgb = image.convert("RGB")
    return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()


def freeze(path, data):
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError(f"Refusing to replace a frozen split: {path}")
    path.write_text(text)


def prepare(source, audit_path, destination, dev=12, holdout=24, seed=20261014):
    audit = json.loads(audit_path.read_text())
    revision = json.loads((source / "source_revision.json").read_text())
    if revision["revision"] != audit["source_revision"]:
        raise ValueError("Intake audit and source revision differ")
    rows = {
        str(r["id"]): r
        for r in map(json.loads, (source / "test.jsonl").read_text().splitlines())
    }
    # Exclude exact pixel duplicates of available prior PNG inputs in these evaluation directories.
    prior_pixels = set()
    for folder in [
        "quality",
        "independent",
        "medical_test",
        "quality_v3/test20",
        "quality_v4/test20",
        "team/test20",
        "team60/test60",
    ]:
        for path in (ROOT / "data" / folder).glob("*.png"):
            with Image.open(path) as image:
                prior_pixels.add(pixel_hash(image))
    candidates, dropped = {}, []
    with zipfile.ZipFile(source / "images.zip") as archive:
        names = {Path(name).name: name for name in archive.namelist()}
        for identifier in audit["approved_ids"]:
            row = rows[identifier]
            if row["medical_task"] != "Diagnosis" or len(row["images"]) != 1:
                raise ValueError("Approved input violates task/image contract")
            with Image.open(io.BytesIO(archive.read(names[row["images"][0]]))) as image:
                rgb = image.convert("RGB")
                digest = pixel_hash(rgb)
                if digest in prior_pixels or digest in candidates:
                    dropped.append(identifier)
                    continue
                candidates[digest] = (row, rgb.copy())
        hashes = sorted(candidates)
        random.Random(seed).shuffle(hashes)
        if dev + holdout > len(hashes):
            raise ValueError("Not enough approved unique images for disjoint splits")
        summaries = {}
        for split, selected in [
            ("development", hashes[:dev]),
            ("holdout", hashes[dev : dev + holdout]),
        ]:
            folder = destination / split
            folder.mkdir(parents=True, exist_ok=True)
            examples = []
            for digest in selected:
                row, image = candidates[digest]
                identifier = str(row["id"])
                path = folder / f"{identifier}.png"
                image.save(path)
                stem = re.split(
                    r"\bAnswer Choices\s*:", row["question"], maxsplit=1, flags=re.I
                )[0].strip()
                question = (
                    stem
                    + "\n"
                    + "\n".join(
                        f"{letter}. {value}" for letter, value in row["options"].items()
                    )
                )
                examples.append(
                    {
                        "case_id": identifier,
                        "question_id": identifier,
                        "title": "MedXpertQA " + identifier,
                        "context": "Published clinical vignette and task (task sentences are instructions, not confirmed patient findings):\n"
                        + stem,
                        "question": question,
                        "image_path": str(path.relative_to(ROOT / "data")),
                        "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "pixel_sha256": digest,
                        "reference_answer": row["label"],
                        "options": row["options"],
                        "categories": row["medical_task"],
                        "original_image": row["images"][0],
                    }
                )
            manifest = {
                "task": "multiple_choice",
                "split": split,
                "seed": seed,
                "source": audit["source"],
                "source_revision": audit["source_revision"],
                "metadata_sha256": hashlib.sha256(
                    (source / "test.jsonl").read_bytes()
                ).hexdigest(),
                "intake_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                "selection": "Seeded shuffle of approved, unique chest-radiograph pixel hashes; reference answers unused for selection",
                "limitations": [
                    "Image disjointness does not establish patient disjointness.",
                    "Public exam questions may have pretraining contamination.",
                    "Different source and task distribution from frozen ChestAgentBench; do not merge scores.",
                    "Modality inspection does not establish clinical answerability or label correctness.",
                ],
                "examples": examples,
            }
            freeze(folder / "manifest.json", manifest)
            summaries[split] = {
                "count": len(examples),
                "case_ids": [r["case_id"] for r in examples],
                "manifest_sha256": hashlib.sha256(
                    (folder / "manifest.json").read_bytes()
                ).hexdigest(),
                "pixel_sha256": [r["pixel_sha256"] for r in examples],
            }
    summary = {
        "source": audit["source"],
        "source_revision": audit["source_revision"],
        "seed": seed,
        "approved_images": len(audit["approved_ids"]),
        "unique_eligible_images": len(candidates),
        "duplicate_images_excluded": dropped,
        "splits": summaries,
        "holdout_evaluated": False,
        "limitations": manifest["limitations"],
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/medxpert/source")
    parser.add_argument(
        "--audit", type=Path, default=ROOT / "evals/medxpert-intake.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/medxpert")
    parser.add_argument("--dev", type=int, default=12)
    parser.add_argument("--holdout", type=int, default=24)
    args = parser.parse_args()
    summary = prepare(args.source, args.audit, args.output, args.dev, args.holdout)
    freeze(ROOT / "evals/medxpert-splits.json", summary)
    print(json.dumps({key: value["count"] for key, value in summary["splits"].items()}))
