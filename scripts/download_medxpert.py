"""Download the audited MedXpertQA revision without changing existing source files."""

import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent


def download():
    audit = json.loads((ROOT / "evals/medxpert-intake.json").read_text())
    repository = "TsinghuaC3I/MedXpertQA"
    revision = audit["source_revision"]
    directory = ROOT / "data/medxpert/source"
    directory.mkdir(parents=True, exist_ok=True)
    revision_path = directory / "source_revision.json"
    if (
        revision_path.exists()
        and json.loads(revision_path.read_text())["revision"] != revision
    ):
        raise ValueError("Existing source directory contains another revision")
    response = requests.get(
        f"https://huggingface.co/api/datasets/{repository}/tree/{revision}?recursive=true",
        timeout=30,
    )
    response.raise_for_status()
    entries = {entry["path"]: entry for entry in response.json()}
    for name in ["MM/test.jsonl", "README.md", "images.zip"]:
        target = directory / Path(name).name
        if not target.exists():
            partial = target.with_suffix(target.suffix + ".part")
            with requests.get(
                f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{name}",
                stream=True,
                timeout=45,
            ) as response:
                response.raise_for_status()
                with partial.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
            partial.replace(target)
        expected = entries[name].get("lfs", {}).get("oid")
        if expected:
            digest = hashlib.sha256()
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(
                    "Archive checksum does not match pinned upstream source"
                )
        print(name, target.stat().st_size, flush=True)
    if not revision_path.exists():
        revision_path.write_text(
            json.dumps({"repository": repository, "revision": revision}, indent=2)
        )


if __name__ == "__main__":
    download()
