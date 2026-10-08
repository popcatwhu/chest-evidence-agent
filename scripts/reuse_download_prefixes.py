"""Reuse sequential Hub partial bytes by LFS identity; final SHA256 remains mandatory."""

import argparse
import json
import os
from pathlib import Path
import struct
from huggingface_hub import HfApi
from download_medical_models import MODELS, ROOT


def reuse(size):
    repo, revision = MODELS[size]
    root = ROOT.parent / "models" / repo.split("/")[-1]
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    total = 0
    chunk_size = 16 * 1024 * 1024
    for item in info.siblings:
        if not item.rfilename.endswith(".safetensors"):
            continue
        matches = list(
            (root / ".cache/huggingface/download").glob(
                f"*.{item.lfs.sha256}.incomplete"
            )
        )
        if not matches:
            continue
        source = matches[0]
        path = root / item.rfilename
        part = path.with_suffix(".range.part")
        progress = path.with_suffix(".range.json")
        if path.exists() or not part.exists():
            continue
        header = {
            "repository": repo,
            "revision": revision,
            "sha256": item.lfs.sha256,
            "size": item.size,
            "chunk_size": chunk_size,
        }
        saved = (
            json.loads(progress.read_text())
            if progress.exists()
            else {"header": header, "done": []}
        )
        if saved["header"] != header:
            raise ValueError("Download identity differs")
        done = set(saved["done"])
        reused = []
        with source.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if header_size > 10 * 1024 * 1024:
                raise ValueError("Not a safetensors prefix")
            keys = set(json.loads(stream.read(header_size))) - {"__metadata__"}
            if keys != {k for k, v in index.items() if v == item.rfilename}:
                raise ValueError("Cached prefix belongs to a different shard")
            fd = os.open(part, os.O_RDWR)
            try:
                for start in range(
                    0,
                    min(source.stat().st_size, item.size) // chunk_size * chunk_size,
                    chunk_size,
                ):
                    if start in done:
                        continue
                    stream.seek(start)
                    block = stream.read(chunk_size)
                    if len(block) != chunk_size:
                        raise ValueError("Cached prefix ended unexpectedly")
                    offset = start
                    view = memoryview(block)
                    while view:
                        n = os.pwrite(fd, view, offset)
                        if n <= 0:
                            raise RuntimeError("Failed write")
                        offset += n
                        view = view[n:]
                    done.add(start)
                    reused.append(start)
                    total += chunk_size
                os.fsync(fd)
            finally:
                os.close(fd)
        temp = progress.with_suffix(".tmp")
        temp.write_text(json.dumps({"header": header, "done": sorted(done)}))
        temp.replace(progress)
        if reused:
            path.with_suffix(".reused.json").write_text(
                json.dumps({"offsets": reused, "source": source.name})
            )
            print(
                "Reused",
                item.rfilename,
                len(reused) * 16,
                "MiB; pending full official SHA256 verification",
                flush=True,
            )
    print("Total reused MiB", total // 2**20, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("size", choices=MODELS)
    reuse(parser.parse_args().size)
