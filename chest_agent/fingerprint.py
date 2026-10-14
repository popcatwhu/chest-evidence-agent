"""Fingerprint local pipeline sources to identify the running evaluation server."""

import hashlib
from pathlib import Path


def source_fingerprint():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
