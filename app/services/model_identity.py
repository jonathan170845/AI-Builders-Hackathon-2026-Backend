"""Identify a local model by content, without inventing a Hub revision."""

import hashlib
from pathlib import Path

from app.services.artifacts import file_sha256


def local_model_fingerprint(path: Path) -> str:
    files = sorted(
        p
        for p in path.rglob("*")
        if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".txt", ".model"}
    )
    if not files or not (path / "modules.json").is_file():
        raise ValueError("Expected a complete local SentenceTransformer model")
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(file_sha256(item).encode())
    return digest.hexdigest()
