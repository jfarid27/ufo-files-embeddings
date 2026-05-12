"""Persist and restore embedding matrices and sidecar JSON metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pdf_embedding_analysis.types import ChunkMetadata, PathsBundle


def _parse_chunk_metadata_list(raw: object) -> list[ChunkMetadata]:
    if not isinstance(raw, list):
        raise ValueError("metadata.json must contain a JSON array")
    rows: list[ChunkMetadata] = []
    required = {"source_pdf", "chunk_index", "text", "char_length"}
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"metadata row {i} must be an object")
        keys = set(item.keys())
        if not required.issubset(keys):
            missing = required - keys
            raise ValueError(f"metadata row {i} missing keys: {sorted(missing)}")
        row: ChunkMetadata = {
            "source_pdf": str(item["source_pdf"]),
            "chunk_index": int(item["chunk_index"]),
            "text": str(item["text"]),
            "char_length": int(item["char_length"]),
        }
        if "token_count" in item:
            row["token_count"] = int(item["token_count"])
        rows.append(row)
    return rows


def ensure_embedding_dir(embedding_dir: Path, *, debug: bool = False) -> None:
    """Create the storage directory if missing."""
    embedding_dir.mkdir(parents=True, exist_ok=True)
    if debug:
        print(f"[ensure_embedding_dir] Ready: {embedding_dir}")


def save_embeddings_bundle(
    embeddings: NDArray[np.floating],
    metadata: list[ChunkMetadata],
    paths: PathsBundle,
    *,
    debug: bool = False,
) -> None:
    """Write ``embeddings.npy`` and ``metadata.json``."""
    np.save(paths["embeddings_path"], embeddings)
    paths["metadata_path"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if debug:
        print(
            f"[save_embeddings_bundle] Wrote {paths['embeddings_path']} "
            f"and {paths['metadata_path']} (n={embeddings.shape[0]})"
        )


def try_load_embeddings_bundle(
    paths: PathsBundle,
    *,
    debug: bool = False,
) -> tuple[NDArray[np.float64], list[ChunkMetadata]] | None:
    """Load embeddings + metadata if both files exist and are consistent."""
    emb_path, meta_path = paths["embeddings_path"], paths["metadata_path"]
    if not (emb_path.is_file() and meta_path.is_file()):
        if debug:
            print("[try_load_embeddings_bundle] Missing embeddings or metadata file; will rebuild")
        return None
    embeddings = np.load(emb_path)
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata = _parse_chunk_metadata_list(raw)
    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"Row count mismatch: embeddings {embeddings.shape[0]} vs metadata {len(metadata)}"
        )
    if debug:
        print(f"[try_load_embeddings_bundle] Loaded {len(metadata)} rows from disk")
    return np.asarray(embeddings, dtype=np.float64), metadata


def save_tsne_coordinates(coords: NDArray[np.floating], paths: PathsBundle, *, debug: bool = False) -> None:
    """Persist 3-D t-SNE coordinates aligned with embedding rows."""
    np.save(paths["tsne_path"], coords)
    if debug:
        print(f"[save_tsne_coordinates] Wrote {paths['tsne_path']} shape={coords.shape}")


def try_load_tsne_coordinates(paths: PathsBundle, *, debug: bool = False) -> NDArray[np.float64] | None:
    """Return stored t-SNE coordinates, or ``None`` if absent."""
    if not paths["tsne_path"].is_file():
        if debug:
            print("[try_load_tsne_coordinates] No t-SNE file on disk")
        return None
    arr = np.load(paths["tsne_path"])
    if debug:
        print(f"[try_load_tsne_coordinates] Loaded shape={arr.shape}")
    return np.asarray(arr, dtype=np.float64)
