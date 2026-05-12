"""Shared typing helpers and structural types for the pipeline."""

from __future__ import annotations

from pathlib import Path

from typing_extensions import NotRequired, TypedDict

class ChunkMetadata(TypedDict):
    """Metadata for one text chunk extracted from a PDF."""

    source_pdf: str
    chunk_index: int
    text: str
    char_length: int
    token_count: NotRequired[int]


class PathsBundle(TypedDict):
    """Standard on-disk layout under the embedding storage directory."""

    embeddings_path: Path
    metadata_path: Path
    tsne_path: Path


def default_storage_paths(embedding_dir: Path) -> PathsBundle:
    """Return canonical filenames for persisted artifacts."""
    return PathsBundle(
        embeddings_path=embedding_dir / "embeddings.npy",
        metadata_path=embedding_dir / "metadata.json",
        tsne_path=embedding_dir / "tsne_3d.npy",
    )
