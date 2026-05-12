"""Embedding generation.

``tiktoken`` does not compute dense embeddings; it tokenizes text. Dense
vectors use FastEmbed (ONNX via ``onnxruntime``) through LangChain—no PyTorch.
Optionally report per-chunk token counts with ``tiktoken`` when ``debug`` is true.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import tiktoken
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from numpy.typing import NDArray

from pdf_embedding_analysis.types import ChunkMetadata


class EmbeddingModel(Protocol):
    """Protocol for pluggable dense embedding backends."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


def _token_counts_for_chunks(chunks: list[str], *, debug: bool) -> list[int]:
    """Use ``cl100k_base`` (GPT-4 family) for reproducible token counts."""
    enc = tiktoken.get_encoding("cl100k_base")
    counts = [len(enc.encode(c)) for c in chunks]
    if debug:
        for i, (c, n) in enumerate(zip(chunks, counts)):
            preview = c.replace("\n", " ")[:80]
            print(f"  chunk {i}: tokens={n} chars={len(c)} preview={preview!r}")
    return counts


def build_default_embedder(
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> FastEmbedEmbeddings:
    """Create a local ONNX sentence embedding model (FastEmbed / no PyTorch)."""
    return FastEmbedEmbeddings(model_name=model_name)


def embed_chunks(
    chunks: list[str],
    *,
    embedder: EmbeddingModel | None = None,
    debug: bool = False,
) -> NDArray[np.float64]:
    """Embed each chunk; returns shape ``(n_chunks, dim)``."""
    if not chunks:
        raise ValueError("chunks must be non-empty")
    model = embedder or build_default_embedder()
    if debug:
        print(f"[embed_chunks] Embedding {len(chunks)} chunk(s)")
    vectors_list = model.embed_documents(chunks)
    arr = np.asarray(vectors_list, dtype=np.float64)
    if arr.ndim != 2:
        raise RuntimeError(f"Expected 2D embedding matrix, got shape {arr.shape}")
    if debug:
        print(f"[embed_chunks] Embedding matrix shape: {arr.shape}")
    return arr


def attach_token_counts(metadata: list[ChunkMetadata], *, debug: bool = False) -> None:
    """Mutate metadata dicts in-place with ``token_count`` using tiktoken."""
    texts = [m["text"] for m in metadata]
    counts = _token_counts_for_chunks(texts, debug=debug)
    for row, n in zip(metadata, counts):
        row["token_count"] = n
