"""High-level controller for notebook-driven exploration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from pdf_embedding_analysis.corpus import build_chunk_metadata_from_pdf_folder
from pdf_embedding_analysis.embedding import (
    EmbeddingModel,
    attach_token_counts,
    build_default_embedder,
    embed_chunks,
)
from pdf_embedding_analysis.storage import (
    ensure_embedding_dir,
    save_embeddings_bundle,
    save_tsne_coordinates,
    try_load_embeddings_bundle,
    try_load_tsne_coordinates,
)
from pdf_embedding_analysis.tsne_step import compute_tsne_3d
from pdf_embedding_analysis.types import ChunkMetadata, default_storage_paths
from pdf_embedding_analysis.visualization import plot_tsne_3d_by_filename, plot_tsne_scatter_3d


def _short_label_for_chunk(meta: ChunkMetadata) -> str:
    return f"{Path(meta['source_pdf']).name}#{meta['chunk_index']}"


class PdfEmbeddingAnalysis:
    """Run PDF ingestion → embeddings → t-SNE → Plotly with cached artifacts.

    Delete files under ``embedding_storage`` (or only ``tsne_3d.npy``) to force
    recomputation after adding PDFs.
    """

    def __init__(
        self,
        pdf_folder: str | Path,
        embedding_storage: str | Path,
        *,
        max_chunk_chars: int = 256,
        embedder: EmbeddingModel | None = None,
        tsne_random_state: int = 42,
        tsne_perplexity: float | None = None,
        tsne_max_iter: int = 1000,
        debug: bool = False,
    ) -> None:
        self.pdf_folder = Path(pdf_folder).expanduser().resolve()
        self.embedding_storage = Path(embedding_storage).expanduser().resolve()
        self.max_chunk_chars = max_chunk_chars
        self._embedder: EmbeddingModel = embedder if embedder is not None else build_default_embedder()
        self.tsne_random_state = tsne_random_state
        self.tsne_perplexity = tsne_perplexity
        self.tsne_max_iter = tsne_max_iter
        self.debug = debug

        self._paths = default_storage_paths(self.embedding_storage)
        self._metadata: list[ChunkMetadata] | None = None
        self._embeddings: NDArray[np.float64] | None = None
        self._tsne_3d: NDArray[np.float64] | None = None

    def run_analysis(self) -> None:
        """Load cached artifacts when present; otherwise compute and persist."""
        ensure_embedding_dir(self.embedding_storage, debug=self.debug)

        loaded = try_load_embeddings_bundle(self._paths, debug=self.debug)
        if loaded is None:
            self._build_embeddings_from_pdfs()
        else:
            self._embeddings, self._metadata = loaded
            if self.debug:
                print("[PdfEmbeddingAnalysis.run_analysis] Using cached embeddings + metadata")

        tsne = try_load_tsne_coordinates(self._paths, debug=self.debug)
        if tsne is None:
            self._compute_and_save_tsne()
        else:
            self._tsne_3d = tsne
            if self.debug:
                print("[PdfEmbeddingAnalysis.run_analysis] Using cached t-SNE coordinates")

    def _build_embeddings_from_pdfs(self) -> None:
        metadata = build_chunk_metadata_from_pdf_folder(
            self.pdf_folder,
            self.max_chunk_chars,
            debug=self.debug,
        )
        if not metadata:
            raise ValueError(
                f"No text chunks produced from PDFs in {self.pdf_folder}. "
                "Add non-empty .pdf files or check extraction."
            )
        attach_token_counts(metadata, debug=self.debug)
        texts = [row["text"] for row in metadata]
        embeddings = embed_chunks(texts, embedder=self._embedder, debug=self.debug)
        save_embeddings_bundle(embeddings, metadata, self._paths, debug=self.debug)
        self._metadata = metadata
        self._embeddings = embeddings

    def _compute_and_save_tsne(self) -> None:
        if self._embeddings is None:
            raise RuntimeError("Embeddings are not loaded")
        if self._embeddings.shape[0] < 2:
            raise ValueError(
                "t-SNE requires at least two chunks. Add more PDF text or lower chunk size."
            )
        coords = compute_tsne_3d(
            self._embeddings,
            random_state=self.tsne_random_state,
            perplexity=self.tsne_perplexity,
            max_iter=self.tsne_max_iter,
            debug=self.debug,
        )
        save_tsne_coordinates(coords, self._paths, debug=self.debug)
        self._tsne_3d = coords

    @property
    def metadata(self) -> list[ChunkMetadata]:
        """Chunk metadata rows (must call ``run_analysis`` first)."""
        if self._metadata is None:
            raise RuntimeError("Call run_analysis() before accessing metadata")
        return self._metadata

    @property
    def embeddings(self) -> NDArray[np.float64]:
        """Dense embedding matrix ``(n_chunks, dim)``."""
        if self._embeddings is None:
            raise RuntimeError("Call run_analysis() before accessing embeddings")
        return self._embeddings

    @property
    def tsne_3d(self) -> NDArray[np.float64]:
        """t-SNE coordinates ``(n_chunks, 3)``."""
        if self._tsne_3d is None:
            raise RuntimeError("Call run_analysis() before accessing tsne_3d")
        return self._tsne_3d

    def plot_tsne_scatter_3d(
        self,
        *,
        hover_max_chars: int = 220,
        title: str | None = None,
        marker_size: int = 5,
        opacity: float = 0.85,
    ) -> go.Figure:
        """Return a Plotly ``scatter_3d`` figure for the current t-SNE projection."""
        coords = self.tsne_3d
        meta = self.metadata
        labels = [_short_label_for_chunk(m) for m in meta]
        hovers: list[str] = []
        for m in meta:
            snippet = m["text"].replace("\n", " ")
            if len(snippet) > hover_max_chars:
                snippet = snippet[: hover_max_chars - 1] + "…"
            pdf_name = Path(m["source_pdf"]).name
            extra = ""
            if "token_count" in m:
                extra = f" tokens={m['token_count']}"
            hovers.append(f"{pdf_name} chunk {m['chunk_index']}{extra}<br>{snippet}")
        return plot_tsne_scatter_3d(
            coords,
            labels=labels,
            hover_texts=hovers,
            title=title or "t-SNE of PDF chunks (3D)",
            marker_size=marker_size,
            opacity=opacity,
            debug=self.debug,
        )

    def plot_tsne_by_filename(
        self,
        *,
        hover_max_chars: int = 220,
        title: str | None = None,
        marker_size: int = 5,
        opacity: float = 0.85,
        show_legend: bool = True,
    ) -> go.Figure:
        """3-D t-SNE plot with **one series per PDF**; legend entries are filenames."""
        coords = self.tsne_3d
        meta = self.metadata
        source_paths = [m["source_pdf"] for m in meta]
        hover_lines: list[str] = []
        for m in meta:
            snippet = m["text"].replace("\n", " ")
            if len(snippet) > hover_max_chars:
                snippet = snippet[: hover_max_chars - 1] + "…"
            extra = ""
            if "token_count" in m:
                extra = f" tokens={m['token_count']}"
            hover_lines.append(f"chunk {m['chunk_index']}{extra}<br>{snippet}")
        return plot_tsne_3d_by_filename(
            coords,
            source_paths,
            hover_texts=hover_lines,
            title=title or "t-SNE by PDF filename (3D)",
            marker_size=marker_size,
            opacity=opacity,
            show_legend=show_legend,
            debug=self.debug,
        )
