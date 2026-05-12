"""Composable PDF → embeddings → t-SNE → Plotly pipeline.

Public API:

- :class:`~pdf_embedding_analysis.controller.PdfEmbeddingAnalysis` — cached pipeline
  and notebook helpers (see project ``README.md`` and ``notebooks/controller_demo.ipynb``).
- :func:`~pdf_embedding_analysis.visualization.plot_tsne_3d_by_filename` — Plotly 3D
  scatter grouped by source filename.

Lower-level steps (PDF I/O, chunking, storage, t-SNE) live in sibling modules; see
``docs/architecture.md``.
"""

from pdf_embedding_analysis.controller import PdfEmbeddingAnalysis
from pdf_embedding_analysis.visualization import plot_tsne_3d_by_filename

__all__ = ["PdfEmbeddingAnalysis", "plot_tsne_3d_by_filename"]
