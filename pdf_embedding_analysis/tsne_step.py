"""Dimensionality reduction with scikit-learn t-SNE."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.manifold import TSNE


def compute_tsne_3d(
    embeddings: NDArray[np.floating],
    *,
    random_state: int = 42,
    perplexity: float | None = None,
    max_iter: int = 1000,
    learning_rate: str | float = "auto",
    debug: bool = False,
) -> NDArray[np.float64]:
    """Project high-dimensional embeddings to three dimensions."""
    n_samples = int(embeddings.shape[0])
    if n_samples < 2:
        raise ValueError("t-SNE requires at least two samples")
    n_features = int(embeddings.shape[1])
    if n_features < 1:
        raise ValueError("Embeddings must have at least one dimension")

    perp = float(perplexity) if perplexity is not None else float(min(30, max(5, n_samples // 4)))
    # sklearn requires perplexity < n_samples
    perp = min(perp, max(1.0, (n_samples - 1) / 3.0))

    if debug:
        print(
            f"[compute_tsne_3d] n_samples={n_samples} n_features={n_features} "
            f"perplexity={perp:.3f} max_iter={max_iter}"
        )

    tsne = TSNE(
        n_components=3,
        perplexity=perp,
        random_state=random_state,
        max_iter=max_iter,
        learning_rate=learning_rate,
        init="pca",
        verbose=2 if debug else 0,
    )
    coords = tsne.fit_transform(np.asarray(embeddings, dtype=np.float64))
    return np.asarray(coords, dtype=np.float64)
