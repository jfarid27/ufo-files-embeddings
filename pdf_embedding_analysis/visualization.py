"""Plotly 3-D scatter for t-SNE coordinates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray


def plot_tsne_scatter_3d(
    coords: NDArray[np.floating],
    *,
    labels: Sequence[str],
    hover_texts: Sequence[str] | None = None,
    title: str = "t-SNE embedding space (3D)",
    marker_size: int = 5,
    opacity: float = 0.85,
    colorscale: str = "Turbo",
    debug: bool = False,
) -> go.Figure:
    """Build a ``scatter_3d`` figure from ``(n, 3)`` coordinates.

    ``labels`` are used for categorical coloring (e.g. source PDF name).
    """
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"coords must have shape (n, 3); got {arr.shape}")
    n = arr.shape[0]
    if len(labels) != n:
        raise ValueError(f"labels length {len(labels)} != n {n}")
    hovers: Sequence[str]
    if hover_texts is None:
        hovers = labels
    else:
        if len(hover_texts) != n:
            raise ValueError(f"hover_texts length {len(hover_texts)} != n {n}")
        hovers = hover_texts

    if debug:
        print(f"[plot_tsne_scatter_3d] Plotting n={n} points")

    unique_labels = list(dict.fromkeys(labels))
    label_to_index = {lab: idx for idx, lab in enumerate(unique_labels)}
    color_values = [float(label_to_index[lab]) for lab in labels]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=arr[:, 0],
                y=arr[:, 1],
                z=arr[:, 2],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    opacity=opacity,
                    color=color_values,
                    colorscale=colorscale,
                    showscale=True,
                    colorbar=dict(title="category id"),
                ),
                text=list(hovers),
                hovertemplate="%{text}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="t-SNE 1", yaxis_title="t-SNE 2", zaxis_title="t-SNE 3"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def plot_tsne_3d_by_filename(
    coords: NDArray[np.floating],
    source_paths: Sequence[str | Path],
    *,
    hover_texts: Sequence[str] | None = None,
    title: str = "t-SNE by source PDF (3D)",
    marker_size: int = 5,
    opacity: float = 0.85,
    show_legend: bool = True,
    debug: bool = False,
) -> go.Figure:
    """Plot t-SNE coordinates with **one trace per PDF** so the legend lists filenames.

    Each row of ``coords`` must align with the corresponding ``source_paths`` entry
    (typically one path per text chunk). Points from the same file share a color and
    trace name (the path's ``.name``). Use ``hover_texts`` for per-point detail
    (e.g. chunk index and snippet).
    """
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"coords must have shape (n, 3); got {arr.shape}")
    n = arr.shape[0]
    paths = [Path(str(p)) for p in source_paths]
    if len(paths) != n:
        raise ValueError(f"source_paths length {len(paths)} != n {n}")
    hovers: list[str | None]
    if hover_texts is None:
        hovers = [None] * n
    else:
        if len(hover_texts) != n:
            raise ValueError(f"hover_texts length {len(hover_texts)} != n {n}")
        hovers = [str(h) for h in hover_texts]

    # Group by basename; disambiguate duplicate basenames with parent folder name.
    basename_counts: dict[str, int] = defaultdict(int)
    for p in paths:
        basename_counts[p.name] += 1

    def legend_label(p: Path) -> str:
        if basename_counts[p.name] > 1:
            parent = p.parent.name or str(p.parent)
            return f"{parent}/{p.name}"
        return p.name

    by_label: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(paths):
        by_label[legend_label(p)].append(i)

    if debug:
        print(f"[plot_tsne_3d_by_filename] n={n} traces={len(by_label)}")

    traces: list[go.Scatter3d] = []
    for label in sorted(by_label.keys()):
        idx = by_label[label]
        xs = arr[idx, 0]
        ys = arr[idx, 1]
        zs = arr[idx, 2]
        point_text = [hovers[i] if hovers[i] is not None else label for i in idx]
        traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                name=label,
                legendgroup=label,
                marker=dict(size=marker_size, opacity=opacity),
                text=point_text,
                hovertemplate="<b>%{fullData.name}</b><br>%{text}<extra></extra>",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="t-SNE 1", yaxis_title="t-SNE 2", zaxis_title="t-SNE 3"),
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=show_legend,
        legend=dict(
            itemsizing="constant",
            tracegroupgap=2,
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(size=10),
            traceorder="normal",
        ),
    )
    return fig
