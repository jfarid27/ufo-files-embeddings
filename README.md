# aweeumz

PDF text extraction, **FastEmbed** dense embeddings (ONNX, no PyTorch), local caching, **t-SNE** in 3D, and **Plotly** exploration—wired together through a small Python package and a notebook-friendly controller.

## Requirements

- Python 3.10+
- [Poetry](https://python-poetry.org/) 2.x

## Install

From the repository root:

```bash
POETRY_KEYRING_ENABLED=false poetry install
```

On some Linux setups Poetry tries to use the system keyring over DBus; setting `POETRY_KEYRING_ENABLED=false` avoids failures when no keyring is available.

## Quick start (terminal)

```bash
POETRY_KEYRING_ENABLED=false poetry run python -c "
from pathlib import Path
from pdf_embedding_analysis import PdfEmbeddingAnalysis

pdf_dir = Path('data/Release_1')          # folder containing .pdf files (non-recursive)
store = Path('data/Release_1_embeddings') # cache directory for vectors + t-SNE

analysis = PdfEmbeddingAnalysis(pdf_dir, store, debug=False)
analysis.run_analysis()
print(analysis.embeddings.shape, analysis.tsne_3d.shape)
"
```

## Jupyter Notebook

1. **Kernel**: use the project virtualenv so imports resolve.

   ```bash
   POETRY_KEYRING_ENABLED=false poetry run python -m ipykernel install --user --name=aweeumz
   ```

   In Jupyter, choose the kernel **aweeumz** (or select the interpreter at `.venv/bin/python`).

2. **Plotly in the notebook** (recommended once per session):

   ```python
   import plotly.io as pio
   pio.renderers.default = "notebook_connected"  # or "browser", "iframe_connected"
   ```

3. **Run the controller** (minimal example):

   ```python
   from pathlib import Path
   from pdf_embedding_analysis import PdfEmbeddingAnalysis

   PDF_DIR = Path("data/Release_1")
   EMBED_DIR = Path("data/Release_1_embeddings")

   analysis = PdfEmbeddingAnalysis(PDF_DIR, EMBED_DIR, debug=False)
   analysis.run_analysis()

   analysis.embeddings.shape   # (n_chunks, embedding_dim)
   analysis.tsne_3d.shape    # (n_chunks, 3)
   len(analysis.metadata)    # same as n_chunks
   ```

4. **Plots**:

   ```python
   # One trace per PDF filename (legend = files)
   fig_files = analysis.plot_tsne_by_filename()
   fig_files.show()

   # Single trace; color by chunk id; rich hover
   fig_chunks = analysis.plot_tsne_scatter_3d()
   fig_chunks.show()
   ```

A worked notebook with the same steps lives at [`notebooks/controller_demo.ipynb`](notebooks/controller_demo.ipynb).

## Cached artifacts

With `embedding_storage` set to e.g. `data/Release_1_embeddings`, the controller writes:

| File | Contents |
|------|-----------|
| `embeddings.npy` | `float64` array, shape `(n_chunks, dim)` |
| `metadata.json` | Per-chunk rows: source path, chunk index, text, lengths, optional `token_count` |
| `tsne_3d.npy` | `float64` array, shape `(n_chunks, 3)` |

- **Rebuild embeddings** after adding or changing PDFs: delete `embeddings.npy` and `metadata.json` (or the whole folder).
- **Rebuild t-SNE only**: delete `tsne_3d.npy` while keeping embeddings.

## Configuration

`PdfEmbeddingAnalysis` constructor (high level):

| Argument | Role |
|----------|------|
| `pdf_folder` | Directory whose **immediate** children are `.pdf` files |
| `embedding_storage` | Directory for cached `.npy` / `.json` |
| `max_chunk_chars` | Maximum characters per chunk (default `256`) |
| `embedder` | Optional object implementing `embed_documents(list[str]) -> list[list[float]]` |
| `tsne_random_state`, `tsne_perplexity`, `tsne_max_iter` | Passed to scikit-learn `TSNE` |
| `debug` | If `True`, pipeline steps print diagnostic messages |

## Design notes

- **Embeddings**: [FastEmbed](https://qdrant.github.io/fastembed/) via LangChain’s `FastEmbedEmbeddings` (default model `BAAI/bge-small-en-v1.5`). **tiktoken** is used for optional token-count metadata, not for producing vectors.
- **Composable functions**: lower-level steps live under `pdf_embedding_analysis/` (PDF I/O, chunking, embedding, storage, t-SNE, Plotly). See [docs/architecture.md](docs/architecture.md).

## Type checking

```bash
POETRY_KEYRING_ENABLED=false poetry run mypy pdf_embedding_analysis
```

## License

Add a license file if you distribute this project.
