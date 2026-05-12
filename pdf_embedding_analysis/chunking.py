"""Fixed-size character chunking."""

from __future__ import annotations


def chunk_text(text: str, max_chars: int = 256, *, debug: bool = False) -> list[str]:
    """Split ``text`` into non-overlapping segments of at most ``max_chars`` characters.

    Whitespace-only segments are dropped.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    chunks: list[str] = []
    for start in range(0, len(text), max_chars):
        piece = text[start : start + max_chars]
        if piece.strip():
            chunks.append(piece)
    if debug:
        print(f"[chunk_text] Produced {len(chunks)} chunk(s) (max_chars={max_chars})")
    return chunks
