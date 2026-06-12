from dataclasses import dataclass

@dataclass
class Chunk:
    index: int
    text: str
    char_start: int
    char_end: int

def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """
    Split text into overlapping chunks.

    Strategy:
      - Fixed-size chunks with overlap.
      - Prefer breaking on whitespace/newlines.
      - Only search for breakpoints near the chunk boundary.
      - Avoid tiny chunks.
      - Avoid tail-chunk explosion bug.

    Args:
        text: Full document text.
        chunk_size: Maximum chunk size in characters.
        overlap: Overlap between chunks.

    Returns:
        List[Chunk]
    """
    if chunk_size <= 0:
        raise ValueError(
            f"chunk_size must be positive, got {chunk_size}"
        )

    if overlap < 0:
        raise ValueError(
            f"overlap must be >= 0, got {overlap}"
        )

    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    if not text.strip():
        return []

    chunks: list[Chunk] = []

    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            search_start = max(start, end - 100)

            newline = text.rfind("\n", search_start, end)
            space = text.rfind(" ", search_start, end)

            break_at = max(newline, space)

            if break_at > start:
                end = break_at

        chunk_text_content = text[start:end].strip()

        if chunk_text_content:
            chunks.append(
                Chunk(
                    index=index,
                    text=chunk_text_content,
                    char_start=start,
                    char_end=end,
                )
            )
            index += 1

        if end == len(text):
            break

        next_start = end - overlap

        if next_start <= start:
            next_start = start + 1

        start = next_start

    return chunks