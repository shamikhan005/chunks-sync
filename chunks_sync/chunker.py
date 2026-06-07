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
    Split text into overlapping chunks by character count.

    Args:
        text:       Full document text.
        chunk_size: Max characters per chunk.
        overlap:    Characters of overlap between consecutive chunks.
                    Must be less than chunk_size.

    Returns:
        List of Chunk objects with index and position info.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not text.strip():
        return []

    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            newline = text.rfind("\n", start, end)
            space = text.rfind(" ", start, end)
            break_at = newline if newline > start else space
            if break_at > start:
                end = break_at

        chunk_text_content = text[start:end].strip()
        if chunk_text_content:
            chunks.append(Chunk(
                index=index,
                text=chunk_text_content,
                char_start=start,
                char_end=end,
            ))
            index += 1

        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks