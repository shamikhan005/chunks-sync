import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional
from .chunker import Chunk, chunk_text
from .hasher import hash_content, hash_doc_id, hash_metadata
from .registry import ChunkRegistry

class ChangeType(Enum):
    NEW = "new"
    UPDATED = "updated"
    METADATA_ONLY = "metadata_only"
    DELETED = "deleted"
    UNCHANGED = "unchanged"


@dataclass
class DocumentChange:
    doc_id: str
    doc_path: str
    change_type: ChangeType
    chunks: list[Chunk]
    metadata: dict


def _read_file(path: Path) -> Optional[str]:
    """Read text from supported file types."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md", ".rst"):
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            return "\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF support: pip install pypdf"
            )
    return None


def _build_metadata(path: Path, extra: dict) -> dict:
    """Build metadata dict for a document."""
    meta = {
        "source": "local",
        "filename": path.name,
        "path": str(path),
    }
    meta.update(extra)
    return meta


def scan_directory(
    directory: str,
    registry: ChunkRegistry,
    chunk_size: int = 512,
    overlap: int = 64,
    metadata_fn: Optional[Callable[[Path], dict]] = None,
    glob: str = "**/*",
) -> list[DocumentChange]:
    """
    Scan a directory and return DocumentChange events by comparing
    current files against the chunk registry.

    Args:
        directory:   Path to scan.
        registry:    ChunkRegistry instance to compare against.
        chunk_size:  Characters per chunk.
        overlap:     Overlap between chunks.
        metadata_fn: Optional function to attach metadata per file.
        glob:        Glob pattern for file discovery.

    Returns:
        List of DocumentChange events (new, updated, deleted, unchanged).
    """
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    supported = {".txt", ".md", ".rst", ".pdf"}
    changes = []
    seen_doc_ids = set()

    for path in sorted(root.glob(glob)):
        if not path.is_file():
            continue
        if path.suffix.lower() not in supported:
            continue

        text = _read_file(path)
        if text is None:
            continue

        extra_meta = metadata_fn(path) if metadata_fn else {}
        metadata = _build_metadata(path, extra_meta)
        doc_id = hash_doc_id("local", str(path))
        seen_doc_ids.add(doc_id)

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            continue

        existing = registry.get_chunks_for_doc(doc_id)
        existing_content_hashes = {r.chunk_id: r.content_hash for r in existing}
        existing_meta_hashes = {r.chunk_id: r.metadata_hash for r in existing}

        is_new = len(existing) == 0
        has_content_change = False
        has_metadata_change = False

        if not is_new:
            from .hasher import hash_chunk_id
            new_meta_hash = hash_metadata(metadata)

            for chunk in chunks:
                cid = hash_chunk_id(doc_id, chunk.index)
                new_content_hash = hash_content(chunk.text)

                if cid not in existing_content_hashes:
                    # New chunk within existing doc
                    has_content_change = True
                    break
                if existing_content_hashes[cid] != new_content_hash:
                    has_content_change = True
                    break

            if not has_content_change:
                first_chunk_id = list(existing_meta_hashes.keys())[0]
                if existing_meta_hashes.get(first_chunk_id) != hash_metadata(metadata):
                    has_metadata_change = True

            if len(chunks) != len(existing):
                has_content_change = True

        if is_new:
            change_type = ChangeType.NEW
        elif has_content_change:
            change_type = ChangeType.UPDATED
        elif has_metadata_change:
            change_type = ChangeType.METADATA_ONLY
        else:
            change_type = ChangeType.UNCHANGED

        changes.append(DocumentChange(
            doc_id=doc_id,
            doc_path=str(path),
            change_type=change_type,
            chunks=chunks,
            metadata=metadata,
        ))

    registry_doc_ids = registry.get_all_doc_ids()
    for deleted_doc_id in registry_doc_ids - seen_doc_ids:
        existing = registry.get_chunks_for_doc(deleted_doc_id)
        doc_path = existing[0].source if existing else "unknown"
        changes.append(DocumentChange(
            doc_id=deleted_doc_id,
            doc_path=doc_path,
            change_type=ChangeType.DELETED,
            chunks=[],
            metadata={},
        ))

    return changes