import json
import xxhash

def hash_content(text: str) -> str:
    """Hash chunk text content. Change triggers re-embedding."""
    return xxhash.xxh64(text.encode("utf-8")).hexdigest()

def hash_metadata(metadata: dict) -> str:
    """
    Hash chunk metadata (permissions, owner, source_url, etc).
    Change triggers a lightweight PATCH — no re-embedding needed.
    Dict is sorted before hashing so key order doesn't matter.
    """
    serialized = json.dumps(metadata, sort_keys=True, ensure_ascii=False)
    return xxhash.xxh64(serialized.encode("utf-8")).hexdigest()

def hash_doc_id(source: str, doc_path: str) -> str:
    """
    Stable doc ID from source name + path.
    e.g. hash_doc_id("local", "./docs/handbook.pdf")
    """
    key = f"{source}::{doc_path}"
    return xxhash.xxh64(key.encode("utf-8")).hexdigest()

def hash_chunk_id(doc_id: str, chunk_index: int) -> str:
    """
    Stable chunk ID from doc ID + position in document.
    Chunk positions are stable as long as chunking strategy is stable.
    """
    key = f"{doc_id}::chunk::{chunk_index}"
    return xxhash.xxh64(key.encode("utf-8")).hexdigest()