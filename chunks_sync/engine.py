import time
import uuid
from typing import Callable, Optional
from .adapters import VectorDBAdapter
from .chunker import Chunk
from .hasher import hash_chunk_id, hash_content, hash_metadata
from .loader import ChangeType, DocumentChange, scan_directory
from .registry import ChunkRecord, ChunkRegistry, SyncReport

_DEFAULT_COST_PER_1K_TOKENS = 0.00002

def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

def _embed_chunks(
    chunks: list[Chunk],
    embed_fn: Callable[[list[str]], list[list[float]]],
) -> list[list[float]]:
    """Call embed_fn with chunk texts, return vectors."""
    texts = [c.text for c in chunks]
    return embed_fn(texts)

def sync(
    source: str,
    vector_db: VectorDBAdapter,
    embed_fn: Callable[[list[str]], list[list[float]]],
    embedding_model: str = "text-embedding-3-small",
    cost_per_1k_tokens: float = _DEFAULT_COST_PER_1K_TOKENS,
    chunk_size: int = 512,
    overlap: int = 64,
    registry_path: str = ".chunks_sync.db",
    metadata_fn: Optional[Callable] = None,
    verbose: bool = True,
) -> SyncReport:
    """
    Sync a source directory to a vector DB incrementally.

    Only re-embeds chunks whose content actually changed.
    Automatically deletes chunks from documents that were removed.

    Args:
        source:          Directory path containing source documents.
        vector_db:       VectorDBAdapter instance (Pinecone, Qdrant, Weaviate).
        embed_fn:        Callable: list[str] -> list[list[float]].
                         Wrap your embedding API call here.
        embedding_model: Model name stored in registry for tracking.
        chunk_size:      Characters per chunk.
        overlap:         Overlap between consecutive chunks.
        registry_path:   Path to local SQLite registry file.
        metadata_fn:     Optional fn(Path) -> dict for per-doc metadata.
        verbose:         Print sync report after completion.

    Returns:
        SyncReport with counts and cost estimates.
    """
    run_id = str(uuid.uuid4())
    started_at = time.time()

    registry = ChunkRegistry(db_path=registry_path)
    registry.start_run(run_id)

    new_chunks = 0
    updated_chunks = 0
    deleted_chunks = 0
    skipped_chunks = 0
    tokens_used = 0
    tokens_saved = 0

    changes: list[DocumentChange] = scan_directory(
        directory=source,
        registry=registry,
        chunk_size=chunk_size,
        overlap=overlap,
        metadata_fn=metadata_fn,
    )

    for change in changes:
        if change.change_type == ChangeType.DELETED:
            chunk_ids = registry.mark_chunks_deleted(change.doc_id)
            if chunk_ids:
                vector_db.delete(chunk_ids)
                deleted_chunks += len(chunk_ids)
            continue

        if change.change_type == ChangeType.UNCHANGED:
            for chunk in change.chunks:
                tokens_saved += _estimate_tokens(chunk.text)
                skipped_chunks += 1
            continue

        if change.change_type == ChangeType.METADATA_ONLY:
            existing_records = registry.get_chunks_for_doc(change.doc_id)
            batch_records = []
            for record in existing_records:
                vector_db.patch_metadata(record.chunk_id, change.metadata)
                new_meta_hash = hash_metadata(change.metadata)
                batch_records.append(ChunkRecord(
                    chunk_id=record.chunk_id,
                    doc_id=record.doc_id,
                    source=change.doc_path,
                    content_hash=record.content_hash,
                    metadata_hash=new_meta_hash,
                    embedding_model=record.embedding_model,
                    active=True,
                    source_version=record.source_version,
                    last_synced=time.time(),
                ))
                tokens_saved += _estimate_tokens(
                    next((c.text for c in change.chunks
                          if hash_chunk_id(change.doc_id, c.index) == record.chunk_id),
                         "")
                )
                skipped_chunks += 1
            registry.upsert_chunks_batch(batch_records)
            continue

        existing_records = {
            r.chunk_id: r
            for r in registry.get_chunks_for_doc(change.doc_id)
        }

        chunks_to_embed: list[tuple[int, Chunk]] = []
        chunks_to_patch: list[Chunk] = []
        current_chunk_ids: set[str] = set()

        for chunk in change.chunks:
            chunk_id = hash_chunk_id(change.doc_id, chunk.index)
            current_chunk_ids.add(chunk_id)
            new_content_hash = hash_content(chunk.text)
            new_meta_hash = hash_metadata(change.metadata)
            existing = existing_records.get(chunk_id)

            if existing is None:
                chunks_to_embed.append((len(chunks_to_embed), chunk))
            elif existing.content_hash != new_content_hash:
                chunks_to_embed.append((len(chunks_to_embed), chunk))
            elif existing.metadata_hash != new_meta_hash:
                chunks_to_patch.append(chunk)
            else:
                tokens_saved += _estimate_tokens(chunk.text)
                skipped_chunks += 1

        if chunks_to_embed:
            raw_chunks = [c for (_, c) in chunks_to_embed]
            vectors = _embed_chunks(raw_chunks, embed_fn)
            batch_records = []

            for (_, chunk), vector in zip(chunks_to_embed, vectors):
                chunk_id = hash_chunk_id(change.doc_id, chunk.index)
                content_hash = hash_content(chunk.text)
                meta_hash = hash_metadata(change.metadata)
                tok = _estimate_tokens(chunk.text)
                tokens_used += tok

                vector_db.upsert(chunk_id, vector, {
                    **change.metadata,
                    "chunk_index": chunk.index,
                    "doc_id": change.doc_id,
                    "chunk_id": chunk_id,
                })

                is_update = chunk_id in existing_records
                batch_records.append(ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=change.doc_id,
                    source=change.doc_path,
                    content_hash=content_hash,
                    metadata_hash=meta_hash,
                    embedding_model=embedding_model,
                    active=True,
                    source_version=(
                        existing_records[chunk_id].source_version + 1
                        if is_update else 1
                    ),
                    last_synced=time.time(),
                ))

                if is_update:
                    updated_chunks += 1
                else:
                    new_chunks += 1

            registry.upsert_chunks_batch(batch_records)

        if chunks_to_patch:
            patch_records = []
            for chunk in chunks_to_patch:
                chunk_id = hash_chunk_id(change.doc_id, chunk.index)
                vector_db.patch_metadata(chunk_id, change.metadata)
                meta_hash = hash_metadata(change.metadata)
                existing = existing_records[chunk_id]
                patch_records.append(ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=change.doc_id,
                    source=change.doc_path,
                    content_hash=existing.content_hash,
                    metadata_hash=meta_hash,
                    embedding_model=embedding_model,
                    active=True,
                    source_version=existing.source_version,
                    last_synced=time.time(),
                ))
                tokens_saved += _estimate_tokens(chunk.text)
                skipped_chunks += 1
            registry.upsert_chunks_batch(patch_records)

        removed_ids = list(set(existing_records.keys()) - current_chunk_ids)
        if removed_ids:
            vector_db.delete(removed_ids)
            registry.mark_specific_chunks_deleted(removed_ids)
            deleted_chunks += len(removed_ids)

    total_chunks = new_chunks + updated_chunks + skipped_chunks
    cost = (tokens_used / 1000) * cost_per_1k_tokens
    savings = (tokens_saved / 1000) * cost_per_1k_tokens

    report = SyncReport(
        run_id=run_id,
        total_chunks=total_chunks,
        new_chunks=new_chunks,
        updated_chunks=updated_chunks,
        deleted_chunks=deleted_chunks,
        skipped_chunks=skipped_chunks,
        tokens_used=tokens_used,
        tokens_saved=tokens_saved,
        estimated_cost_usd=cost,
        estimated_savings_usd=savings,
        duration_seconds=time.time() - started_at,
    )

    registry.finish_run(run_id, report)
    registry.close()

    if verbose:
        report.print()

    return report