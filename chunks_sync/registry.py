import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    source: str
    content_hash: str
    metadata_hash: str
    embedding_model: str
    active: bool
    source_version: int
    last_synced: float

@dataclass
class SyncReport:
    run_id: str
    total_chunks: int
    new_chunks: int
    updated_chunks: int
    deleted_chunks: int
    skipped_chunks: int
    tokens_used: int
    tokens_saved: int
    estimated_cost_usd: float
    estimated_savings_usd: float
    duration_seconds: float

    def print(self):
        print("\n── chunks-sync report ──────────────────────")
        print(f"  total chunks tracked : {self.total_chunks}")
        print(f"  new                  : {self.new_chunks}")
        print(f"  updated              : {self.updated_chunks}")
        print(f"  deleted              : {self.deleted_chunks}")
        print(f"  skipped (unchanged)  : {self.skipped_chunks}")
        print(f"  tokens used          : {self.tokens_used:,}")
        print(f"  tokens saved         : {self.tokens_saved:,}")
        print(f"  cost (this run)      : ${self.estimated_cost_usd:.4f}")
        print(f"  cost saved           : ${self.estimated_savings_usd:.4f}")
        print(f"  duration             : {self.duration_seconds:.1f}s")
        print("────────────────────────────────────────────\n")

class ChunkRegistry:
    """
    Persistent SQLite registry mapping source documents to their
    vector DB chunks. Every sync() reads and writes through here.
    """

    def __init__(self, db_path: str = ".chunks_sync.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id        TEXT PRIMARY KEY,
                doc_id          TEXT NOT NULL,
                source          TEXT NOT NULL,
                content_hash    TEXT NOT NULL,
                metadata_hash   TEXT NOT NULL,
                embedding_model TEXT NOT NULL DEFAULT '',
                active          INTEGER NOT NULL DEFAULT 1,
                source_version  INTEGER NOT NULL DEFAULT 1,
                last_synced     REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
                ON chunks(doc_id);

            CREATE INDEX IF NOT EXISTS idx_chunks_active
                ON chunks(active);

            -- composite index for the hot query: get active chunks for a doc
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_active
                ON chunks(doc_id, active);

            CREATE TABLE IF NOT EXISTS sync_runs (
                run_id               TEXT PRIMARY KEY,
                started_at           REAL NOT NULL,
                finished_at          REAL,
                total_chunks         INTEGER DEFAULT 0,
                new_chunks           INTEGER DEFAULT 0,
                updated_chunks       INTEGER DEFAULT 0,
                deleted_chunks       INTEGER DEFAULT 0,
                skipped_chunks       INTEGER DEFAULT 0,
                tokens_used          INTEGER DEFAULT 0,
                tokens_saved         INTEGER DEFAULT 0,
                estimated_cost_usd   REAL DEFAULT 0.0,
                estimated_savings_usd REAL DEFAULT 0.0
            );
        """)
        self._conn.commit()

    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        row = self._conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return self._row_to_record(row) if row else None

    def get_chunks_for_doc(self, doc_id: str) -> list[ChunkRecord]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE doc_id = ? AND active = 1",
            (doc_id,)
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_all_doc_ids(self) -> set[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT doc_id FROM chunks WHERE active = 1"
        ).fetchall()
        return {r["doc_id"] for r in rows}

    def upsert_chunk(self, record: ChunkRecord):
        """Upsert a single chunk. For bulk operations prefer upsert_chunks_batch."""
        self._conn.execute("""
            INSERT INTO chunks
                (chunk_id, doc_id, source, content_hash, metadata_hash,
                 embedding_model, active, source_version, last_synced)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                content_hash    = excluded.content_hash,
                metadata_hash   = excluded.metadata_hash,
                embedding_model = excluded.embedding_model,
                active          = excluded.active,
                source_version  = excluded.source_version,
                last_synced     = excluded.last_synced
        """, (
            record.chunk_id, record.doc_id, record.source,
            record.content_hash, record.metadata_hash,
            record.embedding_model, int(record.active),
            record.source_version, record.last_synced
        ))
        self._conn.commit()

    def upsert_chunks_batch(self, records: list[ChunkRecord]) -> None:
        """
        Upsert multiple chunks in a single transaction.
        Significantly faster than calling upsert_chunk() in a loop —
        one commit instead of N commits.
        """
        if not records:
            return
        self._conn.executemany("""
            INSERT INTO chunks
                (chunk_id, doc_id, source, content_hash, metadata_hash,
                 embedding_model, active, source_version, last_synced)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                content_hash    = excluded.content_hash,
                metadata_hash   = excluded.metadata_hash,
                embedding_model = excluded.embedding_model,
                active          = excluded.active,
                source_version  = excluded.source_version,
                last_synced     = excluded.last_synced
        """, [
            (r.chunk_id, r.doc_id, r.source,
             r.content_hash, r.metadata_hash,
             r.embedding_model, int(r.active),
             r.source_version, r.last_synced)
            for r in records
        ])
        self._conn.commit()

    def mark_chunks_deleted(self, doc_id: str) -> list[str]:
        """Soft-delete ALL chunks for a doc (whole-doc deletion). Returns chunk_ids."""
        rows = self._conn.execute(
            "SELECT chunk_id FROM chunks WHERE doc_id = ? AND active = 1",
            (doc_id,)
        ).fetchall()
        chunk_ids = [r["chunk_id"] for r in rows]
        if chunk_ids:
            self._conn.execute(
                "UPDATE chunks SET active = 0 WHERE doc_id = ?", (doc_id,)
            )
            self._conn.commit()
        return chunk_ids

    def mark_specific_chunks_deleted(self, chunk_ids: list[str]) -> None:
        """
        Soft-delete specific chunk IDs only.
        Used when paragraphs are removed from the middle of a document —
        only those specific chunks should be marked inactive, not the whole doc.
        """
        if not chunk_ids:
            return
        placeholders = ",".join("?" * len(chunk_ids))
        self._conn.execute(
            f"UPDATE chunks SET active = 0 WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
        self._conn.commit()

    def start_run(self, run_id: str):
        self._conn.execute(
            "INSERT INTO sync_runs (run_id, started_at) VALUES (?, ?)",
            (run_id, time.time())
        )
        self._conn.commit()

    def finish_run(self, run_id: str, report: SyncReport):
        self._conn.execute("""
            UPDATE sync_runs SET
                finished_at           = ?,
                total_chunks          = ?,
                new_chunks            = ?,
                updated_chunks        = ?,
                deleted_chunks        = ?,
                skipped_chunks        = ?,
                tokens_used           = ?,
                tokens_saved          = ?,
                estimated_cost_usd    = ?,
                estimated_savings_usd = ?
            WHERE run_id = ?
        """, (
            time.time(),
            report.total_chunks, report.new_chunks,
            report.updated_chunks, report.deleted_chunks,
            report.skipped_chunks, report.tokens_used,
            report.tokens_saved, report.estimated_cost_usd,
            report.estimated_savings_usd, run_id
        ))
        self._conn.commit()

    def _row_to_record(self, row) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            source=row["source"],
            content_hash=row["content_hash"],
            metadata_hash=row["metadata_hash"],
            embedding_model=row["embedding_model"],
            active=bool(row["active"]),
            source_version=row["source_version"],
            last_synced=row["last_synced"],
        )

    def close(self):
        self._conn.close()