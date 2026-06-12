import os
import shutil
import tempfile
from collections import defaultdict
from chunks_sync import sync
from chunks_sync.adapters import VectorDBAdapter

class MockVectorDB(VectorDBAdapter):
    def __init__(self):
        self.vectors: dict[str, list[float]] = {}
        self.metadata: dict[str, dict] = {}
        self.upsert_calls: list[str] = []
        self.patch_calls: list[str] = []
        self.delete_calls: list[str] = []

    def upsert(self, chunk_id, vector, metadata):
        self.vectors[chunk_id] = vector
        self.metadata[chunk_id] = metadata
        self.upsert_calls.append(chunk_id)

    def patch_metadata(self, chunk_id, metadata):
        if chunk_id in self.metadata:
            self.metadata[chunk_id].update(metadata)
        self.patch_calls.append(chunk_id)

    def delete(self, chunk_ids):
        for cid in chunk_ids:
            self.vectors.pop(cid, None)
            self.metadata.pop(cid, None)
        self.delete_calls.extend(chunk_ids)

def mock_embed(texts: list[str]) -> list[list[float]]:
    """Returns deterministic fake vectors. One call per text."""
    return [[float(hash(t) % 1000) / 1000.0] * 8 for t in texts]

def make_temp_dir():
    return tempfile.mkdtemp(prefix="chunks_sync_test_")


def write_doc(dir_path, filename, content):
    path = os.path.join(dir_path, filename)
    with open(path, "w") as f:
        f.write(content)
    return path

def test_first_sync():
    print("test 1: first sync — all chunks ingested")
    tmpdir = make_temp_dir()
    db_path = os.path.join(tmpdir, "registry.db")
    try:
        write_doc(tmpdir, "a.txt", "The quick brown fox jumps over the lazy dog. " * 10)
        write_doc(tmpdir, "b.txt", "RAG systems need fresh data to answer correctly. " * 10)

        vdb = MockVectorDB()
        report = sync(
            source=tmpdir,
            vector_db=vdb,
            embed_fn=mock_embed,
            registry_path=db_path,
            verbose=False,
        )

        assert report.new_chunks > 0, "Expected new chunks on first sync"
        assert report.skipped_chunks == 0, "Nothing should be skipped on first sync"
        assert report.deleted_chunks == 0
        assert len(vdb.vectors) == report.new_chunks
        print(f"  PASS — {report.new_chunks} chunks ingested")
    finally:
        shutil.rmtree(tmpdir)


def test_no_change_sync():
    print("test 2: no-change sync — nothing re-embedded")
    tmpdir = make_temp_dir()
    db_path = os.path.join(tmpdir, "registry.db")
    try:
        write_doc(tmpdir, "a.txt", "The quick brown fox jumps over the lazy dog. " * 10)

        vdb = MockVectorDB()
        r1 = sync(source=tmpdir, vector_db=vdb, embed_fn=mock_embed,
                  registry_path=db_path, verbose=False)

        vdb.upsert_calls.clear()
        r2 = sync(source=tmpdir, vector_db=vdb, embed_fn=mock_embed,
                  registry_path=db_path, verbose=False)

        assert r2.new_chunks == 0, "No new chunks expected"
        assert r2.updated_chunks == 0, "No updated chunks expected"
        assert r2.skipped_chunks == r1.new_chunks, "All chunks should be skipped"
        assert len(vdb.upsert_calls) == 0, "No upsert calls on unchanged sync"
        print(f"  PASS — {r2.skipped_chunks} chunks skipped, 0 API calls")
    finally:
        shutil.rmtree(tmpdir)


def test_update_sync():
    print("test 3: update sync — only changed chunk re-embedded")
    tmpdir = make_temp_dir()
    db_path = os.path.join(tmpdir, "registry.db")
    try:
        write_doc(tmpdir, "a.txt", "Original content. " * 20)

        vdb = MockVectorDB()
        r1 = sync(source=tmpdir, vector_db=vdb, embed_fn=mock_embed,
                  registry_path=db_path, verbose=False)

        vdb.upsert_calls.clear()
        write_doc(tmpdir, "a.txt", "Modified content — completely different. " * 20)

        r2 = sync(source=tmpdir, vector_db=vdb, embed_fn=mock_embed,
                  registry_path=db_path, verbose=False)

        assert r2.updated_chunks > 0 or r2.new_chunks > 0, "Expected re-embedding after update"
        assert r2.tokens_saved >= 0
        print(f"  PASS — {r2.updated_chunks} chunks updated, "
              f"{r2.skipped_chunks} skipped, "
              f"{r2.tokens_saved} tokens saved")
    finally:
        shutil.rmtree(tmpdir)


def test_delete_propagation():
    print("test 4: delete propagation — orphaned chunks removed from vector DB")
    tmpdir = make_temp_dir()
    db_path = os.path.join(tmpdir, "registry.db")
    try:
        write_doc(tmpdir, "keep.txt", "This document stays. " * 20)
        write_doc(tmpdir, "delete_me.txt", "This document will be deleted. " * 20)

        vdb = MockVectorDB()
        r1 = sync(source=tmpdir, vector_db=vdb, embed_fn=mock_embed,
                  registry_path=db_path, verbose=False)
        vectors_before = len(vdb.vectors)

        os.remove(os.path.join(tmpdir, "delete_me.txt"))

        r2 = sync(source=tmpdir, vector_db=vdb, embed_fn=mock_embed,
                  registry_path=db_path, verbose=False)

        assert r2.deleted_chunks > 0, "Expected deleted chunks"
        assert len(vdb.vectors) < vectors_before, "Vectors should be removed from DB"
        assert len(vdb.delete_calls) > 0, "Delete should have been called on vector DB"
        print(f"  PASS — {r2.deleted_chunks} orphaned chunks removed from vector DB")
    finally:
        shutil.rmtree(tmpdir)


def test_partial_chunk_deletion():
    print("test 5: partial chunk deletion — removing paragraphs doesn't nuke whole doc")
    tmpdir = make_temp_dir()
    db_path = os.path.join(tmpdir, "registry.db")
    try:
        # Write a document long enough to produce multiple chunks
        long_content = "\n\n".join([f"Section {i}: " + ("content " * 30) for i in range(8)])
        write_doc(tmpdir, "doc.txt", long_content)

        vdb = MockVectorDB()
        r1 = sync(source=tmpdir, vector_db=vdb, embed_fn=mock_embed,
                  registry_path=db_path, verbose=False)
        chunks_before = len(vdb.vectors)
        assert chunks_before > 1, "Need multiple chunks for this test"

        # Shorten the document — removes the last few sections
        short_content = "\n\n".join([f"Section {i}: " + ("content " * 30) for i in range(3)])
        write_doc(tmpdir, "doc.txt", short_content)

        r2 = sync(source=tmpdir, vector_db=vdb, embed_fn=mock_embed,
                  registry_path=db_path, verbose=False)

        chunks_after = len(vdb.vectors)
        assert chunks_after < chunks_before, "Some chunks should have been deleted"
        assert chunks_after > 0, "Remaining chunks should still exist — doc wasn't fully deleted"
        assert r2.deleted_chunks > 0, "deleted_chunks counter should reflect removals"
        print(f"  PASS — {chunks_before} chunks before, {chunks_after} after partial deletion, "
              f"{r2.deleted_chunks} removed correctly")
    finally:
        shutil.rmtree(tmpdir)


def test_metadata_only_update():
    print("test 6: metadata-only update — PATCH issued, no re-embedding")
    tmpdir = make_temp_dir()
    db_path = os.path.join(tmpdir, "registry.db")
    try:
        content = "Stable document content that will not change. " * 20
        write_doc(tmpdir, "doc.txt", content)

        vdb = MockVectorDB()
        r1 = sync(source=tmpdir, vector_db=vdb, embed_fn=mock_embed,
                  registry_path=db_path, verbose=False)

        upserts_after_first = len(vdb.upsert_calls)
        vdb.upsert_calls.clear()
        vdb.patch_calls.clear()

        from pathlib import Path
        call_count = {"n": 0}

        def meta_with_new_permissions(path: Path) -> dict:
            return {"owner": "hr_team", "access": "restricted"}

        r2 = sync(
            source=tmpdir,
            vector_db=vdb,
            embed_fn=mock_embed,
            registry_path=db_path,
            metadata_fn=meta_with_new_permissions,
            verbose=False,
        )

        assert len(vdb.upsert_calls) == 0, \
            f"No re-embedding should happen on metadata-only change, got {len(vdb.upsert_calls)} upserts"
        assert len(vdb.patch_calls) > 0, \
            "PATCH should have been called to update metadata on chunks"
        print(f"  PASS — 0 re-embeds, {len(vdb.patch_calls)} metadata patches issued")
    finally:
        shutil.rmtree(tmpdir)


def test_file_rename_behavior():
    print("test 7: file rename — currently treated as delete + new (documented limitation)")
    tmpdir = make_temp_dir()
    db_path = os.path.join(tmpdir, "registry.db")
    try:
        write_doc(tmpdir, "original.txt", "Content that will be renamed. " * 20)

        vdb = MockVectorDB()
        r1 = sync(source=tmpdir, vector_db=vdb, embed_fn=mock_embed,
                  registry_path=db_path, verbose=False)
        chunks_after_first = len(vdb.vectors)

        # Rename the file
        os.rename(
            os.path.join(tmpdir, "original.txt"),
            os.path.join(tmpdir, "renamed.txt"),
        )

        vdb.upsert_calls.clear()
        vdb.delete_calls.clear()
        r2 = sync(source=tmpdir, vector_db=vdb, embed_fn=mock_embed,
                  registry_path=db_path, verbose=False)

        # Document behavior: rename = delete old + ingest new (re-embeds everything)
        # This is the known chunk identity limitation from the review
        assert r2.deleted_chunks > 0, "Old chunks should be deleted"
        assert r2.new_chunks > 0, "New chunks should be ingested under new path"
        assert len(vdb.upsert_calls) > 0, "Re-embedding happens on rename (known limitation)"
        print(f"  PASS (known limitation documented) — rename triggers "
              f"{r2.deleted_chunks} deletes + {r2.new_chunks} re-embeds")
    finally:
        shutil.rmtree(tmpdir)


def test_cost_savings_reported():
    print("test 8: cost report — tokens saved estimated correctly")
    tmpdir = make_temp_dir()
    db_path = os.path.join(tmpdir, "registry.db")
    try:
        write_doc(tmpdir, "a.txt", "Cost saving test content. " * 30)
        write_doc(tmpdir, "b.txt", "More content for cost testing. " * 30)

        vdb = MockVectorDB()
        sync(source=tmpdir, vector_db=vdb, embed_fn=mock_embed,
             registry_path=db_path, verbose=False)

        r2 = sync(source=tmpdir, vector_db=vdb, embed_fn=mock_embed,
                  registry_path=db_path, verbose=False)

        assert r2.tokens_saved > 0, "Expected tokens saved on second sync"
        assert r2.estimated_savings_usd >= 0
        print(f"  PASS — {r2.tokens_saved:,} tokens saved "
              f"(${r2.estimated_savings_usd:.6f} estimated savings)")
    finally:
        shutil.rmtree(tmpdir)


def test_overlap_validation():
    print("test 9: chunker validation — overlap >= chunk_size raises ValueError")
    from chunks_sync.chunker import chunk_text
    try:
        chunk_text("some text", chunk_size=100, overlap=100)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    try:
        chunk_text("some text", chunk_size=100, overlap=200)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  PASS — invalid overlap correctly rejected")

def test_no_tail_chunk_explosion():
    print("test 10: no tail chunk explosion")

    from chunks_sync.chunker import chunk_text

    text = "A" * 600

    chunks = chunk_text(
        text,
        chunk_size=512,
        overlap=64,
    )

    assert len(chunks) == 2, (
        f"Expected 2 chunks, got {len(chunks)}"
    )

    print("  PASS — tail chunk explosion prevented")

def test_realistic_chunk_density():
    print("test 11: realistic chunk density")

    from chunks_sync.chunker import chunk_text

    text = "The quick brown fox jumps over the lazy dog. " * 800

    chunks = chunk_text(
        text,
        chunk_size=512,
        overlap=64,
    )

    avg_chars_per_chunk = len(text) / len(chunks)

    assert avg_chars_per_chunk > 250, (
        f"Chunk density too low: {avg_chars_per_chunk}"
    )

    print(
        f"  PASS — {avg_chars_per_chunk:.1f} chars/chunk"
    )

def test_chunk_count_sanity():
    print("test 12: chunk count sanity")

    from chunks_sync.chunker import chunk_text

    text = "The quick brown fox jumps over the lazy dog. " * 800

    chunks = chunk_text(
        text,
        chunk_size=512,
        overlap=64,
    )

    assert len(chunks) < 200, (
        f"Produced suspiciously many chunks: {len(chunks)}"
    )

    print(
        f"  PASS — {len(chunks)} chunks generated"
    )

if __name__ == "__main__":
    print("\nrunning chunks-sync tests\n")

    test_first_sync()
    test_no_change_sync()
    test_update_sync()
    test_delete_propagation()
    test_partial_chunk_deletion()
    test_metadata_only_update()
    test_file_rename_behavior()
    test_cost_savings_reported()
    test_overlap_validation()

    test_no_tail_chunk_explosion()
    test_realistic_chunk_density()
    test_chunk_count_sanity()

    print("\nall tests passed ✓\n")