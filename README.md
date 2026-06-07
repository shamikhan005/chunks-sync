# chunks-sync

Incremental synchronization for RAG pipelines.

Most RAG ingestion pipelines re-embed every document whenever a file changes, even if only one paragraph was edited. At scale, this wastes significant compute and API budget.

`chunks-sync` maintains a local **chunk registry** and only re-embeds the chunks whose content actually changed. Deletions propagate automatically. Permission changes update vector metadata without re-embedding anything.

## install

```bash
pip install git+https://github.com/shamikhan005/chunks-sync.git

# with uv
uv add git+https://github.com/shamikhan005/chunks-sync.git
```

Vector DB extras:

```bash
pip install "chunks-sync[pinecone]"   # Pinecone
pip install "chunks-sync[qdrant]"     # Qdrant
pip install "chunks-sync[weaviate]"   # Weaviate
pip install "chunks-sync[all]"        # all adapters
```

## quickstart

```python
from chunks_sync import sync
from chunks_sync.adapters import PineconeAdapter
import pinecone
import openai
 
pc = pinecone.Pinecone(api_key="YOUR_KEY")
index = pc.Index("your-index")
adapter = PineconeAdapter(index)
 
client = openai.OpenAI()
 
def embed(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [r.embedding for r in response.data]
 
# First run — ingests everything
report = sync(source="./docs", vector_db=adapter, embed_fn=embed)
 
# Edit a file, run again — only changed chunks are re-embedded
report = sync(source="./docs", vector_db=adapter, embed_fn=embed)
```

**First sync:** 3 documents, 197 chunks ingested:

```
── chunks-sync report ──────────────────────
  total chunks tracked : 197
  new                  : 197
  updated              : 0
  deleted              : 0
  skipped (unchanged)  : 0
  tokens used          : 2,241
  tokens saved         : 0
  cost (this run)      : $0.0000
  cost saved           : $0.0000
  duration             : 0.1s
────────────────────────────────────────────
```

**After editing one file:** 133 chunks skipped, only changed chunks re-embedded:
 
```
── chunks-sync report ──────────────────────
  total chunks tracked : 198
  new                  : 1
  updated              : 64
  deleted              : 0
  skipped (unchanged)  : 133
  tokens used          : 639
  tokens saved         : 1,649
  cost (this run)      : $0.0000
  cost saved           : $0.0000
  duration             : 0.0s
────────────────────────────────────────────
```
 
67% of embedding calls skipped on a single-file edit across a 3-document corpus. Savings compound as your corpus grows.

## features

- **Incremental chunk-level re-embedding:** only changed chunks are sent to the embedding API
- **Metadata-only updates:** permission and ACL changes propagate as lightweight PATCH calls, no re-embedding
- **Deletion propagation:** when a source document is deleted, all its chunks are removed from the vector DB automatically
- **Local chunk registry:** SQLite state table mapping every source document to its chunk IDs and content hashes
- **Cost saving report:** every sync reports tokens used, tokens saved, and estimated API cost avoided
- **Vector DB agnostic:** same `sync()` call works across Pinecone, Qdrant, and weaviate

## how it works

<img width="888" height="532" alt="Screenshot 2026-06-07 194153" src="https://github.com/user-attachments/assets/ef83f09e-21f9-4767-9783-f00a94d91e55" />

The chunk registry stores one row per chunk:

| field | description |
|---|---|
| `chunk_id` | stable ID derived from doc path + chunk index |
| `doc_id` | which source document this chunk came from |
| `content_hash` | xxHash of chunk text: change triggers re-embedding |
| `metadata_hash` | xxHash of metadata: change triggers PATCH only |
| `source_version` | incremented on each content update |
| `active` | set to false when source document is deleted |
| `last_synced` | timestamp of last sync |

Content and metadata hashes are tracked separately. a permission change on a document propagates to all its chunks as a cheap metadata PATCH, no re-embedding, no GPU or API cost.

## supported vector databases

| Vector DB | Adapter | Install |
|-----------|---------|---------|
| Pinecone  | `PineconeAdapter` | `pip install "chunks-sync[pinecone]"` |
| Qdrant    | `QdrantAdapter` | `pip install "chunks-sync[qdrant]"` |
| Weaviate  | `WeaviateAdapter` | `pip install "chunks-sync[weaviate]"` |
 
Adding a new adapter requires implementing three methods: `upsert`, `patch_metadata`, and `delete`. See `chunks_sync/adapters.py`.

## permission and metadata sync

When access rights change on a document, pass updated metadata via `metadata_fn`:

```python
from pathlib import Path
 
def get_permissions(path: Path) -> dict:
    return {
        "owner": "hr_team",
        "access": "restricted",
    }
 
report = sync(
    source="./docs",
    vector_db=adapter,
    embed_fn=embed,
    metadata_fn=get_permissions,
)
# chunks-sync detects metadata changed, issues PATCH to vector DB
# zero chunks re-embedded
```

## known limitations

**Chunk identity is position-based.** Chunk IDs are derived from `doc_id + chunk_index`. Inserting content near the beginning of a document shifts downstream chunk indexes, which may cause more chunks to be re-embedded than strictly necessary. Paragraph-aware diffing is on the roadmap.
 
**File renames are treated as delete + create.** Renaming `handbook.md` to `employee_handbook.md` causes all chunks to be deleted and re-ingested under the new path.
 
**SQLite registry is local.** The registry lives at `.chunks_sync.db` next to where you run sync. For distributed or multi-process deployments, a shared registry backend would be needed.

**Swtiching embedding models mid-corpus is not yet detected.** If you change `embed_fn` to a different model, delete `.chunks_sync.db` to force a full re-index.

## roadmap

- [ ] Paragraph-aware diffing: content-addressed chunk IDs to reduce re-embedding on large edits
- [ ] Batch vector operations: single API call per sync instead of per chunk
- [ ] Registry inspection CLI: `chunks-sync status`, `chunks-sync list-docs`
- [ ] Governance metadata: ownership, sensitivity classification, expiry per chunk
- [ ] Connectors: Notion, Confluence, S3, Google Drive

## contributing

```bash
git clone https://github.com/shamikhan005/chunks-sync
cd chunks-sync
uv sync
uv run python test_core.py   # 9/9 should pass
```

Open an issue if you want a specific connector or vector DB adapter next.

## License

[MIT](LICENSE)
