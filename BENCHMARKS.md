## benchmarks

Benchmarked against 99 Wikipedia articles (3.59M characters, ~36K chars/doc average).
Corpus represents a realistic internal knowledge base, varied topics, substantial length.

Dataset: public Wikipedia articles via `wikipedia-api`.
Benchmark script: [`benchmarks/run_benchmark.py`](benchmarks/run_benchmark.py)
Reproducible: `uv run python benchmarks/run_benchmark.py --data benchmarks/data/wiki_100`

Embedding model:
- OpenAI-compatible benchmark assumptions
- Approximate token count: 1,015,8591 characters
- Cost model: text-embedding-small-3-small pricing ($0.00002 / 1K tokens)

### Cold Start

First sync into an empty vector index.

| Metric | Value |
|----------|----------|
| Chunks embedded | 8,130 |
| Tokens embedded | 1,015,891 |
| API calls | 8,130 |
| Time | 1.25s |

All chunks are embedded and written to the vector database.

### No Changes

Second sync with no document modifications.

| Metric | Value |
|----------|----------|
| Chunks embedded | 0 |
| Chunks skipped | 8,130 |
| API calls | 0 |
| Embedding savings | 100% |
| Time | 0.09s |

No embeddings are generated because chunk content hashes match the registry.

### Edit 1% of Documents

1 of 99 documents modified.

| Metric | Value |
|----------|----------|
| Chunks re-embedded | 1 |
| Chunks skipped | 8,129 |
| API calls | 1 |
| Embedding savings | 99.99% |
| Time | 0.09s |

The benchmark appends a short paragraph to one document. Because chunk IDs are position-based, existing chunks stay at the same positions and are skipped. Only the newly approached chunk is embedded.

### Edit 5% of Documents

4 of 99 documents modified.

| Metric | Value |
|----------|----------|
| Chunks re-embedded | 4 |
| Chunks skipped | 8,126 |
| API calls | 4 |
| Embedding savings | 99.95% |
| Time | 0.13s |

Only affected chunks are re-embedded.

### Edit 10% of Documents

9 of 99 documents modified.

| Metric | Value |
|----------|----------|
| Chunks re-embedded | 10 |
| Chunks skipped | 8,121 |
| API calls | 10 |
| Embedding savings | 99.88% |
| Time | 0.19s |

Even with 10% of documents modified, fewer than 0.2% of chunks required re-embedding.

### Delete 5% of Documents

4 documents removed from the corpus.

| Metric | Value |
|----------|----------|
| Chunks deleted | 260 |
| Embedding calls | 0 |
| Time | 0.09s |

Orphaned vectors are automatically removed from the index. No manual cleanup required.

### Metadata-Only Update

ACL / permission metadata changed.

| Metric | Value |
|----------|----------|
| Chunks re-embedded | 0 |
| Metadata PATCH operations | 208 |
| Embedding calls | 0 |

Permission changes propagate as lightweight PATCH calls to the vector DB. No re-embedding. No GPU or API cost.

<img width="957" height="632" alt="Screenshot 2026-06-12 222826" src="https://github.com/user-attachments/assets/3da871a2-8082-4229-a931-cc2f1e3ab19d" />

<img width="1037" height="517" alt="Screenshot 2026-06-12 222833" src="https://github.com/user-attachments/assets/83de2bde-ccf6-42a0-a614-741ea90075b3" />

<img width="982" height="636" alt="Screenshot 2026-06-12 222905" src="https://github.com/user-attachments/assets/131e4ee1-8dd4-458f-a3fd-3f543e14e824" />

<img width="952" height="508" alt="Screenshot 2026-06-12 222918" src="https://github.com/user-attachments/assets/341fdb03-6d6f-403c-845a-64a2e2aa4a20" />

<img width="927" height="630" alt="Screenshot 2026-06-12 223048" src="https://github.com/user-attachments/assets/f6840eb0-86de-4433-9a38-6955339058dc" />

<img width="1008" height="511" alt="Screenshot 2026-06-12 223101" src="https://github.com/user-attachments/assets/856e76a2-a5df-4b94-8549-e12a0b1d82fb" />

### note on timing

Sync times above reflect hashing and SQLite overhead only, the benchmark uses a mock
embedding function with no network latency. In a real deployment, incremental sync time
depends on how many chunks need re-embedding × your embedding API latency. The API call
counts above are exact and directly translate to real cost savings.

Cold start (first sync, 8,130 chunks): **1.25 seconds** of processing overhead.
Incremental sync (subsequent runs): **~0.09-0.19 seconds** regardless of corpus size,
as long as the changed fraction is small.
