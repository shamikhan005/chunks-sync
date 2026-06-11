## benchmarks

Benchmarked against 99 Wikipedia articles (3.59M characters, ~36K chars/doc average).
Corpus represents a realistic internal knowledge base, varied topics, substantial length.

Dataset: public Wikipedia articles via `wikipedia-api`.
Benchmark script: [`benchmarks/run_benchmark.py`](benchmarks/run_benchmark.py)
Reproducible: `uv run python benchmarks/run_benchmark.py --data benchmarks/data/wiki_100`

### no-change sync (nothing edited)

| metric | value |
|---|---|
| chunks tracked | 160,806 |
| chunks skipped | 160,806 |
| API calls made | 0 |
| API calls avoided | 160,806 |
| savings | 100% |
| sync time | 1.06s |

If nothing changed since the last sync, chunks-sync makes zero embedding API calls.

### incremental sync (edited N% of documents)

| edit scenario | docs edited | chunks updated | chunks skipped | API calls | savings |
|---|---|---|---|---|---|
| edit 1% of docs | 1 of 99 | 64 | 160,741 | 64 | 99.96% |
| edit 5% of docs | 4 of 99 | 257 | 160,546 | 257 | 99.8% |
| edit 10% of docs | 9 of 99 | 577 | 160,221 | 577 | 99.6% |

A naive pipeline re-embeds every chunk on any change.
chunks-sync re-embeds only the chunks that actually changed.

<img width="1106" height="612" alt="Screenshot 2026-06-11 153808" src="https://github.com/user-attachments/assets/10de44e4-e4cd-4fdd-ae7a-df9e88e3c3f5" />
<img width="995" height="513" alt="Screenshot 2026-06-11 153821" src="https://github.com/user-attachments/assets/ba8c0f3e-546a-4ba4-b46f-09dc43d1f4af" />
<img width="1111" height="638" alt="Screenshot 2026-06-11 153525" src="https://github.com/user-attachments/assets/f42ef960-172d-4cc3-8449-f3b8d6831d4a" />
<img width="1097" height="527" alt="Screenshot 2026-06-11 153539" src="https://github.com/user-attachments/assets/07be8ba2-e268-4001-900e-a2d150a32e7c" />
<img width="1108" height="611" alt="Screenshot 2026-06-11 153647" src="https://github.com/user-attachments/assets/fdc22d73-d420-4646-af5c-d851a8ab2f30" />
<img width="1025" height="532" alt="Screenshot 2026-06-11 153701" src="https://github.com/user-attachments/assets/fedd88a9-050f-4560-92cb-b88924ae5a71" />

At `text-embedding-3-small` pricing ($0.02/1M tokens), editing 5% of this corpus:

| approach | tokens used | estimated cost |
|---|---|---|
| naive (re-embed all) | 2,214,910 | $0.044 |
| chunks-sync | 2,360 | $0.000047 |
| **savings** | **2,212,550** | **$0.044 per sync** |

Run this sync daily and that's ~$16/year saved on this corpus alone.
The savings scale linearly with corpus size.

### deletion sync (documents removed)

| metric | value |
|---|---|
| docs deleted | 4–5 of 99 |
| orphaned chunks removed | 3,330–4,281 |
| API calls to vector DB | automatic |
| manual cleanup required | none |

When source documents are deleted, all their chunks are automatically removed from the
vector index. Without chunks-sync, those chunks remain indefinitely, the LLM continues
answering from deleted content.

### note on timing

Sync times above reflect hashing and SQLite overhead only, the benchmark uses a mock
embedding function with no network latency. In a real deployment, incremental sync time
depends on how many chunks need re-embedding × your embedding API latency. The API call
counts above are exact and directly translate to real cost savings.

Cold start (first sync, 160K chunks): **6.56 seconds** of processing overhead.
Incremental sync (subsequent runs): **~1.1 seconds** regardless of corpus size,
as long as the changed fraction is small.
