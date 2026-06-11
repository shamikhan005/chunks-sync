"""
chunks-sync benchmark runner.

Measures incremental sync performance across corpus sizes and edit scenarios.

Usage:
    uv run python benchmarks/run_benchmark.py --data benchmarks/data/wiki_100
    uv run python benchmarks/run_benchmark.py --data benchmarks/data/wiki_100 --edit-pct 0.1

What it measures:
    Run 1 — cold start (first sync, nothing in registry)
    Run 2 — no changes (everything should be skipped)
    Run 3 — edit N% of documents (only changed chunks re-embedded)
    Run 4 — delete 5% of documents (chunk deletion propagation)

Output:
    Prints a results table and saves benchmarks/results/results_<timestamp>.json
"""

import argparse
import json
import os
import random
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from chunks_sync import sync
from chunks_sync.adapters import VectorDBAdapter


class BenchmarkAdapter(VectorDBAdapter):
    """
    Tracks upsert/patch/delete calls without a real vector DB.
    Used for benchmarking so results don't depend on network latency.
    """
    def __init__(self):
        self.vectors: dict[str, list[float]] = {}
        self.upsert_count = 0
        self.patch_count = 0
        self.delete_count = 0

    def upsert(self, chunk_id, vector, metadata):
        self.vectors[chunk_id] = vector
        self.upsert_count += 1

    def patch_metadata(self, chunk_id, metadata):
        self.patch_count += 1

    def delete(self, chunk_ids):
        for cid in chunk_ids:
            self.vectors.pop(cid, None)
        self.delete_count += len(chunk_ids)

    def reset_counters(self):
        self.upsert_count = 0
        self.patch_count = 0
        self.delete_count = 0


def mock_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic fake embeddings — no API calls needed for benchmarking."""
    return [[float(hash(t) % 10000) / 10000.0] * 8 for t in texts]

@dataclass
class RunResult:
    label: str
    duration_seconds: float
    total_chunks: int
    new_chunks: int
    updated_chunks: int
    deleted_chunks: int
    skipped_chunks: int
    tokens_used: int
    tokens_saved: int
    api_calls_made: int       
    api_calls_avoided: int  
    savings_pct: float

@dataclass
class BenchmarkResult:
    dataset: str
    doc_count: int
    total_chars: int
    avg_chars_per_doc: int
    edit_pct: float
    runs: list[RunResult]

def run_benchmark(data_dir: Path, edit_pct: float = 0.05) -> BenchmarkResult:
    """
    Run the full benchmark suite against a dataset directory.

    Args:
        data_dir:  Path to directory with downloaded .txt files.
        edit_pct:  Fraction of documents to edit in run 3 (default 5%).
    """
    manifest_path = data_dir / "_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No _manifest.json found in {data_dir}. "
            "Run benchmarks/download_dataset.py first."
        )
    manifest = json.loads(manifest_path.read_text())
    doc_count = len(manifest)
    total_chars = sum(a["chars"] for a in manifest)

    print(f"\n── chunks-sync benchmark ───────────────────────────────")
    print(f"  dataset     : {data_dir.name}")
    print(f"  documents   : {doc_count}")
    print(f"  total chars : {total_chars:,}")
    print(f"  avg per doc : {total_chars // doc_count:,} chars")
    print(f"  edit pct    : {edit_pct * 100:.0f}%")
    print(f"────────────────────────────────────────────────────────\n")

    tmpdir = tempfile.mkdtemp(prefix="chunks_sync_bench_")
    work_dir = Path(tmpdir) / "docs"
    db_path = Path(tmpdir) / "registry.db"

    try:
        shutil.copytree(str(data_dir), str(work_dir))
        (work_dir / "_manifest.json").unlink(missing_ok=True)

        doc_files = sorted(work_dir.glob("*.txt"))
        adapter = BenchmarkAdapter()
        runs = []

        print("run 1/4: cold start (first sync)...")
        adapter.reset_counters()
        r1 = sync(
            source=str(work_dir),
            vector_db=adapter,
            embed_fn=mock_embed,
            registry_path=str(db_path),
            verbose=False,
        )
        total_possible_calls_r1 = r1.new_chunks + r1.updated_chunks + r1.skipped_chunks
        runs.append(RunResult(
            label="cold start",
            duration_seconds=round(r1.duration_seconds, 3),
            total_chunks=r1.total_chunks,
            new_chunks=r1.new_chunks,
            updated_chunks=r1.updated_chunks,
            deleted_chunks=r1.deleted_chunks,
            skipped_chunks=r1.skipped_chunks,
            tokens_used=r1.tokens_used,
            tokens_saved=r1.tokens_saved,
            api_calls_made=adapter.upsert_count,
            api_calls_avoided=0,  # first sync — nothing to avoid
            savings_pct=0.0,
        ))
        print(f"  → {r1.total_chunks:,} chunks, {r1.duration_seconds:.2f}s\n")

        print("run 2/4: no changes (second sync, nothing edited)...")
        adapter.reset_counters()
        r2 = sync(
            source=str(work_dir),
            vector_db=adapter,
            embed_fn=mock_embed,
            registry_path=str(db_path),
            verbose=False,
        )
        calls_avoided_r2 = r2.skipped_chunks
        savings_r2 = (calls_avoided_r2 / r1.total_chunks * 100) if r1.total_chunks else 0
        runs.append(RunResult(
            label="no changes",
            duration_seconds=round(r2.duration_seconds, 3),
            total_chunks=r2.total_chunks,
            new_chunks=r2.new_chunks,
            updated_chunks=r2.updated_chunks,
            deleted_chunks=r2.deleted_chunks,
            skipped_chunks=r2.skipped_chunks,
            tokens_used=r2.tokens_used,
            tokens_saved=r2.tokens_saved,
            api_calls_made=adapter.upsert_count,
            api_calls_avoided=calls_avoided_r2,
            savings_pct=round(savings_r2, 1),
        ))
        print(f"  → {r2.skipped_chunks:,} chunks skipped, "
              f"{adapter.upsert_count} API calls, "
              f"{savings_r2:.0f}% savings\n")

        n_to_edit = max(1, int(len(doc_files) * edit_pct))
        docs_to_edit = random.sample(doc_files, n_to_edit)
        print(f"run 3/4: editing {n_to_edit} of {len(doc_files)} documents "
              f"({edit_pct*100:.0f}%)...")

        for doc_path in tqdm(docs_to_edit, desc="  editing", leave=False):
            original = doc_path.read_text(encoding="utf-8")
            amendment = (
                f"\n\n== Amendment (benchmark edit) ==\n"
                f"This section was added during benchmarking to simulate "
                f"a real document update. Timestamp: {time.time():.0f}.\n"
            )
            doc_path.write_text(original + amendment, encoding="utf-8")

        adapter.reset_counters()
        r3 = sync(
            source=str(work_dir),
            vector_db=adapter,
            embed_fn=mock_embed,
            registry_path=str(db_path),
            verbose=False,
        )
        calls_would_have_been = r3.total_chunks
        calls_avoided_r3 = r3.skipped_chunks
        savings_r3 = (calls_avoided_r3 / calls_would_have_been * 100) if calls_would_have_been else 0
        runs.append(RunResult(
            label=f"edit {edit_pct*100:.0f}% of docs",
            duration_seconds=round(r3.duration_seconds, 3),
            total_chunks=r3.total_chunks,
            new_chunks=r3.new_chunks,
            updated_chunks=r3.updated_chunks,
            deleted_chunks=r3.deleted_chunks,
            skipped_chunks=r3.skipped_chunks,
            tokens_used=r3.tokens_used,
            tokens_saved=r3.tokens_saved,
            api_calls_made=adapter.upsert_count,
            api_calls_avoided=calls_avoided_r3,
            savings_pct=round(savings_r3, 1),
        ))
        print(f"  → {r3.updated_chunks} chunks updated, "
              f"{r3.skipped_chunks:,} skipped, "
              f"{savings_r3:.0f}% savings\n")

        n_to_delete = max(1, int(len(doc_files) * 0.05))
        docs_to_delete = random.sample(
            [f for f in doc_files if f not in docs_to_edit], n_to_delete
        )
        print(f"run 4/4: deleting {n_to_delete} documents...")

        for doc_path in docs_to_delete:
            doc_path.unlink()

        adapter.reset_counters()
        r4 = sync(
            source=str(work_dir),
            vector_db=adapter,
            embed_fn=mock_embed,
            registry_path=str(db_path),
            verbose=False,
        )
        runs.append(RunResult(
            label=f"delete 5% of docs",
            duration_seconds=round(r4.duration_seconds, 3),
            total_chunks=r4.total_chunks,
            new_chunks=r4.new_chunks,
            updated_chunks=r4.updated_chunks,
            deleted_chunks=r4.deleted_chunks,
            skipped_chunks=r4.skipped_chunks,
            tokens_used=r4.tokens_used,
            tokens_saved=r4.tokens_saved,
            api_calls_made=adapter.upsert_count,
            api_calls_avoided=r4.skipped_chunks,
            savings_pct=0.0,
        ))
        print(f"  → {r4.deleted_chunks} chunks removed from index\n")

        return BenchmarkResult(
            dataset=data_dir.name,
            doc_count=doc_count,
            total_chars=total_chars,
            avg_chars_per_doc=total_chars // doc_count,
            edit_pct=edit_pct,
            runs=runs,
        )

    finally:
        shutil.rmtree(tmpdir)

def print_results(result: BenchmarkResult):
    print(f"\n{'═' * 70}")
    print(f"  BENCHMARK RESULTS — {result.dataset}")
    print(f"  {result.doc_count} docs · {result.total_chars:,} chars · "
          f"{result.avg_chars_per_doc:,} avg chars/doc")
    print(f"{'═' * 70}\n")

    header = f"{'run':<22} {'chunks':>8} {'skipped':>8} {'api calls':>10} {'savings':>8} {'time':>8}"
    print(header)
    print("─" * 70)

    for run in result.runs:
        print(
            f"{run.label:<22} "
            f"{run.total_chunks:>8,} "
            f"{run.skipped_chunks:>8,} "
            f"{run.api_calls_made:>10,} "
            f"{run.savings_pct:>7.1f}% "
            f"{run.duration_seconds:>7.2f}s"
        )

    print("─" * 70)

    no_change_run = next((r for r in result.runs if r.label == "no changes"), None)
    edit_run = next((r for r in result.runs if "edit" in r.label), None)

    if no_change_run:
        print(f"\n  no-change sync  : {no_change_run.savings_pct:.0f}% of API calls avoided "
              f"({no_change_run.skipped_chunks:,} chunks skipped)")
    if edit_run:
        print(f"  {edit_run.label:<16}: {edit_run.savings_pct:.0f}% of API calls avoided "
              f"({edit_run.skipped_chunks:,} chunks skipped)")

    delete_run = next((r for r in result.runs if "delete" in r.label), None)
    if delete_run:
        print(f"  deletion sync   : {delete_run.deleted_chunks} orphaned chunks "
              f"removed from index")
    print()


def save_results(result: BenchmarkResult, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"results_{result.dataset}_{timestamp}.json"
    output_path.write_text(
        json.dumps(asdict(result), indent=2),
        encoding="utf-8"
    )
    print(f"  results saved → {output_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run chunks-sync benchmark against a Wikipedia dataset."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="benchmarks/data/wiki_100",
        help="Path to dataset directory (default: benchmarks/data/wiki_100)"
    )
    parser.add_argument(
        "--edit-pct",
        type=float,
        default=0.05,
        help="Fraction of docs to edit in run 3 (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"\nError: {data_dir} not found.")
        print("Run this first:")
        print(f"  uv run python benchmarks/download_dataset.py --articles 100\n")
        sys.exit(1)

    result = run_benchmark(data_dir, edit_pct=args.edit_pct)
    print_results(result)
    save_results(result, Path("benchmarks/results"))