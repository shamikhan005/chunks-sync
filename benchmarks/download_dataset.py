"""
Download Wikipedia articles for benchmarking.

Usage:
    uv run python benchmarks/download_dataset.py --articles 100
    uv run python benchmarks/download_dataset.py --articles 1000

Articles are saved as individual .txt files in benchmarks/data/wiki_100/
or benchmarks/data/wiki_1000/ etc.

Uses Wikipedia's "featured articles" list — these are long, high-quality,
varied in topic. Good proxy for a real enterprise document corpus.
"""

import argparse
import json
import time
from pathlib import Path

import wikipediaapi
from tqdm import tqdm

ARTICLE_TITLES = [
    "Python (programming language)", "Transformer (deep learning architecture)",
    "BERT (language model)", "GPT-4", "Large language model",
    "Retrieval-augmented generation", "Vector database", "Word2vec",
    "Attention mechanism", "Recurrent neural network",
    "Convolutional neural network", "Random forest", "Support vector machine",
    "Gradient boosting", "Reinforcement learning",
    "Alan Turing", "Claude Shannon", "John von Neumann", "Ada Lovelace",
    "Grace Hopper", "Donald Knuth", "Edsger Dijkstra", "Linus Torvalds",
    "Tim Berners-Lee", "Vint Cerf",
    "Linux", "Git", "Docker (software)", "Kubernetes", "PostgreSQL",
    "MongoDB", "Redis", "Apache Kafka", "Elasticsearch", "TensorFlow",
    "PyTorch", "NumPy", "Pandas (software)", "Scikit-learn", "Jupyter",
    "Amazon Web Services", "Microsoft Azure", "Google Cloud Platform",
    "Internet", "World Wide Web", "HTTP", "TCP/IP", "DNS",
    "Encryption", "Public-key cryptography", "Transport Layer Security",
    "Blockchain", "Bitcoin", "Ethereum",
    "Climate change", "Global warming", "Renewable energy", "Solar panel",
    "Wind power", "Nuclear power", "Carbon capture",
    "COVID-19 pandemic", "mRNA vaccine", "CRISPR", "Human genome",
    "Protein folding", "AlphaFold", "Neuroscience", "Consciousness",
    "Artificial intelligence", "Machine learning", "Deep learning",
    "Computer vision", "Natural language processing", "Speech recognition",
    "Autonomous vehicle", "Robotics", "Quantum computing", "Semiconductor",
    "Moore's law", "Integrated circuit", "Central processing unit",
    "Graphics processing unit", "Memory management", "Operating system",
    "Compiler", "Algorithm", "Data structure", "Binary search tree",
    "Hash table", "Graph theory", "Sorting algorithm", "Dynamic programming",
    "Byzantine fault tolerance", "CAP theorem", "MapReduce", "Hadoop",
    "Apache Spark", "Data warehouse", "ETL", "Data lake",
    "United States", "United Kingdom", "India", "China", "European Union",
    "World War II", "Cold War", "Industrial Revolution", "Renaissance",
    "Ancient Rome", "Ancient Greece", "Silk Road", "Colonialism",
    "French Revolution", "American Revolution", "Russian Revolution",
    "Economics", "Capitalism", "Macroeconomics", "Game theory",
    "Supply and demand", "Inflation", "Monetary policy", "Stock market",
]

def download_articles(n: int, output_dir: Path, lang: str = "en"):
    wiki = wikipediaapi.Wikipedia(
        language=lang,
        user_agent="chunks-sync-benchmark/1.0 (github.com/shamikhan005/chunks-sync)"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    downloaded = 0
    skipped = 0

    titles = ARTICLE_TITLES[:n] if n <= len(ARTICLE_TITLES) else ARTICLE_TITLES
    if n > len(ARTICLE_TITLES):
        print(f"Note: only {len(ARTICLE_TITLES)} articles defined, downloading all.")

    print(f"\nDownloading {len(titles)} Wikipedia articles → {output_dir}\n")

    for title in tqdm(titles):
        page = wiki.page(title)
        if not page.exists():
            skipped += 1
            continue

        text = page.text.strip()
        if len(text) < 500:   
            skipped += 1
            continue

        safe_title = title.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
        filepath = output_dir / f"{safe_title}.txt"
        filepath.write_text(text, encoding="utf-8")

        manifest.append({
            "title": title,
            "filename": filepath.name,
            "chars": len(text),
            "words": len(text.split()),
        })
        downloaded += 1
        time.sleep(0.1)  

    manifest_path = output_dir / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_chars = sum(a["chars"] for a in manifest)
    avg_chars = total_chars // len(manifest) if manifest else 0

    print(f"\n── download complete ──────────────────────")
    print(f"  downloaded  : {downloaded}")
    print(f"  skipped     : {skipped}")
    print(f"  total chars : {total_chars:,}")
    print(f"  avg per doc : {avg_chars:,} chars")
    print(f"  manifest    : {manifest_path}")
    print(f"───────────────────────────────────────────\n")

    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles", type=int, default=100,
                        help="Number of articles to download (default: 100)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: benchmarks/data/wiki_N)")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else \
        Path(f"benchmarks/data/wiki_{args.articles}")

    download_articles(args.articles, output_dir)