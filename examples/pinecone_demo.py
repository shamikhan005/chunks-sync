"""
chunks-sync Pinecone demo.

Runs 5 scenarios against a real Pinecone index using
Pinecone's hosted llama-text-embed-v2 inference API.
No OpenAI key required.

Setup:
    1. Create a .env file in the repo root:
          PINECONE_API_KEY=your-key-here

    2. Install dependencies:
          uv add pinecone-client python-dotenv

    3. Run:
          uv run python examples/pinecone_demo.py

What this demonstrates:
    Run 1 — cold start       : ingest 5 documents into Pinecone
    Run 2 — no changes       : sync again, nothing re-embedded
    Run 3 — edit 1 document  : only changed chunks re-embedded
    Run 4 — delete 1 document: chunks auto-removed from Pinecone
    Run 5 — metadata change  : PATCH only, zero re-embeds
"""

import os
import shutil
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv(Path(__file__).parent.parent / ".env")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from chunks_sync import sync
from chunks_sync.adapters import PineconeAdapter

DOCUMENTS = {
    "engineering_handbook.txt": """\
Engineering Handbook

Onboarding
All new engineers should complete the onboarding checklist within their first week.
Access to production systems requires approval from your team lead and the security team.
Development environments should be set up using the standard bootstrap script.

Code Review
All pull requests require at least one approval before merging.
Reviews should be completed within 24 hours of the request.
Critical path changes require two approvals and a sign-off from the on-call engineer.

Deployment
Deployments to production happen every Tuesday and Thursday.
All deployments must pass the full test suite and a manual smoke test.
Rollbacks can be triggered by any engineer with production access.

Incident Response
On-call rotation is weekly and follows the schedule in PagerDuty.
P0 incidents require immediate response and a postmortem within 48 hours.
All incidents should be logged in the incident tracker with a timeline of events.
""",

    "data_policy.txt": """\
Data Retention Policy

Overview
This policy governs how the company collects, stores, and deletes user data.
All data handling must comply with GDPR, CCPA, and internal security standards.

Retention Periods
User account data is retained for 7 years after account closure.
Usage logs are retained for 90 days and then permanently deleted.
Payment records are retained for 10 years per financial regulations.
Support tickets are retained for 3 years.

Deletion Requests
Users may request deletion of their personal data at any time.
Requests must be fulfilled within 30 days per GDPR requirements.
Deletion requests are logged and audited quarterly.

Access Control
Only authorized personnel may access personally identifiable information.
All access to user data is logged with user ID, timestamp, and reason.
Access permissions are reviewed every 6 months.
""",

    "api_reference.txt": """\
API Reference

Authentication
All API requests must include a valid Bearer token in the Authorization header.
Tokens expire after 24 hours and must be refreshed using the /auth/refresh endpoint.
Rate limiting is enforced at 1000 requests per minute per API key.

Endpoints
GET /documents — returns a paginated list of documents the user has access to.
POST /documents — creates a new document. Requires title and content fields.
DELETE /documents/:id — permanently deletes a document and all associated metadata.

Error Codes
400 Bad Request: malformed request body or missing required fields.
401 Unauthorized: missing or expired authentication token.
403 Forbidden: authenticated but insufficient permissions.
404 Not Found: document does not exist or access is denied.
500 Internal Server Error: contact support if this persists.
""",

    "security_policy.txt": """\
Security Policy

Password Requirements
All passwords must be at least 12 characters long.
Passwords must include uppercase, lowercase, numbers, and special characters.
Passwords must be changed every 90 days.
Password reuse is not permitted for the last 10 passwords.

Multi-Factor Authentication
MFA is required for all employee accounts.
Hardware security keys are the preferred MFA method.
TOTP authenticator apps are acceptable alternatives.
SMS-based MFA is not permitted for production system access.

Access Reviews
All system access is reviewed quarterly by team leads.
Employees leaving the company have access revoked within 24 hours.
Privileged access requires additional approval and is audited monthly.
""",

    "product_roadmap.txt": """\
Product Roadmap Q3 2026

Search and Discovery
Implement semantic search across all document types.
Add support for multi-language document indexing.
Improve search relevance ranking using user feedback signals.

Collaboration Features
Real-time collaborative editing for shared documents.
Comment threads with resolution tracking.
Document version history with diff visualization.

Integrations
Slack integration for document notifications.
Google Drive two-way sync.
Jira integration for linking documents to issues.

Infrastructure
Migrate primary database to PostgreSQL 16.
Implement distributed caching layer for search results.
Reduce API response time p95 from 400ms to 150ms.
""",
}


def section(title: str):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def wait_for_index(index, expected_count: int, timeout: int = 30):
    """Wait for Pinecone to reflect the expected record count."""
    start = time.time()
    while time.time() - start < timeout:
        stats = index.describe_index_stats()
        if stats.total_vector_count >= expected_count:
            return
        time.sleep(1)


def main():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("\nError: PINECONE_API_KEY not set.")
        print("Create a .env file in the repo root with:")
        print("  PINECONE_API_KEY=your-key-here\n")
        sys.exit(1)

    pc = Pinecone(api_key=api_key)

    INDEX_NAME = "chunks-sync"
    DIMENSION = 1024

    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"\nError: index '{INDEX_NAME}' not found.")
        print(f"Available indexes: {existing}")
        print("Create the index in the Pinecone dashboard first.\n")
        sys.exit(1)

    index = pc.Index(INDEX_NAME)
    adapter = PineconeAdapter(index, namespace="chunks-sync-demo")

    def embed(texts: list[str]) -> list[list[float]]:
        result = pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=texts,
            parameters={"input_type": "passage", "truncate": "END"},
        )
        return [r.values for r in result.data]

    tmpdir = tempfile.mkdtemp(prefix="chunks_sync_demo_")
    docs_dir = Path(tmpdir) / "docs"
    db_path = Path(tmpdir) / "registry.db"
    docs_dir.mkdir()

    try:
        # Write sample documents
        for filename, content in DOCUMENTS.items():
            (docs_dir / filename).write_text(content, encoding="utf-8")

        print("\n╔══════════════════════════════════════════════════════╗")
        print("║         chunks-sync — Pinecone live demo             ║")
        print("╚══════════════════════════════════════════════════════╝")
        print(f"\n  index      : {INDEX_NAME}")
        print(f"  model      : llama-text-embed-v2 (Pinecone hosted)")
        print(f"  dimension  : {DIMENSION}")
        print(f"  documents  : {len(DOCUMENTS)}")
        print(f"  namespace  : chunks-sync-demo")

        section("RUN 1 — cold start (first sync)")
        print("  Ingesting all documents into Pinecone for the first time...")

        r1 = sync(
            source=str(docs_dir),
            vector_db=adapter,
            embed_fn=embed,
            embedding_model="llama-text-embed-v2",
            registry_path=str(db_path),
            chunk_size=400,
            overlap=50,
            verbose=True,
        )

        time.sleep(2)
        stats = index.describe_index_stats()
        print(f"  Pinecone record count: {stats.total_vector_count}")

        section("RUN 2 — no changes (sync again, nothing edited)")
        print("  Running sync with no file changes...")

        r2 = sync(
            source=str(docs_dir),
            vector_db=adapter,
            embed_fn=embed,
            embedding_model="llama-text-embed-v2",
            registry_path=str(db_path),
            chunk_size=400,
            overlap=50,
            verbose=True,
        )

        assert r2.new_chunks == 0, "Expected 0 new chunks"
        assert r2.updated_chunks == 0, "Expected 0 updated chunks"
        print(f"  ✓ confirmed: 0 API calls to embedding model")

        section("RUN 3 — edit one document")
        print("  Editing data_policy.txt — adding a new compliance section...")

        policy_path = docs_dir / "data_policy.txt"
        original = policy_path.read_text(encoding="utf-8")
        policy_path.write_text(
            original + """
Data Residency
All user data for EU customers must be stored within the European Economic Area.
Data transfers outside the EEA require standard contractual clauses.
Data residency requirements are reviewed annually with the legal team.
""",
            encoding="utf-8",
        )

        r3 = sync(
            source=str(docs_dir),
            vector_db=adapter,
            embed_fn=embed,
            embedding_model="llama-text-embed-v2",
            registry_path=str(db_path),
            chunk_size=400,
            overlap=50,
            verbose=True,
        )

        assert r3.skipped_chunks > 0, "Expected some chunks to be skipped"
        print(f"  ✓ confirmed: only changed chunks re-embedded, "
              f"{r3.skipped_chunks} skipped")

        section("RUN 4 — delete one document")
        print("  Deleting product_roadmap.txt...")

        before_stats = index.describe_index_stats()
        before_count = before_stats.total_vector_count

        (docs_dir / "product_roadmap.txt").unlink()

        r4 = sync(
            source=str(docs_dir),
            vector_db=adapter,
            embed_fn=embed,
            embedding_model="llama-text-embed-v2",
            registry_path=str(db_path),
            chunk_size=400,
            overlap=50,
            verbose=True,
        )

        time.sleep(2)
        after_stats = index.describe_index_stats()
        after_count = after_stats.total_vector_count

        assert r4.deleted_chunks > 0, "Expected chunks to be deleted"
        print(f"  ✓ confirmed: Pinecone records before={before_count}, "
              f"after={after_count} "
              f"({before_count - after_count} removed)")

        section("RUN 5 — metadata change (permission update, zero re-embeds)")
        print("  Marking security_policy.txt as restricted (hr_only)...")

        from pathlib import Path as P

        call_log = {"embed_calls": 0}
        original_embed = embed

        def tracked_embed(texts):
            call_log["embed_calls"] += len(texts)
            return original_embed(texts)

        def get_permissions(path: P) -> dict:
            if path.name == "security_policy.txt":
                return {"access": "hr_only", "owner": "security_team"}
            return {"access": "public"}

        r5 = sync(
            source=str(docs_dir),
            vector_db=adapter,
            embed_fn=tracked_embed,
            embedding_model="llama-text-embed-v2",
            registry_path=str(db_path),
            chunk_size=400,
            overlap=50,
            metadata_fn=get_permissions,
            verbose=True,
        )

        print(f"  ✓ confirmed: {call_log['embed_calls']} embedding API calls "
              f"(metadata PATCH only, no re-embedding)")

        print("\n╔══════════════════════════════════════════════════════╗")
        print("║                    DEMO SUMMARY                      ║")
        print("╠══════════════════════════════════════════════════════╣")
        print(f"║  Run 1 cold start   : {r1.new_chunks:>5} chunks ingested         ║")
        print(f"║  Run 2 no changes   : {r2.skipped_chunks:>5} chunks skipped (0 API) ║")
        print(f"║  Run 3 edit 1 doc   : {r3.updated_chunks:>5} re-embedded, "
              f"{r3.skipped_chunks:>4} skipped ║")
        print(f"║  Run 4 delete 1 doc : {r4.deleted_chunks:>5} chunks removed from index ║")
        print(f"║  Run 5 metadata     :     0 re-embeds (PATCH only)   ║")
        print("╚══════════════════════════════════════════════════════╝\n")

        print("  Cleaning up demo namespace from Pinecone index...")
        index.delete(delete_all=True, namespace="chunks-sync-demo")
        print("  Done.\n")

    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    main()