from abc import ABC, abstractmethod
from typing import Any

class VectorDBAdapter(ABC):
    """
    Abstract adapter. Implement for each vector DB.
    All methods receive chunk_id as the stable identifier.
    """
    @abstractmethod
    def upsert(self, chunk_id: str, vector: list[float], metadata: dict):
        """Insert or update a chunk vector + metadata."""
    @abstractmethod
    def patch_metadata(self, chunk_id: str, metadata: dict):
        """Update metadata only — no vector change. For ACL/permission updates."""
    @abstractmethod
    def delete(self, chunk_ids: list[str]):
        """Delete chunks by ID. Called on document deletion."""

class PineconeAdapter(VectorDBAdapter):
    """
    Adapter for Pinecone serverless and pod-based indexes.

    Usage:
        import pinecone
        pc = pinecone.Pinecone(api_key="...")
        index = pc.Index("your-index")
        adapter = PineconeAdapter(index)
    """

    def __init__(self, index: Any, namespace: str = ""):
        self.index = index
        self.namespace = namespace

    def upsert(self, chunk_id: str, vector: list[float], metadata: dict):
        self.index.upsert(
            vectors=[{"id": chunk_id, "values": vector, "metadata": metadata}],
            namespace=self.namespace,
        )

    def patch_metadata(self, chunk_id: str, metadata: dict):
        self.index.update(
            id=chunk_id,
            set_metadata=metadata,
            namespace=self.namespace,
        )

    def delete(self, chunk_ids: list[str]):
        if chunk_ids:
            self.index.delete(ids=chunk_ids, namespace=self.namespace)

class QdrantAdapter(VectorDBAdapter):
    """
    Adapter for Qdrant (local or cloud).

    Usage:
        from qdrant_client import QdrantClient
        client = QdrantClient(url="http://localhost:6333")
        adapter = QdrantAdapter(client, collection_name="your-collection")
    """

    def __init__(self, client: Any, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def upsert(self, chunk_id: str, vector: list[float], metadata: dict):
        from qdrant_client.models import PointStruct
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=chunk_id, vector=vector, payload=metadata)],
        )

    def patch_metadata(self, chunk_id: str, metadata: dict):
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=metadata,
            points=[chunk_id],
        )

    def delete(self, chunk_ids: list[str]):
        if chunk_ids:
            from qdrant_client.models import PointIdsList
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=chunk_ids),
            )


class WeaviateAdapter(VectorDBAdapter):
    """
    Adapter for Weaviate (local or cloud).

    Usage:
        import weaviate
        client = weaviate.connect_to_local()
        adapter = WeaviateAdapter(client, class_name="Document")
    """

    def __init__(self, client: Any, class_name: str):
        self.client = client
        self.class_name = class_name

    def upsert(self, chunk_id: str, vector: list[float], metadata: dict):
        collection = self.client.collections.get(self.class_name)
        collection.data.insert(
            properties=metadata,
            vector=vector,
            uuid=chunk_id,
        )

    def patch_metadata(self, chunk_id: str, metadata: dict):
        collection = self.client.collections.get(self.class_name)
        collection.data.update(uuid=chunk_id, properties=metadata)

    def delete(self, chunk_ids: list[str]):
        collection = self.client.collections.get(self.class_name)
        for cid in chunk_ids:
            collection.data.delete_by_id(cid)