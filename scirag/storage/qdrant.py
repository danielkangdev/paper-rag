"""Qdrant vector index for chunks.

Default is qdrant-client's embedded local mode (``path=./data/qdrant``), so no
server or Docker is required; set ``qdrant.url`` to use a Qdrant server.
Only ids are stored in the payload; chunk text lives in SQLite.
"""

from __future__ import annotations

import uuid

_NS = uuid.UUID("6f1c3b8e-2d4a-4b8e-9a51-5c0f2e7d9a10")


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NS, chunk_id))


class QdrantStore:
    def __init__(self, *, url: str | None, path: str, collection: str, dim: int):
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=url) if url else QdrantClient(path=path)
        self.collection = collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from qdrant_client import models

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
            )

    def upsert(self, items: list[tuple[str, str, list[float]]]) -> int:
        """items: (chunk_id, paper_id, vector)."""
        from qdrant_client import models

        points = [
            models.PointStruct(id=point_id(cid), vector=vec, payload={"chunk_id": cid, "paper_id": pid})
            for cid, pid, vec in items
        ]
        for i in range(0, len(points), 256):
            self.client.upsert(self.collection, points=points[i : i + 256], wait=True)
        return len(points)

    def delete_paper(self, paper_id: str) -> None:
        from qdrant_client import models

        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="paper_id", match=models.MatchValue(value=paper_id))]
                )
            ),
            wait=True,
        )

    def search(self, vector: list[float], top_k: int) -> list[tuple[str, float]]:
        res = self.client.query_points(self.collection, query=vector, limit=top_k, with_payload=True)
        return [(p.payload["chunk_id"], float(p.score)) for p in res.points]

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def close(self) -> None:
        self.client.close()
