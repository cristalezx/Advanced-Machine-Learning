"""
Qdrant 向量存储实现。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
    ScrollRequest,
)

from ..base import BaseVectorStore

logger = logging.getLogger(__name__)


class QdrantVectorStore(BaseVectorStore):
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "memories",
        vector_size: int = 1536,
        api_key: Optional[str] = None,
    ) -> None:
        self.client = AsyncQdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._collection_ready = False

    async def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        collections = await self.client.get_collections()
        names = [c.name for c in collections.collections]
        if self.collection_name not in names:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection: %s", self.collection_name)
        self._collection_ready = True

    async def upsert(
        self,
        id: str,
        embedding: List[float],
        payload: Dict[str, Any],
    ) -> None:
        await self._ensure_collection()
        await self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=id, vector=embedding, payload=payload)],
        )

    async def search(
        self,
        embedding: List[float],
        top_k: int,
        filters: Dict[str, Any],
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        await self._ensure_collection()
        qdrant_filter = self._build_filter(filters)
        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [(str(r.id), r.score, r.payload or {}) for r in results]

    async def get(self, id: str) -> Optional[Dict[str, Any]]:
        await self._ensure_collection()
        points = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=[id],
            with_payload=True,
        )
        return points[0].payload if points else None

    async def delete(self, id: str) -> None:
        await self._ensure_collection()
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[id]),
        )

    async def list_by_user(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        await self._ensure_collection()
        conditions = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        if memory_type:
            conditions.append(
                FieldCondition(key="memory_type", match=MatchValue(value=memory_type))
            )
        results, _ = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=conditions),
            limit=limit,
            with_payload=True,
        )
        return [r.payload for r in results if r.payload]

    def _build_filter(self, filters: Dict[str, Any]) -> Optional[Filter]:
        if not filters:
            return None
        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append(
                    FieldCondition(key=key, match=MatchAny(any=value))
                )
            else:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        return Filter(must=conditions)
