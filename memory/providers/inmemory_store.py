"""
纯内存向量存储 — 用于本地测试，无需外部依赖。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from ..base import BaseVectorStore


class InMemoryVectorStore(BaseVectorStore):
    """
    基于余弦相似度的内存向量库，不依赖任何外部服务。
    适合单元测试和快速原型。
    """

    def __init__(self) -> None:
        # {id: (embedding, payload)}
        self._data: Dict[str, Tuple[List[float], Dict[str, Any]]] = {}

    async def upsert(
        self,
        id: str,
        embedding: List[float],
        payload: Dict[str, Any],
    ) -> None:
        self._data[id] = (embedding, payload)

    async def search(
        self,
        embedding: List[float],
        top_k: int,
        filters: Dict[str, Any],
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        scored: List[Tuple[str, float, Dict[str, Any]]] = []
        for id, (vec, payload) in self._data.items():
            if not self._matches(payload, filters):
                continue
            score = _cosine(embedding, vec)
            scored.append((id, score, payload))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    async def get(self, id: str) -> Optional[Dict[str, Any]]:
        entry = self._data.get(id)
        return entry[1] if entry else None

    async def delete(self, id: str) -> None:
        self._data.pop(id, None)

    async def list_by_user(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        results = []
        for _, payload in self._data.values():
            if payload.get("user_id") != user_id:
                continue
            if memory_type and payload.get("memory_type") != memory_type:
                continue
            results.append(payload)
            if len(results) >= limit:
                break
        return results

    # ------------------------------------------------------------------

    @staticmethod
    def _matches(payload: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            pval = payload.get(key)
            if isinstance(value, list):
                if pval not in value:
                    return False
            else:
                if pval != value:
                    return False
        return True


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    denom = norm_a * norm_b
    return dot / denom if denom > 1e-9 else 0.0
