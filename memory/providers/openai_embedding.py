"""
OpenAI / 兼容 OpenAI 接口的 Embedding 客户端。
"""
from __future__ import annotations

from typing import List

import httpx
from openai import AsyncOpenAI

from ..base import BaseEmbeddingClient


class OpenAIEmbeddingClient(BaseEmbeddingClient):
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )
        self.model = model

    async def embed(self, text: str) -> List[float]:
        resp = await self.client.embeddings.create(
            input=text,
            model=self.model,
        )
        return resp.data[0].embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        resp = await self.client.embeddings.create(
            input=texts,
            model=self.model,
        )
        # 保证顺序与输入一致
        resp.data.sort(key=lambda d: d.index)
        return [d.embedding for d in resp.data]
