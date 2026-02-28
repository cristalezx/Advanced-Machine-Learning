"""
Abstract base interfaces — LLM、Embedding、VectorStore 均可独立替换。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class BaseLLMClient(ABC):
    """LLM 接口：用于从对话中提取记忆和客户画像。"""

    @abstractmethod
    async def extract_memories(
        self,
        messages: List[Dict[str, str]],
        existing_memories: List[str],
    ) -> List[Dict[str, Any]]:
        """
        从对话消息中提取新记忆片段。

        Args:
            messages: 对话消息列表，每条包含 role / content
            existing_memories: 该用户已有的记忆文本列表（用于去重/合并）

        Returns:
            记忆条目列表，每条至少含 {"content": "..."} 字段
        """

    @abstractmethod
    async def extract_profile(
        self,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        从对话消息中提取客户画像字段。

        Returns:
            与 CustomerProfile 字段对应的 dict，缺失字段可省略
        """

    @abstractmethod
    async def merge_memory(
        self,
        existing: str,
        new_info: str,
    ) -> str:
        """
        将已有记忆与新信息合并为单条更新后的记忆文本。
        """


class BaseEmbeddingClient(ABC):
    """Embedding 接口：文本向量化。"""

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """将单条文本转为向量。"""

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化。"""


class BaseVectorStore(ABC):
    """向量存储接口：支持 upsert / search / get / delete。"""

    @abstractmethod
    async def upsert(
        self,
        id: str,
        embedding: List[float],
        payload: Dict[str, Any],
    ) -> None:
        """插入或更新一条记录（id 相同则覆盖）。"""

    @abstractmethod
    async def search(
        self,
        embedding: List[float],
        top_k: int,
        filters: Dict[str, Any],
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        语义检索。

        Returns:
            [(id, score, payload), ...]，按 score 降序
        """

    @abstractmethod
    async def get(self, id: str) -> Optional[Dict[str, Any]]:
        """根据 id 精确获取 payload，不存在则返回 None。"""

    @abstractmethod
    async def delete(self, id: str) -> None:
        """删除指定 id 的记录。"""

    @abstractmethod
    async def list_by_user(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """列出某用户的所有记忆（可按类型过滤）。"""
