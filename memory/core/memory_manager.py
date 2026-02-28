"""
MemoryManager — 核心编排器。
负责协调 LLM、Embedding、VectorStore 完成记忆的插入、更新、查询。
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..base import BaseLLMClient, BaseEmbeddingClient, BaseVectorStore
from ..models import (
    ChannelType,
    CustomerProfile,
    MemoryItem,
    MemoryQuery,
    MemorySearchResult,
    MemoryType,
)

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        embedding_client: BaseEmbeddingClient,
        vector_store: BaseVectorStore,
    ) -> None:
        self.llm = llm_client
        self.embedder = embedding_client
        self.store = vector_store

    # ------------------------------------------------------------------
    # 基础 CRUD
    # ------------------------------------------------------------------

    async def insert(self, item: MemoryItem) -> MemoryItem:
        """插入一条新记忆（自动生成 embedding 并写入向量库）。"""
        embedding = await self.embedder.embed(item.content)
        await self.store.upsert(
            id=item.id,
            embedding=embedding,
            payload=item.to_store_payload(),
        )
        logger.debug("Inserted memory %s for user %s", item.id, item.user_id)
        return item

    async def update(
        self,
        memory_id: str,
        updates: Dict[str, Any],
    ) -> MemoryItem:
        """
        更新已有记忆。若 content 发生变化则重新计算 embedding。

        Raises:
            KeyError: memory_id 不存在时抛出
        """
        payload = await self.store.get(memory_id)
        if payload is None:
            raise KeyError(f"Memory {memory_id} not found")

        item = MemoryItem.from_store_payload(payload)

        # 合并更新字段
        for key, value in updates.items():
            if key == "profile" and isinstance(value, dict):
                value = CustomerProfile(**value)
            if hasattr(item, key):
                setattr(item, key, value)

        item.updated_at = datetime.utcnow()

        # content 变化时重新向量化
        embedding = await self.embedder.embed(item.content)
        await self.store.upsert(
            id=item.id,
            embedding=embedding,
            payload=item.to_store_payload(),
        )
        logger.debug("Updated memory %s", memory_id)
        return item

    async def delete(self, memory_id: str) -> None:
        """删除指定记忆。"""
        await self.store.delete(memory_id)

    async def get(self, memory_id: str) -> Optional[MemoryItem]:
        """根据 id 获取单条记忆。"""
        payload = await self.store.get(memory_id)
        return MemoryItem.from_store_payload(payload) if payload else None

    async def query(self, query: MemoryQuery) -> List[MemorySearchResult]:
        """
        语义检索记忆，支持按 memory_type / source_channel 过滤。
        """
        query_embedding = await self.embedder.embed(query.query)

        filters: Dict[str, Any] = {"user_id": query.user_id}
        if query.memory_types:
            filters["memory_type"] = [t.value for t in query.memory_types]
        if query.source_channels:
            filters["source_channel"] = [c.value for c in query.source_channels]
        filters.update(query.filters)

        raw = await self.store.search(
            embedding=query_embedding,
            top_k=query.top_k,
            filters=filters,
        )

        results = []
        for _id, score, payload in raw:
            if score < query.score_threshold:
                continue
            results.append(
                MemorySearchResult(
                    item=MemoryItem.from_store_payload(payload),
                    score=score,
                )
            )
        return results

    async def list_all(
        self,
        user_id: str,
        memory_type: Optional[MemoryType] = None,
    ) -> List[MemoryItem]:
        """列出某用户的全部记忆（不做语义排序）。"""
        payloads = await self.store.list_by_user(
            user_id=user_id,
            memory_type=memory_type.value if memory_type else None,
        )
        return [MemoryItem.from_store_payload(p) for p in payloads]

    # ------------------------------------------------------------------
    # 高层操作：对话记忆提取（mem0 风格）
    # ------------------------------------------------------------------

    async def add_from_conversation(
        self,
        user_id: str,
        messages: List[Dict[str, str]],
        extract_profile: bool = True,
    ) -> List[MemoryItem]:
        """
        从一段对话中提取记忆并持久化。

        步骤：
        1. 检索当前已有的近似记忆（用于让 LLM 去重）
        2. LLM 提取新记忆片段
        3. （可选）提取客户画像信息并更新
        """
        # 用最近几条消息做候选检索
        recent_text = " ".join(
            m.get("content", "") for m in messages[-4:] if m.get("content")
        )
        existing_results = await self.query(
            MemoryQuery(user_id=user_id, query=recent_text, top_k=10)
        )
        existing_texts = [r.item.content for r in existing_results]

        extracted = await self.llm.extract_memories(messages, existing_texts)

        saved: List[MemoryItem] = []
        for mem in extracted:
            item = MemoryItem(
                user_id=user_id,
                memory_type=MemoryType.CONVERSATIONAL,
                content=mem["content"],
                metadata=mem.get("metadata", {}),
                source_channel=ChannelType.CHAT,
            )
            saved.append(await self.insert(item))

        if extract_profile:
            profile_data = await self.llm.extract_profile(messages)
            if any(v for v in profile_data.values() if v):
                await self.update_customer_profile(user_id, profile_data)

        logger.info(
            "Extracted %d memories from conversation for user %s",
            len(saved),
            user_id,
        )
        return saved

    # ------------------------------------------------------------------
    # 高层操作：客户画像
    # ------------------------------------------------------------------

    async def update_customer_profile(
        self,
        user_id: str,
        profile_updates: Dict[str, Any],
        channel: ChannelType = ChannelType.CHAT,
    ) -> MemoryItem:
        """
        更新或创建用户画像记忆。
        若已存在则做字段级增量合并；否则新建。
        """
        existing_payloads = await self.store.list_by_user(
            user_id=user_id,
            memory_type=MemoryType.CUSTOMER_PROFILE.value,
            limit=1,
        )

        if existing_payloads:
            item = MemoryItem.from_store_payload(existing_payloads[0])
            existing_profile = item.profile or CustomerProfile()

            # 增量合并：只覆盖非 None 的字段
            for key, value in profile_updates.items():
                if value is None:
                    continue
                if key == "preferences" and isinstance(value, dict):
                    existing_profile.preferences.update(value)
                elif key == "tags" and isinstance(value, list):
                    existing_profile.tags = list(
                        set(existing_profile.tags) | set(value)
                    )
                elif key == "custom_fields" and isinstance(value, dict):
                    existing_profile.custom_fields.update(value)
                elif hasattr(existing_profile, key):
                    setattr(existing_profile, key, value)

            return await self.update(
                item.id,
                {
                    "profile": existing_profile.model_dump(),
                    "content": _profile_to_text(existing_profile),
                    "source_channel": channel.value,
                },
            )
        else:
            profile = CustomerProfile(
                **{
                    k: v
                    for k, v in profile_updates.items()
                    if hasattr(CustomerProfile.model_fields, k) or k in CustomerProfile.model_fields
                }
            )
            item = MemoryItem(
                user_id=user_id,
                memory_type=MemoryType.CUSTOMER_PROFILE,
                content=_profile_to_text(profile),
                profile=profile,
                source_channel=channel,
            )
            return await self.insert(item)

    async def get_customer_profile(
        self, user_id: str
    ) -> Optional[CustomerProfile]:
        """获取用户画像，不存在则返回 None。"""
        payloads = await self.store.list_by_user(
            user_id=user_id,
            memory_type=MemoryType.CUSTOMER_PROFILE.value,
            limit=1,
        )
        if not payloads:
            return None
        item = MemoryItem.from_store_payload(payloads[0])
        return item.profile

    # ------------------------------------------------------------------
    # 高层操作：外部渠道数据
    # ------------------------------------------------------------------

    async def add_external_channel_data(
        self,
        user_id: str,
        data: Dict[str, Any],
        channel: ChannelType,
        content_text: Optional[str] = None,
    ) -> MemoryItem:
        """
        将外部渠道（CRM、社交、电话等）带来的客户信息写入记忆。

        Args:
            data: 原始数据 dict，会保存在 metadata
            channel: 数据来源渠道
            content_text: 用于向量化的文本描述；若为空则自动序列化 data
        """
        import json

        content = content_text or json.dumps(data, ensure_ascii=False)
        item = MemoryItem(
            user_id=user_id,
            memory_type=MemoryType.EXTERNAL_CHANNEL,
            content=content,
            metadata=data,
            source_channel=channel,
        )
        return await self.insert(item)


# ------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------

def _profile_to_text(profile: CustomerProfile) -> str:
    parts = []
    if profile.name:
        parts.append(f"姓名: {profile.name}")
    if profile.email:
        parts.append(f"邮箱: {profile.email}")
    if profile.phone:
        parts.append(f"电话: {profile.phone}")
    if profile.age:
        parts.append(f"年龄: {profile.age}")
    if profile.gender:
        parts.append(f"性别: {profile.gender}")
    if profile.location:
        parts.append(f"地区: {profile.location}")
    if profile.occupation:
        parts.append(f"职业: {profile.occupation}")
    if profile.preferences:
        parts.append(f"偏好: {profile.preferences}")
    if profile.tags:
        parts.append(f"标签: {', '.join(profile.tags)}")
    if profile.custom_fields:
        parts.append(f"自定义: {profile.custom_fields}")
    return "; ".join(parts) if parts else "客户画像（暂无信息）"
