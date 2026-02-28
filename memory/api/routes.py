"""
FastAPI REST API — 对外暴露记忆组件的 HTTP 接口。

端点概览：
  POST   /memories                  插入单条记忆
  GET    /memories/{id}             获取单条记忆
  PUT    /memories/{id}             更新单条记忆
  DELETE /memories/{id}            删除单条记忆
  POST   /memories/query            语义检索
  GET    /memories/user/{user_id}   列出用户全部记忆

  POST   /memories/conversation     从对话提取并保存记忆（mem0 风格）

  GET    /profile/{user_id}         获取客户画像
  PUT    /profile/{user_id}         更新客户画像

  POST   /external                  写入外部渠道数据
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from ..core.memory_manager import MemoryManager
from ..models import (
    ChannelType,
    CustomerProfile,
    MemoryItem,
    MemoryQuery,
    MemorySearchResult,
    MemoryType,
)

app = FastAPI(
    title="Memory API",
    description="独立记忆组件，支持对话记忆、客户画像和多渠道数据管理。",
    version="1.0.0",
)

# 通过 app.state 注入 MemoryManager
# 启动时调用: app.state.memory = MemoryManager(...)


def _get_manager() -> MemoryManager:
    manager: Optional[MemoryManager] = getattr(app.state, "memory", None)
    if manager is None:
        raise RuntimeError(
            "MemoryManager not initialized. Set app.state.memory before starting."
        )
    return manager


# ============================================================
# Request / Response schemas
# ============================================================

class InsertRequest(BaseModel):
    user_id: str
    content: str
    memory_type: MemoryType = MemoryType.CONVERSATIONAL
    metadata: Dict[str, Any] = {}
    source_channel: Optional[ChannelType] = None


class UpdateRequest(BaseModel):
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    source_channel: Optional[ChannelType] = None


class ConversationRequest(BaseModel):
    user_id: str
    messages: List[Dict[str, str]]
    extract_profile: bool = True


class ProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    location: Optional[str] = None
    occupation: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    custom_fields: Optional[Dict[str, Any]] = None
    channel: ChannelType = ChannelType.CRM


class ExternalChannelRequest(BaseModel):
    user_id: str
    data: Dict[str, Any]
    channel: ChannelType
    content_text: Optional[str] = None


# ============================================================
# 基础 CRUD
# ============================================================

@app.post("/memories", response_model=MemoryItem, tags=["Memory"])
async def insert_memory(req: InsertRequest):
    """插入一条新记忆。"""
    manager = _get_manager()
    item = MemoryItem(
        user_id=req.user_id,
        content=req.content,
        memory_type=req.memory_type,
        metadata=req.metadata,
        source_channel=req.source_channel,
    )
    return await manager.insert(item)


@app.get("/memories/{memory_id}", response_model=MemoryItem, tags=["Memory"])
async def get_memory(memory_id: str):
    """根据 ID 获取单条记忆。"""
    manager = _get_manager()
    item = await manager.get(memory_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return item


@app.put("/memories/{memory_id}", response_model=MemoryItem, tags=["Memory"])
async def update_memory(memory_id: str, req: UpdateRequest):
    """更新已有记忆的内容或元数据。"""
    manager = _get_manager()
    updates = req.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    try:
        return await manager.update(memory_id, updates)
    except KeyError:
        raise HTTPException(status_code=404, detail="Memory not found")


@app.delete("/memories/{memory_id}", tags=["Memory"])
async def delete_memory(memory_id: str):
    """删除指定记忆。"""
    manager = _get_manager()
    await manager.delete(memory_id)
    return {"deleted": memory_id}


@app.get("/memories/user/{user_id}", response_model=List[MemoryItem], tags=["Memory"])
async def list_memories(
    user_id: str,
    memory_type: Optional[MemoryType] = Query(default=None),
):
    """列出某用户的全部记忆（可按类型过滤）。"""
    manager = _get_manager()
    return await manager.list_all(user_id=user_id, memory_type=memory_type)


# ============================================================
# 语义检索
# ============================================================

@app.post("/memories/query", response_model=List[MemorySearchResult], tags=["Query"])
async def query_memories(query: MemoryQuery):
    """
    对记忆库进行语义检索，返回最相关的记忆列表及相似度评分。
    """
    manager = _get_manager()
    return await manager.query(query)


# ============================================================
# 对话提取（mem0 风格）
# ============================================================

@app.post("/memories/conversation", response_model=List[MemoryItem], tags=["Conversation"])
async def process_conversation(req: ConversationRequest):
    """
    传入一段对话，自动提取重要记忆并（可选）更新客户画像。
    """
    manager = _get_manager()
    return await manager.add_from_conversation(
        user_id=req.user_id,
        messages=req.messages,
        extract_profile=req.extract_profile,
    )


# ============================================================
# 客户画像
# ============================================================

@app.get("/profile/{user_id}", response_model=Optional[CustomerProfile], tags=["Profile"])
async def get_profile(user_id: str):
    """获取用户的客户画像，不存在则返回 null。"""
    manager = _get_manager()
    return await manager.get_customer_profile(user_id)


@app.put("/profile/{user_id}", response_model=MemoryItem, tags=["Profile"])
async def update_profile(user_id: str, req: ProfileUpdateRequest):
    """
    更新（或创建）用户画像。支持增量合并：
    - preferences / custom_fields 做 dict merge
    - tags 做 set union
    - 其他字段直接覆盖（仅覆盖非 None 字段）
    """
    manager = _get_manager()
    profile_data = req.model_dump(exclude={"channel"}, exclude_none=True)
    return await manager.update_customer_profile(
        user_id=user_id,
        profile_updates=profile_data,
        channel=req.channel,
    )


# ============================================================
# 外部渠道数据
# ============================================================

@app.post("/external", response_model=MemoryItem, tags=["External Channel"])
async def add_external_data(req: ExternalChannelRequest):
    """
    写入来自外部渠道（CRM、社交媒体、电话等）的客户信息。
    数据保存为 EXTERNAL_CHANNEL 类型记忆，同样支持语义检索。
    """
    manager = _get_manager()
    return await manager.add_external_channel_data(
        user_id=req.user_id,
        data=req.data,
        channel=req.channel,
        content_text=req.content_text,
    )


# ============================================================
# 健康检查
# ============================================================

@app.get("/health", tags=["System"])
async def health():
    return {"status": "ok"}
