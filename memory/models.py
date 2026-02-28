"""
Data models for the memory component.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    CONVERSATIONAL = "conversational"   # 对话中提取的记忆
    CUSTOMER_PROFILE = "customer_profile"  # 客户基础画像
    EXTERNAL_CHANNEL = "external_channel"  # 外部渠道数据


class ChannelType(str, Enum):
    CHAT = "chat"
    CRM = "crm"
    SOCIAL = "social"
    EMAIL = "email"
    PHONE = "phone"
    CUSTOM = "custom"


class MemoryAction(str, Enum):
    """LLM 对每条候选记忆的决策动作。"""
    ADD = "ADD"        # 新信息，直接插入
    UPDATE = "UPDATE"  # 与某条已有记忆冲突/补充，合并后覆盖
    DELETE = "DELETE"  # 已有记忆已过时或被对话明确否定
    NONE = "NONE"      # 已有记忆已涵盖，无需操作


class MemoryOperation(BaseModel):
    """LLM 返回的单条操作指令。"""
    action: MemoryAction
    content: Optional[str] = None     # ADD / UPDATE 时的最终文本
    memory_id: Optional[str] = None   # UPDATE / DELETE 时指向的已有记忆 id
    reason: Optional[str] = None      # DELETE 时说明原因（可选，用于日志）


class AddMemoryResult(BaseModel):
    """add_from_conversation 的返回结果，记录本次操作明细。"""
    added: List[MemoryItem] = Field(default_factory=list)
    updated: List[MemoryItem] = Field(default_factory=list)
    deleted: List[str] = Field(default_factory=list)  # 被删除的 memory_id 列表

    @property
    def total_changes(self) -> int:
        return len(self.added) + len(self.updated) + len(self.deleted)


class CustomerProfile(BaseModel):
    """客户基础画像结构"""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    location: Optional[str] = None
    occupation: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class MemoryItem(BaseModel):
    """单条记忆条目"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    memory_type: MemoryType
    content: str                            # 文本化内容，用于向量检索
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_channel: Optional[ChannelType] = None
    profile: Optional[CustomerProfile] = None
    # source_time: 这段对话/事件实际发生的时间（区别于 created_at 入库时间）
    source_time: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def to_store_payload(self) -> Dict[str, Any]:
        """序列化为向量库可存储的 payload（不含 embedding）"""
        data = self.model_dump()
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        if self.source_time:
            data["source_time"] = self.source_time.isoformat()
        if self.profile:
            data["profile"] = self.profile.model_dump()
        return data

    @classmethod
    def from_store_payload(cls, payload: Dict[str, Any]) -> "MemoryItem":
        """从向量库 payload 反序列化"""
        data = dict(payload)
        for field in ("created_at", "updated_at", "source_time"):
            if isinstance(data.get(field), str):
                data[field] = datetime.fromisoformat(data[field])
        if data.get("profile") and isinstance(data["profile"], dict):
            data["profile"] = CustomerProfile(**data["profile"])
        return cls(**data)


class MemoryQuery(BaseModel):
    """查询参数"""
    user_id: str
    query: str
    memory_types: Optional[List[MemoryType]] = None
    source_channels: Optional[List[ChannelType]] = None
    top_k: int = 5
    score_threshold: float = 0.0
    filters: Dict[str, Any] = Field(default_factory=dict)


class MemorySearchResult(BaseModel):
    """单条检索结果"""
    item: MemoryItem
    score: float
