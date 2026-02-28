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
    content: str                          # 文本化内容，用于向量检索
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_channel: Optional[ChannelType] = None
    profile: Optional[CustomerProfile] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def to_store_payload(self) -> Dict[str, Any]:
        """序列化为向量库可存储的 payload（不含 embedding）"""
        data = self.model_dump()
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        if self.profile:
            data["profile"] = self.profile.model_dump()
        return data

    @classmethod
    def from_store_payload(cls, payload: Dict[str, Any]) -> "MemoryItem":
        """从向量库 payload 反序列化"""
        data = dict(payload)
        for field in ("created_at", "updated_at"):
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
