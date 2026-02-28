"""
memory — 独立记忆组件

快速上手：
    from memory import create_openai_memory_manager, MemoryQuery

    manager = create_openai_memory_manager(api_key="sk-...")
    await manager.add_from_conversation(user_id="u1", messages=[...])
    results = await manager.query(MemoryQuery(user_id="u1", query="用户喜欢什么颜色"))
"""
from .factory import (
    create_memory_manager,
    create_openai_memory_manager,
    create_openai_milvus_memory_manager,
    create_openai_qdrant_memory_manager,
)
from .core.memory_manager import MemoryManager
from .models import (
    AddMemoryResult,
    ChannelType,
    CustomerProfile,
    MemoryAction,
    MemoryItem,
    MemoryOperation,
    MemoryQuery,
    MemorySearchResult,
    MemoryType,
)
from .base import BaseLLMClient, BaseEmbeddingClient, BaseVectorStore

__all__ = [
    # Factories
    "create_memory_manager",
    "create_openai_memory_manager",
    "create_openai_milvus_memory_manager",
    "create_openai_qdrant_memory_manager",
    # Core
    "MemoryManager",
    # Models
    "MemoryItem",
    "MemoryType",
    "MemoryAction",
    "MemoryOperation",
    "MemoryQuery",
    "MemorySearchResult",
    "AddMemoryResult",
    "CustomerProfile",
    "ChannelType",
    # Interfaces
    "BaseLLMClient",
    "BaseEmbeddingClient",
    "BaseVectorStore",
]
