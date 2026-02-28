"""
工厂函数 — 快速创建 MemoryManager 实例。
"""
from __future__ import annotations

from typing import Optional

from .base import BaseLLMClient, BaseEmbeddingClient, BaseVectorStore
from .core.memory_manager import MemoryManager


def create_memory_manager(
    llm_client: BaseLLMClient,
    embedding_client: BaseEmbeddingClient,
    vector_store: BaseVectorStore,
) -> MemoryManager:
    """通用工厂：传入任意实现了接口的三个组件即可。"""
    return MemoryManager(
        llm_client=llm_client,
        embedding_client=embedding_client,
        vector_store=vector_store,
    )


def create_openai_memory_manager(
    api_key: str,
    llm_model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
    base_url: Optional[str] = None,
    vector_store: Optional[BaseVectorStore] = None,
) -> MemoryManager:
    """
    快速创建基于 OpenAI（或兼容 API）的 MemoryManager。

    Args:
        api_key: OpenAI API Key
        llm_model: 用于提取记忆的 LLM 模型名
        embedding_model: 用于向量化的 Embedding 模型名
        base_url: 可选，指向 OpenAI 兼容的第三方服务
        vector_store: 可选，默认使用内存向量库（仅适合测试）
    """
    from .providers.openai_llm import OpenAILLMClient
    from .providers.openai_embedding import OpenAIEmbeddingClient
    from .providers.inmemory_store import InMemoryVectorStore

    llm = OpenAILLMClient(api_key=api_key, model=llm_model, base_url=base_url)
    embedder = OpenAIEmbeddingClient(api_key=api_key, model=embedding_model, base_url=base_url)
    store = vector_store or InMemoryVectorStore()

    return MemoryManager(llm_client=llm, embedding_client=embedder, vector_store=store)


def create_openai_qdrant_memory_manager(
    api_key: str,
    qdrant_url: str = "http://localhost:6333",
    qdrant_collection: str = "memories",
    vector_size: int = 1536,
    llm_model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
    base_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
) -> MemoryManager:
    """
    生产级工厂：OpenAI + Qdrant 组合。
    """
    from .providers.openai_llm import OpenAILLMClient
    from .providers.openai_embedding import OpenAIEmbeddingClient
    from .providers.qdrant_store import QdrantVectorStore

    llm = OpenAILLMClient(api_key=api_key, model=llm_model, base_url=base_url)
    embedder = OpenAIEmbeddingClient(api_key=api_key, model=embedding_model, base_url=base_url)
    store = QdrantVectorStore(
        url=qdrant_url,
        collection_name=qdrant_collection,
        vector_size=vector_size,
        api_key=qdrant_api_key,
    )

    return MemoryManager(llm_client=llm, embedding_client=embedder, vector_store=store)
