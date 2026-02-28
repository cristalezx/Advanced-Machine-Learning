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


def create_openai_milvus_memory_manager(
    api_key: str,
    milvus_uri: str = "http://localhost:19530",
    milvus_collection: str = "memories",
    vector_size: int = 1536,
    llm_model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
    base_url: Optional[str] = None,
    milvus_token: Optional[str] = None,
    milvus_db_name: str = "default",
) -> MemoryManager:
    """
    生产级工厂：OpenAI + Milvus（兼容 Milvus 2.4+ / 3.x）。

    Args:
        api_key:          OpenAI API Key
        milvus_uri:       Milvus 服务地址，如 "http://localhost:19530"
                          或 Milvus Lite 本地路径 "path/to/milvus.db"
        milvus_collection: Collection 名称
        vector_size:      向量维度（需与 embedding_model 一致，
                          text-embedding-3-small 默认 1536）
        llm_model:        用于提取记忆的 LLM 模型名
        embedding_model:  用于向量化的 Embedding 模型名
        base_url:         可选，指向 OpenAI 兼容第三方服务
        milvus_token:     Milvus 认证凭据（"user:password" 或 API Key）
        milvus_db_name:   Milvus 数据库名称
    """
    from .providers.openai_llm import OpenAILLMClient
    from .providers.openai_embedding import OpenAIEmbeddingClient
    from .providers.milvus_store import MilvusVectorStore

    llm = OpenAILLMClient(api_key=api_key, model=llm_model, base_url=base_url)
    embedder = OpenAIEmbeddingClient(api_key=api_key, model=embedding_model, base_url=base_url)
    store = MilvusVectorStore(
        uri=milvus_uri,
        collection_name=milvus_collection,
        vector_size=vector_size,
        token=milvus_token,
        db_name=milvus_db_name,
    )

    return MemoryManager(llm_client=llm, embedding_client=embedder, vector_store=store)


def create_signed_memory_manager(
    app_id: str,
    secret_key: str,
    base_url: str,
    llm_model: str,
    embedding_model: str,
    vector_store: Optional[BaseVectorStore] = None,
    api_key: str = "placeholder",
) -> MemoryManager:
    """
    带 HMAC 签名认证的工厂函数，适合内部/企业接口。

    签名由 HmacSignAuth 自动注入每个请求，调用方无需关心签名细节。

    Args:
        app_id:          服务分配的应用 ID
        secret_key:      签名密钥
        base_url:        内部 LLM 服务地址（OpenAI 兼容）
        llm_model:       LLM 模型名
        embedding_model: Embedding 模型名
        vector_store:    向量库实例，默认使用内存库（仅测试用）
        api_key:         部分网关仍需传固定占位 key，默认 "placeholder"
    """
    import httpx
    from .providers.openai_llm import OpenAILLMClient
    from .providers.openai_embedding import OpenAIEmbeddingClient
    from .providers.inmemory_store import InMemoryVectorStore
    from .providers.signed_auth import HmacSignAuth

    auth = HmacSignAuth(app_id=app_id, secret_key=secret_key)
    # 同一个 httpx.AsyncClient 可被 LLM 和 Embedding 共享
    http_client = httpx.AsyncClient(auth=auth)

    llm = OpenAILLMClient(
        api_key=api_key,
        model=llm_model,
        base_url=base_url,
        http_client=http_client,
    )
    embedder = OpenAIEmbeddingClient(
        api_key=api_key,
        model=embedding_model,
        base_url=base_url,
        http_client=http_client,
    )
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
