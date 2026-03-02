"""
Milvus 向量存储实现，兼容 Milvus server 2.3+。

依赖:
    pymilvus >= 2.4   （含 MilvusClient 高级 API）

实现方式:
    使用同步 MilvusClient，通过 asyncio.to_thread() 桥接至 async 接口。
    所有阻塞 I/O 在线程池中执行，不阻塞事件循环。
    threading.Lock 防止并发初始化竞态。

Collection Schema:
    id           VARCHAR(64)    主键（memory id / profile key）
    embedding    FLOAT_VECTOR   语义向量
    user_id      VARCHAR(256)   过滤字段
    memory_type  VARCHAR(64)    过滤字段
    payload_json JSON           完整 payload 序列化存储

兼容性:
    - Milvus server 2.3+（upsert 在 2.3.0 正式支持）
    - Milvus Lite（本地 .db 文件，适合开发测试）
    - pymilvus >= 2.4（MilvusClient 高级 API，向下兼容 Milvus 2.3 server）
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from pymilvus import DataType, MilvusClient

from ..base import BaseVectorStore

logger = logging.getLogger(__name__)


class MilvusVectorStore(BaseVectorStore):
    """
    基于同步 MilvusClient + asyncio.to_thread 的向量存储。

    兼容 Milvus server 2.3+ / pymilvus 2.4+。
    """

    _INDEX_TYPE = "HNSW"
    _METRIC_TYPE = "COSINE"
    _INDEX_PARAMS = {"M": 16, "efConstruction": 200}

    def __init__(
        self,
        uri: str = "http://localhost:19530",
        collection_name: str = "memories",
        vector_size: int = 1536,
        token: Optional[str] = None,
        db_name: str = "default",
    ) -> None:
        """
        Args:
            uri:             Milvus 服务地址，如 "http://localhost:19530"
                             或 Milvus Lite 本地路径 "path/to/milvus.db"
            collection_name: Collection 名称
            vector_size:     向量维度，需与 Embedding 模型一致
            token:           认证凭据，格式 "user:password" 或云端 API Key
            db_name:         数据库名称（默认 "default"）
        """
        self._uri = uri
        self._token = token
        self._db_name = db_name
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._client: Optional[MilvusClient] = None
        self._collection_ready = False
        self._lock = threading.Lock()   # 保护延迟初始化，防止并发竞态

    # ------------------------------------------------------------------
    # 同步内部方法（在 to_thread 线程池中调用）
    # ------------------------------------------------------------------

    def _get_client(self) -> MilvusClient:
        """获取（延迟初始化的）MilvusClient，线程安全。"""
        with self._lock:
            if self._client is None:
                kwargs: Dict[str, Any] = {"uri": self._uri}
                if self._token:
                    kwargs["token"] = self._token
                if self._db_name != "default":
                    kwargs["db_name"] = self._db_name
                self._client = MilvusClient(**kwargs)
                logger.debug("MilvusClient 已连接: %s", self._uri)
        return self._client

    def _ensure_collection_sync(self) -> None:
        """若 Collection 不存在则同步创建（含 Schema 和 HNSW 索引）。"""
        with self._lock:
            if self._collection_ready:
                return
            client = self._get_client()
            if not client.has_collection(self.collection_name):
                schema = MilvusClient.create_schema(
                    auto_id=False,
                    enable_dynamic_field=False,
                )
                schema.add_field(
                    field_name="id",
                    datatype=DataType.VARCHAR,
                    max_length=64,
                    is_primary=True,
                )
                schema.add_field(
                    field_name="embedding",
                    datatype=DataType.FLOAT_VECTOR,
                    dim=self.vector_size,
                )
                schema.add_field(
                    field_name="user_id",
                    datatype=DataType.VARCHAR,
                    max_length=256,
                )
                schema.add_field(
                    field_name="memory_type",
                    datatype=DataType.VARCHAR,
                    max_length=64,
                )
                schema.add_field(
                    field_name="payload_json",
                    datatype=DataType.JSON,
                )

                index_params = MilvusClient.prepare_index_params()
                index_params.add_index(
                    field_name="embedding",
                    index_type=self._INDEX_TYPE,
                    metric_type=self._METRIC_TYPE,
                    params=self._INDEX_PARAMS,
                )

                client.create_collection(
                    collection_name=self.collection_name,
                    schema=schema,
                    index_params=index_params,
                )
                logger.info(
                    "已创建 Milvus collection: %s (dim=%d)",
                    self.collection_name,
                    self.vector_size,
                )
            self._collection_ready = True

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _to_row(
        id: str,
        embedding: List[float],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "id": id,
            "embedding": embedding,
            "user_id": payload.get("user_id", ""),
            "memory_type": payload.get("memory_type", ""),
            "payload_json": payload,
        }

    @staticmethod
    def _build_filter(filters: Dict[str, Any]) -> str:
        """将 filters dict 转为 Milvus 标量过滤表达式。"""
        if not filters:
            return ""
        parts: List[str] = []
        for key, value in filters.items():
            if isinstance(value, list):
                quoted = [
                    f'"{v}"' if isinstance(v, str) else str(v) for v in value
                ]
                parts.append(f'{key} in [{", ".join(quoted)}]')
            elif isinstance(value, str):
                parts.append(f'{key} == "{value}"')
            else:
                parts.append(f"{key} == {value}")
        return " && ".join(parts)

    # ------------------------------------------------------------------
    # BaseVectorStore 接口实现（均通过 asyncio.to_thread 非阻塞执行）
    # ------------------------------------------------------------------

    async def upsert(
        self,
        id: str,
        embedding: List[float],
        payload: Dict[str, Any],
    ) -> None:
        """插入或更新一条记录（id 已存在则覆盖）。"""
        await asyncio.to_thread(self._ensure_collection_sync)
        row = self._to_row(id, embedding, payload)
        client = self._get_client()
        await asyncio.to_thread(
            client.upsert,
            collection_name=self.collection_name,
            data=[row],
        )

    async def search(
        self,
        embedding: List[float],
        top_k: int,
        filters: Dict[str, Any],
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """语义向量检索，返回 [(id, score, payload), ...]，降序。"""
        await asyncio.to_thread(self._ensure_collection_sync)
        client = self._get_client()
        expr = self._build_filter(filters)
        results = await asyncio.to_thread(
            client.search,
            collection_name=self.collection_name,
            data=[embedding],
            anns_field="embedding",
            limit=top_k,
            filter=expr or None,
            output_fields=["payload_json"],
        )
        hits = results[0] if results else []
        return [
            (
                str(hit["id"]),
                float(hit["distance"]),
                hit["entity"].get("payload_json") or {},
            )
            for hit in hits
        ]

    async def get(self, id: str) -> Optional[Dict[str, Any]]:
        """根据 id 精确获取 payload，不存在返回 None。"""
        await asyncio.to_thread(self._ensure_collection_sync)
        client = self._get_client()
        rows = await asyncio.to_thread(
            client.get,
            collection_name=self.collection_name,
            ids=[id],
            output_fields=["payload_json"],
        )
        if rows:
            return rows[0].get("payload_json")
        return None

    async def delete(self, id: str) -> None:
        """删除指定 id 的记录。"""
        await asyncio.to_thread(self._ensure_collection_sync)
        client = self._get_client()
        await asyncio.to_thread(
            client.delete,
            collection_name=self.collection_name,
            ids=[id],
        )

    async def list_by_user(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """列出某用户的所有记忆，可按 memory_type 过滤。"""
        await asyncio.to_thread(self._ensure_collection_sync)
        client = self._get_client()
        expr = f'user_id == "{user_id}"'
        if memory_type:
            expr += f' && memory_type == "{memory_type}"'
        rows = await asyncio.to_thread(
            client.query,
            collection_name=self.collection_name,
            filter=expr,
            output_fields=["payload_json"],
            limit=limit,
        )
        return [r.get("payload_json") or {} for r in rows]

    def close(self) -> None:
        """关闭连接，释放资源（同步调用）。"""
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None
                self._collection_ready = False
                logger.debug("MilvusClient 已关闭")
