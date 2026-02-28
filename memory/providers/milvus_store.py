"""
Milvus 向量存储实现，兼容 Milvus 2.4+ 和 Milvus 3.x。

依赖:
    pymilvus >= 2.5   （含 AsyncMilvusClient）

Collection Schema:
    id           VARCHAR(64)    主键（memory id / profile key）
    embedding    FLOAT_VECTOR   语义向量
    user_id      VARCHAR(256)   过滤字段
    memory_type  VARCHAR(64)    过滤字段
    payload_json JSON           完整 payload 序列化存储

使用注意:
    - uri 可以是 "http://host:port"（远程 Milvus）
      或本地文件路径 "path/to/local.db"（Milvus Lite，适合开发测试）
    - token 格式为 "user:password" 或云服务 API Key
    - Milvus 2.x 需启用 upsert 特性（2.3+ 默认支持）；
      Milvus 3.x 完整支持所有接口。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from pymilvus import AsyncMilvusClient, DataType, MilvusClient

from ..base import BaseVectorStore

logger = logging.getLogger(__name__)


class MilvusVectorStore(BaseVectorStore):
    """基于 AsyncMilvusClient 的向量存储（async-first，兼容 Milvus 2.4+ / 3.x）。"""

    # HNSW 索引参数（兼容 Milvus 2.x / 3.x，COSINE 指标）
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
            db_name:         数据库名称（Milvus 2.x/3.x 默认 "default"）
        """
        self._uri = uri
        self._token = token
        self._db_name = db_name
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._client: Optional[AsyncMilvusClient] = None
        self._collection_ready = False

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    async def _get_client(self) -> AsyncMilvusClient:
        """延迟初始化 AsyncMilvusClient。"""
        if self._client is None:
            kwargs: Dict[str, Any] = {"uri": self._uri}
            if self._token:
                kwargs["token"] = self._token
            if self._db_name != "default":
                kwargs["db_name"] = self._db_name
            self._client = AsyncMilvusClient(**kwargs)
            logger.debug("AsyncMilvusClient 已连接: %s", self._uri)
        return self._client

    async def _ensure_collection(self) -> None:
        """若 Collection 不存在则自动创建（含 Schema 和 HNSW 索引）。"""
        if self._collection_ready:
            return
        client = await self._get_client()
        exists = await client.has_collection(self.collection_name)
        if not exists:
            # ---- Schema ----
            # 使用同步 MilvusClient 的类方法创建 schema（与连接无关）
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

            # ---- Index ----
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type=self._INDEX_TYPE,
                metric_type=self._METRIC_TYPE,
                params=self._INDEX_PARAMS,
            )

            await client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
            )
            logger.info("已创建 Milvus collection: %s（dim=%d）", self.collection_name, self.vector_size)
        self._collection_ready = True

    @staticmethod
    def _to_row(
        id: str,
        embedding: List[float],
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """将 memory payload 转为 Milvus 行格式。"""
        return {
            "id": id,
            "embedding": embedding,
            "user_id": payload.get("user_id", ""),
            "memory_type": payload.get("memory_type", ""),
            "payload_json": payload,
        }

    @staticmethod
    def _build_filter(filters: Dict[str, Any]) -> str:
        """
        将 filters dict 转为 Milvus 标量过滤表达式。

        示例:
            {"user_id": "u1", "memory_type": "fact"}
            → 'user_id == "u1" && memory_type == "fact"'
        """
        if not filters:
            return ""
        parts: List[str] = []
        for key, value in filters.items():
            if isinstance(value, list):
                quoted = [
                    f'"{v}"' if isinstance(v, str) else str(v)
                    for v in value
                ]
                parts.append(f'{key} in [{", ".join(quoted)}]')
            elif isinstance(value, str):
                parts.append(f'{key} == "{value}"')
            else:
                parts.append(f"{key} == {value}")
        return " && ".join(parts)

    # ------------------------------------------------------------------
    # BaseVectorStore 接口实现
    # ------------------------------------------------------------------

    async def upsert(
        self,
        id: str,
        embedding: List[float],
        payload: Dict[str, Any],
    ) -> None:
        """插入或更新一条记录（id 已存在则覆盖）。"""
        await self._ensure_collection()
        client = await self._get_client()
        row = self._to_row(id, embedding, payload)
        await client.upsert(
            collection_name=self.collection_name,
            data=[row],
        )

    async def search(
        self,
        embedding: List[float],
        top_k: int,
        filters: Dict[str, Any],
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        语义向量检索。

        Returns:
            [(id, score, payload), ...]，score 为余弦相似度，降序排列
        """
        await self._ensure_collection()
        client = await self._get_client()
        expr = self._build_filter(filters)
        results = await client.search(
            collection_name=self.collection_name,
            data=[embedding],
            anns_field="embedding",
            limit=top_k,
            filter=expr or None,
            output_fields=["payload_json"],
        )
        # results[0] 是第一条查询向量对应的 hits 列表
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
        """根据 id 精确获取 payload，不存在则返回 None。"""
        await self._ensure_collection()
        client = await self._get_client()
        rows = await client.get(
            collection_name=self.collection_name,
            ids=[id],
            output_fields=["payload_json"],
        )
        if rows:
            return rows[0].get("payload_json")
        return None

    async def delete(self, id: str) -> None:
        """删除指定 id 的记录。"""
        await self._ensure_collection()
        client = await self._get_client()
        await client.delete(
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
        await self._ensure_collection()
        client = await self._get_client()
        expr = f'user_id == "{user_id}"'
        if memory_type:
            expr += f' && memory_type == "{memory_type}"'
        rows = await client.query(
            collection_name=self.collection_name,
            filter=expr,
            output_fields=["payload_json"],
            limit=limit,
        )
        return [r.get("payload_json") or {} for r in rows]

    async def close(self) -> None:
        """主动关闭连接，释放资源（可选调用）。"""
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._collection_ready = False
            logger.debug("AsyncMilvusClient 已关闭")
