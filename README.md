# Memory Component

一个独立的记忆管理组件，支持 LangGraph 集成，LLM / Embedding / 向量库均为可插拔的独立接口。

---

## 功能特性

- **智能记忆决策**（mem0 风格）：ADD / UPDATE / DELETE / NONE，避免重复和冗余
- **会话时间感知**：记录记忆的实际发生时间，支持时序上下文推理
- **客户画像管理**：结构化存储，字段级增量合并
- **多渠道数据接入**：CRM、社交媒体、电话、邮件等统一入库
- **语义检索**：基于向量相似度的记忆查询
- **LangGraph 集成**：阻塞/非阻塞两种保存节点，开箱即用
- **主动触达决策**：记忆变更事件总线 + LLM 评估是否需要联系客户
- **企业级认证**：HMAC 签名注入，对接内部 LLM 网关
- **多向量库**：Qdrant、Milvus（2.4+/3.x）、内存库（测试）
- **REST API**：完整 HTTP 接口

---

## 架构概览

```
memory/
├── models.py                   # 数据模型
├── base.py                     # 可插拔抽象接口（LLM / Embedding / VectorStore）
├── factory.py                  # 工厂函数（快速创建 MemoryManager）
├── core/
│   ├── memory_manager.py       # 核心编排器
│   └── event_bus.py            # 记忆变更事件总线 + 主动触达决策器
├── providers/
│   ├── openai_llm.py           # OpenAI LLM（支持自定义 http_client / sign）
│   ├── openai_embedding.py     # OpenAI Embedding
│   ├── qdrant_store.py         # Qdrant 向量库
│   ├── milvus_store.py         # Milvus 向量库（2.4+ / 3.x）
│   ├── inmemory_store.py       # 内存向量库（测试用）
│   └── signed_auth.py          # HMAC-SHA256 签名认证（httpx.Auth）
├── integrations/
│   └── langgraph.py            # LangGraph 节点（阻塞 + 非阻塞）
└── api/
    └── routes.py               # FastAPI REST 接口
```

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt

# 按需选择向量库
pip install qdrant-client>=1.9    # Qdrant
pip install pymilvus>=2.5         # Milvus 2.4+ / 3.x
```

### 基础用法

```python
import asyncio
from datetime import datetime
from memory import create_openai_memory_manager, MemoryQuery

async def main():
    manager = create_openai_memory_manager(api_key="sk-...")

    # 从对话中提取记忆（自动 ADD / UPDATE / DELETE 决策）
    result = await manager.add_from_conversation(
        user_id="user_001",
        messages=[
            {"role": "user",      "content": "我叫李明，住在上海，预算 5000 元。",
             "timestamp": "2026-02-28T10:00:00"},   # 可选：消息级时间戳
            {"role": "assistant", "content": "好的，我为您推荐..."},
        ],
        conversation_time=datetime(2026, 2, 28, 10, 0),  # 可选：会话级时间
    )
    print(f"新增: {len(result.added)}  更新: {len(result.updated)}  删除: {len(result.deleted)}")

    # 语义检索
    hits = await manager.query(
        MemoryQuery(user_id="user_001", query="用户预算", top_k=5)
    )
    for h in hits:
        print(f"[{h.score:.3f}] {h.item.content}")

asyncio.run(main())
```

---

## 记忆决策：ADD / UPDATE / DELETE

每次 `add_from_conversation` 调用时，LLM 会对比已有记忆，输出操作指令：

| 操作 | 含义 |
|------|------|
| `ADD` | 全新信息，直接插入 |
| `UPDATE` | 补充/修正已有记忆，合并后覆盖 |
| `DELETE` | 已有记忆被明确否定或完全过时 |
| `NONE` | 已有记忆已涵盖，跳过 |

```python
result = await manager.add_from_conversation(user_id="u1", messages=[...])
# result: AddMemoryResult
result.added    # List[MemoryItem] — 本次新增
result.updated  # List[MemoryItem] — 本次更新
result.deleted  # List[str]        — 被删除的 memory_id
result.total_changes  # int        — 总变更数
```

---

## 会话时间

记忆区分两个时间：

| 字段 | 含义 |
|------|------|
| `source_time` | 对话/事件实际发生时间（由调用方提供） |
| `created_at`  | 记忆入库时间（系统自动） |
| `updated_at`  | 记忆最后修改时间（系统自动） |

**时间推断优先级**：`conversation_time` 参数 → 消息中最晚的 `timestamp` 字段 → `datetime.utcnow()`

---

## LangGraph 集成

### 阻塞保存（强一致）

适合对记忆完整性要求高的场景，保存完成后才进入下一节点。

```python
from memory.integrations.langgraph import (
    MemoryState, create_memory_retrieve_node,
    create_memory_save_node, inject_memories_into_prompt,
)

graph = StateGraph(MemoryState)
graph.add_node("retrieve", create_memory_retrieve_node(manager, top_k=5))
graph.add_node("llm",      your_llm_node)
graph.add_node("save",     create_memory_save_node(manager))
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "llm")
graph.add_edge("llm", "save")
graph.add_edge("save", END)
```

### 非阻塞保存（低延迟）

`save` 节点立即返回，用户拿到回复后，记忆保存和事件发布在后台异步执行。

```python
from memory.integrations.langgraph import create_async_memory_save_node
from memory.core.event_bus import MemoryEventBus

bus = MemoryEventBus()

graph.add_node("save", create_async_memory_save_node(
    memory_manager=manager,
    event_bus=bus,                                      # 有变更时自动发布事件
    error_handler=lambda e: logging.error(e),
))

# state 中可注入会话时间
state = {
    "user_id": "u1",
    "messages": [...],
    "conversation_time": "2026-02-28T10:00:00",   # 可选
}
```

---

## 主动触达决策

当记忆发生变更时，自动评估是否需要主动联系客户（推销、提醒、跟进）。

```python
from memory.core.event_bus import MemoryEventBus, ProactiveOutreachEvaluator

bus = MemoryEventBus()
evaluator = ProactiveOutreachEvaluator(memory_manager=manager)

# 方式一：装饰器注册
@bus.subscribe
async def on_memory_changed(event):
    decision = await evaluator.evaluate(event)
    if decision.should_reach_out:
        await crm.send(event.user_id, decision.suggested_message)

# 方式二：as_callback 快捷注册
bus.subscribe(evaluator.as_callback(on_decision=your_async_handler))

# OutreachDecision 字段
decision.should_reach_out   # bool
decision.action_type        # "promotion" | "reminder" | "follow_up" | "none"
decision.urgency            # "low" | "medium" | "high"
decision.suggested_message  # 建议话术
decision.reason             # 决策理由
```

完整示例见 `examples/proactive_agent_example.py`。

---

## 企业内部 LLM（HMAC 签名）

对接需要签名认证的内部网关，签名逻辑完全隔离在 `HmacSignAuth` 中，LLM 代码零改动。

```python
from memory import create_signed_memory_manager
from memory.providers.milvus_store import MilvusVectorStore

manager = create_signed_memory_manager(
    app_id="your-app-id",
    secret_key="your-secret",
    base_url="https://internal-llm.company.com/v1",
    llm_model="qwen-max",
    embedding_model="text-embedding-v3",
    vector_store=MilvusVectorStore(uri="http://milvus:19530"),
)
```

自定义签名规则只需修改 `memory/providers/signed_auth.py` 中的 `auth_flow` 方法。

---

## 向量库配置

### Qdrant

```python
from memory import create_openai_qdrant_memory_manager

manager = create_openai_qdrant_memory_manager(
    api_key="sk-...",
    qdrant_url="http://localhost:6333",
    qdrant_collection="memories",
)
```

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Milvus（2.4+ / 3.x）

```python
from memory import create_openai_milvus_memory_manager

manager = create_openai_milvus_memory_manager(
    api_key="sk-...",
    milvus_uri="http://localhost:19530",
    milvus_token="root:Milvus",      # 可选
    milvus_collection="memories",
)
```

```bash
# Milvus 2.x
docker run -p 19530:19530 milvusdb/milvus:v2.4.0 milvus run standalone

# Milvus Lite（本地文件，适合开发）
milvus_uri="path/to/local.db"
```

---

## 替换任意组件

```python
from memory.base import BaseLLMClient, BaseEmbeddingClient, BaseVectorStore
from memory import create_memory_manager

class MyLLM(BaseLLMClient):
    async def extract_memories(self, messages, existing_memories): ...
    async def extract_profile(self, messages): ...
    async def merge_memory(self, existing, new_info): ...
    # 可选：实现 decide_memory_operations 获得 ADD/UPDATE/DELETE 能力
    # 可选：实现 evaluate_outreach 获得主动触达评估能力

class MyEmbedder(BaseEmbeddingClient):
    async def embed(self, text): ...
    async def embed_batch(self, texts): ...

class MyVectorStore(BaseVectorStore):
    async def upsert(self, id, embedding, payload): ...
    async def search(self, embedding, top_k, filters): ...
    async def get(self, id): ...
    async def delete(self, id): ...
    async def list_by_user(self, user_id, memory_type, limit): ...

manager = create_memory_manager(MyLLM(), MyEmbedder(), MyVectorStore())
```

---

## REST API

```bash
OPENAI_API_KEY=sk-... python -m examples.api_server
# 文档：http://localhost:8000/docs
```

| 方法 | 路径 | 功能 |
|------|------|------|
| `POST` | `/memories` | 插入单条记忆 |
| `GET` | `/memories/{id}` | 获取单条记忆 |
| `PUT` | `/memories/{id}` | 更新记忆 |
| `DELETE` | `/memories/{id}` | 删除记忆 |
| `POST` | `/memories/query` | 语义检索 |
| `GET` | `/memories/user/{user_id}` | 列出用户全部记忆 |
| `POST` | `/memories/conversation` | 对话自动提取 |
| `GET` | `/profile/{user_id}` | 获取客户画像 |
| `PUT` | `/profile/{user_id}` | 更新客户画像 |
| `POST` | `/external` | 写入外部渠道数据 |

---

## 三种记忆类型

| 类型 | 说明 |
|------|------|
| `CONVERSATIONAL` | 对话提取的记忆片段（偏好、事件、目标等） |
| `CUSTOMER_PROFILE` | 结构化客户画像（姓名/地区/标签/偏好等） |
| `EXTERNAL_CHANNEL` | 外部渠道数据（CRM、社交媒体、电话等） |

---

## 示例文件

| 文件 | 说明 |
|------|------|
| `examples/basic_usage.py` | 基础读写、查询示例 |
| `examples/langgraph_example.py` | LangGraph 对话机器人（阻塞保存） |
| `examples/proactive_agent_example.py` | 异步保存 + 主动触达完整演示 |
| `examples/api_server.py` | 启动 REST API 服务 |

---

## 更新记录

### v0.3（当前）
- **主动触达决策**：`MemoryEventBus` + `ProactiveOutreachEvaluator`，记忆变更后 LLM 自动评估是否需要联系客户
- **非阻塞 LangGraph 节点**：`create_async_memory_save_node`，fire-and-forget，不阻塞对话响应
- **会话时间感知**：`MemoryItem.source_time`，支持消息级/会话级时间戳注入
- **企业签名认证**：`HmacSignAuth`（httpx.Auth），可直接注入 `AsyncOpenAI`

### v0.2
- **智能记忆决策**：ADD / UPDATE / DELETE / NONE，替代简单的全量插入
- **Milvus 向量库**：`MilvusVectorStore`，兼容 Milvus 2.4+ 和 3.x（AsyncMilvusClient）
- **自定义 HTTP 客户端**：LLM / Embedding 支持注入 `httpx.AsyncClient`

### v0.1
- 基础记忆管理（插入、查询、更新、删除）
- Qdrant 向量库、OpenAI LLM / Embedding
- LangGraph 集成、FastAPI REST API、客户画像管理
