# Memory Component

一个独立的记忆管理组件，支持 LangGraph 集成，LLM 与向量化均为可插拔的独立 API。

## 功能特性

- **对话记忆提取**（mem0 风格）：自动从对话中抽取关键信息并持久化
- **客户画像管理**：结构化存储客户基本信息，支持字段级增量合并
- **多渠道数据接入**：CRM、社交媒体、电话、邮件等外部渠道数据统一入库
- **语义检索**：基于向量相似度的记忆查询
- **LangGraph 集成**：开箱即用的记忆检索/保存节点
- **REST API**：完整的 HTTP 接口，支持插入、更新、查询

## 架构概览

```
memory/
├── models.py                  # 数据模型（MemoryItem / CustomerProfile / MemoryQuery）
├── base.py                    # 可插拔抽象接口（LLM / Embedding / VectorStore）
├── factory.py                 # 快速创建 MemoryManager 的工厂函数
├── core/
│   └── memory_manager.py      # 核心编排器
├── providers/
│   ├── openai_llm.py          # OpenAI LLM（支持自定义 base_url）
│   ├── openai_embedding.py    # OpenAI Embedding
│   ├── qdrant_store.py        # Qdrant 向量库（生产推荐）
│   └── inmemory_store.py      # 内存向量库（测试用）
├── integrations/
│   └── langgraph.py           # LangGraph 节点
└── api/
    └── routes.py              # FastAPI REST 接口
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基础用法（Python）

```python
import asyncio
from memory import create_openai_memory_manager, MemoryQuery

async def main():
    manager = create_openai_memory_manager(api_key="sk-...")

    # 从对话中提取记忆
    await manager.add_from_conversation(
        user_id="user_001",
        messages=[
            {"role": "user", "content": "我叫李明，住在上海，喜欢智能家居产品。"},
            {"role": "assistant", "content": "您好李明！"},
        ]
    )

    # 语义检索
    results = await manager.query(
        MemoryQuery(user_id="user_001", query="用户的兴趣爱好", top_k=5)
    )
    for r in results:
        print(f"[{r.score:.3f}] {r.item.content}")

asyncio.run(main())
```

### 启动 REST API 服务

```bash
OPENAI_API_KEY=sk-... python -m examples.api_server
# 文档地址：http://localhost:8000/docs
```

### LangGraph 集成

```python
from langgraph.graph import StateGraph, START, END
from memory import create_openai_memory_manager
from memory.integrations.langgraph import (
    MemoryState,
    create_memory_retrieve_node,
    create_memory_save_node,
    inject_memories_into_prompt,
)

manager = create_openai_memory_manager(api_key="sk-...")

graph = StateGraph(MemoryState)
graph.add_node("retrieve", create_memory_retrieve_node(manager))
graph.add_node("llm",      your_llm_node)
graph.add_node("save",     create_memory_save_node(manager))

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "llm")
graph.add_edge("llm", "save")
graph.add_edge("save", END)
```

## 三种记忆类型

| 类型 | 说明 |
|------|------|
| `CONVERSATIONAL` | 对话提取的记忆片段（偏好、事件、目标等） |
| `CUSTOMER_PROFILE` | 结构化客户画像（姓名/地区/标签/偏好等） |
| `EXTERNAL_CHANNEL` | 外部渠道数据（CRM、社交媒体、电话录音等） |

## REST API 接口

| 方法 | 路径 | 功能 |
|------|------|------|
| `POST` | `/memories` | 插入单条记忆 |
| `GET` | `/memories/{id}` | 获取单条记忆 |
| `PUT` | `/memories/{id}` | 更新记忆内容/元数据 |
| `DELETE` | `/memories/{id}` | 删除记忆 |
| `POST` | `/memories/query` | 语义检索 |
| `GET` | `/memories/user/{user_id}` | 列出用户全部记忆 |
| `POST` | `/memories/conversation` | 对话自动提取（mem0 风格） |
| `GET` | `/profile/{user_id}` | 获取客户画像 |
| `PUT` | `/profile/{user_id}` | 更新客户画像 |
| `POST` | `/external` | 写入外部渠道数据 |

## 替换 LLM / Embedding / 向量库

实现对应的抽象接口即可：

```python
from memory.base import BaseLLMClient, BaseEmbeddingClient, BaseVectorStore
from memory import create_memory_manager

class MyLLM(BaseLLMClient):
    async def extract_memories(self, messages, existing_memories): ...
    async def extract_profile(self, messages): ...
    async def merge_memory(self, existing, new_info): ...

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

## 生产部署（OpenAI + Qdrant）

```python
from memory import create_openai_qdrant_memory_manager

manager = create_openai_qdrant_memory_manager(
    api_key="sk-...",
    qdrant_url="http://localhost:6333",
    qdrant_collection="memories",
)
```

启动 Qdrant：
```bash
docker run -p 6333:6333 qdrant/qdrant
```

## 使用第三方兼容 API（DeepSeek / Qwen 等）

```python
manager = create_openai_memory_manager(
    api_key="your-key",
    base_url="https://api.deepseek.com/v1",  # 任意 OpenAI 兼容接口
    llm_model="deepseek-chat",
    embedding_model="text-embedding-3-small",
)
```
