"""
启动 FastAPI 记忆服务。

运行：
    OPENAI_API_KEY=sk-... python -m examples.api_server

访问文档：http://localhost:8000/docs
"""
from __future__ import annotations

import os
import sys

import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from memory import create_openai_qdrant_memory_manager, create_openai_memory_manager
from memory.api.routes import app


def build_manager():
    api_key = os.environ["OPENAI_API_KEY"]
    qdrant_url = os.environ.get("QDRANT_URL")

    if qdrant_url:
        return create_openai_qdrant_memory_manager(
            api_key=api_key,
            qdrant_url=qdrant_url,
            qdrant_collection=os.environ.get("QDRANT_COLLECTION", "memories"),
        )
    else:
        print("[WARNING] QDRANT_URL not set, using in-memory store (data will be lost on restart)")
        return create_openai_memory_manager(api_key=api_key)


@app.on_event("startup")
async def startup():
    app.state.memory = build_manager()
    print("MemoryManager initialized.")


if __name__ == "__main__":
    uvicorn.run("examples.api_server:app", host="0.0.0.0", port=8000, reload=False)
