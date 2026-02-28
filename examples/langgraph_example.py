"""
LangGraph 集成示例：带记忆的客服对话机器人。

依赖：pip install langgraph langchain-openai

运行：  python -m examples.langgraph_example
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


async def main() -> None:
    try:
        from langgraph.graph import StateGraph, START, END
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        print("请先安装: pip install langgraph langchain-openai")
        return

    from memory import create_openai_memory_manager, MemoryQuery
    from memory.integrations.langgraph import (
        MemoryState,
        create_memory_retrieve_node,
        create_memory_save_node,
        inject_memories_into_prompt,
    )

    api_key = os.environ.get("OPENAI_API_KEY", "sk-YOUR-KEY")
    manager = create_openai_memory_manager(api_key=api_key)

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

    # ── 节点定义 ──────────────────────────────────────────────

    retrieve_node = create_memory_retrieve_node(manager, top_k=5, include_profile=True)
    save_node = create_memory_save_node(manager, extract_profile=True)

    async def llm_node(state: MemoryState) -> dict:
        base_system = "你是一个专业的智能家居客服助手，请根据用户历史记忆提供个性化服务。"
        memory_context = state.get("memory_context", "")
        system_prompt = inject_memories_into_prompt(base_system, memory_context)

        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        response = await llm.ainvoke(messages)
        return {"messages": state["messages"] + [response]}

    # ── 构建图 ─────────────────────────────────────────────────

    graph = StateGraph(MemoryState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("llm", llm_node)
    graph.add_node("save", save_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "llm")
    graph.add_edge("llm", "save")
    graph.add_edge("save", END)

    compiled = graph.compile()

    # ── 模拟多轮对话 ───────────────────────────────────────────

    user_id = "customer_001"

    turns = [
        "你好，我叫张伟，我对智能摄像头很感兴趣，预算 800 元。",
        "我之前说的预算能买到哪些型号？",     # 测试跨轮记忆
        "我家是在北京，有没有适合北方气候的型号？",
    ]

    for turn_text in turns:
        print(f"\n用户: {turn_text}")
        state = {
            "user_id": user_id,
            "messages": [HumanMessage(content=turn_text)],
        }
        result = await compiled.ainvoke(state)
        last_ai = result["messages"][-1]
        print(f"助手: {last_ai.content[:200]}")

    # 检查保存的记忆
    print("\n── 最终记忆库 ──")
    memories = await manager.list_all(user_id)
    for m in memories:
        print(f"  [{m.memory_type.value}] {m.content[:80]}")


if __name__ == "__main__":
    asyncio.run(main())
