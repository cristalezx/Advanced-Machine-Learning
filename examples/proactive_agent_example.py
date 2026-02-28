"""
示例：异步记忆存储 + 主动触达监听器

展示的功能
----------
1. LangGraph agent 使用 create_async_memory_save_node：
   记忆保存在后台进行，不阻塞对话响应。

2. MemoryEventBus + ProactiveOutreachEvaluator：
   每次记忆有变更时，LLM 评估是否需要主动联系客户，
   触发后调用 on_outreach_triggered（可对接 CRM / 消息推送系统）。

3. 会话时间注入：state 中传入 conversation_time，
   让记忆系统感知对话发生的实际时间。

运行方式
--------
    export OPENAI_API_KEY=sk-...
    python -m examples.proactive_agent_example
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ------------------------------------------------------------------
# 模拟的下游动作（实际场景替换为 CRM API / 消息推送 / 工单系统等）
# ------------------------------------------------------------------

async def on_outreach_triggered(user_id: str, decision) -> None:
    """收到主动触达决策后的处理函数（此处仅打印，实际可调用外部 API）。"""
    print(f"\n{'='*60}")
    print(f"[主动触达] 用户: {user_id}")
    print(f"  动作类型: {decision.action_type}  紧急程度: {decision.urgency}")
    print(f"  决策原因: {decision.reason}")
    print(f"  建议话术: {decision.suggested_message}")
    print(f"{'='*60}\n")


# ------------------------------------------------------------------
# 主程序
# ------------------------------------------------------------------

async def main() -> None:
    try:
        from langgraph.graph import StateGraph, START, END
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        print("请先安装: pip install langgraph langchain-openai")
        return

    from memory import create_openai_memory_manager
    from memory.integrations.langgraph import (
        MemoryState,
        create_memory_retrieve_node,
        create_async_memory_save_node,
        inject_memories_into_prompt,
    )
    from memory.core.event_bus import MemoryEventBus, ProactiveOutreachEvaluator

    api_key = os.environ.get("OPENAI_API_KEY", "sk-YOUR-KEY")
    manager = create_openai_memory_manager(api_key=api_key)
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

    # ── 事件总线 & 主动触达评估器 ──────────────────────────────────
    bus = MemoryEventBus()
    evaluator = ProactiveOutreachEvaluator(
        memory_manager=manager,
        min_changes_to_evaluate=1,
        # 可传入 outreach_system_prompt 定制业务规则
    )

    # 注册评估回调：记忆有变更 → 评估 → 触发下游动作
    bus.subscribe(
        evaluator.as_callback(on_decision=on_outreach_triggered)
    )

    # ── LangGraph 节点 ────────────────────────────────────────────
    retrieve_node = create_memory_retrieve_node(manager, top_k=5, include_profile=True)

    # 异步保存节点：立即返回，后台保存 + 发布事件
    async_save_node = create_async_memory_save_node(
        memory_manager=manager,
        extract_profile=True,
        event_bus=bus,
        error_handler=lambda e: print(f"[ERROR] 记忆保存失败: {e}"),
    )

    async def llm_node(state: MemoryState) -> dict:
        base_system = (
            "你是一个专业的智能家居销售顾问，请根据用户历史记忆提供个性化服务。"
        )
        system_prompt = inject_memories_into_prompt(
            base_system, state.get("memory_context", "")
        )
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        response = await llm.ainvoke(messages)
        return {"messages": state["messages"] + [response]}

    # ── 构建图 ───────────────────────────────────────────────────
    graph = StateGraph(MemoryState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("llm", llm_node)
    graph.add_node("save", async_save_node)   # 非阻塞节点

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "llm")
    graph.add_edge("llm", "save")
    graph.add_edge("save", END)

    compiled = graph.compile()

    # ── 模拟多轮对话（含会话时间） ──────────────────────────────────
    user_id = "customer_001"
    base_time = datetime(2026, 2, 28, 10, 0, 0)

    turns = [
        # (消息内容, 会话时间偏移分钟)
        ("你好，我叫李梅，我想给新家配一套全屋智能家居，预算 3 万。",       0),
        ("我家 120 平米，在上海浦东新区，打算 4 月底入住。",               15),
        ("我之前用过小米生态的产品，体验不错，希望新家继续用这个品牌。",   30),
        ("哦对了，我的预算可以提高到 5 万，想要更好的方案。",             45),
    ]

    print(f"开始对话 | 用户: {user_id}\n{'─'*60}")

    for text, offset_min in turns:
        conv_time = base_time + timedelta(minutes=offset_min)
        print(f"\n[{conv_time.strftime('%H:%M')}] 用户: {text}")

        state = {
            "user_id": user_id,
            "messages": [HumanMessage(content=text)],
            "conversation_time": conv_time.isoformat(),  # 注入会话时间
        }
        result = await compiled.ainvoke(state)
        last_ai = result["messages"][-1]
        print(f"助手: {last_ai.content[:200]}")

        # 给后台任务一点时间执行（实际生产中不需要，这里只为演示输出顺序）
        await asyncio.sleep(0.5)

    # ── 等待后台任务完成并展示最终记忆 ────────────────────────────
    print(f"\n{'─'*60}")
    print("等待后台记忆任务完成...")
    await asyncio.sleep(3)

    print(f"\n{'─'*60}")
    print("最终记忆库：")
    memories = await manager.list_all(user_id)
    for m in memories:
        ts = m.source_time.strftime("%H:%M") if m.source_time else "N/A"
        print(f"  [{m.memory_type.value}] [{ts}] {m.content[:80]}")


if __name__ == "__main__":
    asyncio.run(main())
