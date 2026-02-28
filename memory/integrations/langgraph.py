"""
LangGraph 集成 — 提供开箱即用的记忆检索和保存节点。

用法示例：
    from memory.integrations.langgraph import (
        MemoryState,
        create_memory_retrieve_node,
        create_memory_save_node,
        inject_memories_into_prompt,
    )

    retrieve_node = create_memory_retrieve_node(memory_manager)
    save_node = create_memory_save_node(memory_manager)

    graph = StateGraph(MemoryState)
    graph.add_node("retrieve_memories", retrieve_node)
    graph.add_node("save_memories", save_node)
    ...
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..core.memory_manager import MemoryManager
from ..models import ChannelType, MemoryQuery, MemoryType

logger = logging.getLogger(__name__)

# LangGraph / LangChain 是可选依赖，按需导入
try:
    from langgraph.graph import MessagesState
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False
    MessagesState = dict  # fallback type hint only

from typing import TypedDict, Annotated
import operator


# ------------------------------------------------------------------
# State 定义（不强依赖 LangGraph，也可独立使用）
# ------------------------------------------------------------------

class MemoryState(TypedDict, total=False):
    """
    扩展了标准 MessagesState 的状态定义，
    新增 user_id 与检索到的记忆列表字段。
    """
    messages: List[Any]
    user_id: str
    retrieved_memories: List[str]   # 检索到的记忆文本列表
    memory_context: str             # 格式化后可直接注入 system prompt 的记忆上下文
    conversation_time: Optional[str]  # ISO 时间字符串，可由外部注入


# ------------------------------------------------------------------
# 节点工厂函数
# ------------------------------------------------------------------

def create_memory_retrieve_node(
    memory_manager: MemoryManager,
    top_k: int = 5,
    memory_types: Optional[List[MemoryType]] = None,
    include_profile: bool = True,
) -> Callable[[Dict], Dict]:
    """
    创建「记忆检索」节点。

    在 LLM 回复之前调用，从向量库中找回相关记忆并写入 state。
    """

    async def retrieve_memories(state: Dict) -> Dict:
        messages = state.get("messages", [])
        user_id = state.get("user_id", "anonymous")

        # 从最近的用户消息中提取查询文本
        query_text = _last_human_content(messages)
        if not query_text:
            return {"retrieved_memories": [], "memory_context": ""}

        results = await memory_manager.query(
            MemoryQuery(
                user_id=user_id,
                query=query_text,
                top_k=top_k,
                memory_types=memory_types,
            )
        )

        memory_texts = [r.item.content for r in results]

        # 额外附加客户画像（若需要）
        if include_profile:
            profile = await memory_manager.get_customer_profile(user_id)
            if profile:
                from ..core.memory_manager import _profile_to_text
                memory_texts.insert(0, f"[客户画像] {_profile_to_text(profile)}")

        context = _format_memory_context(memory_texts)
        return {
            "retrieved_memories": memory_texts,
            "memory_context": context,
        }

    return retrieve_memories


def create_memory_save_node(
    memory_manager: MemoryManager,
    extract_profile: bool = True,
) -> Callable[[Dict], Dict]:
    """
    创建「记忆保存」节点（阻塞版）。

    在对话结束后调用，将本轮对话提取并保存到向量库。
    适合对记忆保存有强一致性要求的场景（保存完成才继续下一节点）。
    """

    async def save_memories(state: Dict) -> Dict:
        messages = state.get("messages", [])
        user_id = state.get("user_id", "anonymous")

        msg_dicts = _messages_to_dicts(messages)
        if not msg_dicts:
            return {}

        await memory_manager.add_from_conversation(
            user_id=user_id,
            messages=msg_dicts,
            extract_profile=extract_profile,
        )
        return {}

    return save_memories


def create_async_memory_save_node(
    memory_manager: MemoryManager,
    extract_profile: bool = True,
    source_channel: ChannelType = ChannelType.CHAT,
    event_bus: Optional[Any] = None,
    error_handler: Optional[Callable[[Exception], None]] = None,
) -> Callable[[Dict], Awaitable[Dict]]:
    """
    创建「异步记忆保存」节点（非阻塞，fire-and-forget）。

    节点立即返回，记忆提取和保存在后台 asyncio.Task 中执行，
    不阻塞 LangGraph 主流程，适合对响应延迟敏感的场景。

    Args:
        memory_manager:  MemoryManager 实例
        extract_profile: 是否同步更新客户画像
        source_channel:  来源渠道
        event_bus:       MemoryEventBus 实例，有变更时自动 publish 事件
                         （ProactiveOutreachEvaluator 可订阅此总线）
        error_handler:   后台任务出错时的回调 handler(exc)，
                         默认仅打印 ERROR 日志

    Returns:
        LangGraph 节点函数，签名 async (state) -> {}
    """
    async def save_memories_async(state: Dict) -> Dict:
        messages = state.get("messages", [])
        user_id = state.get("user_id", "anonymous")
        conv_time_str = state.get("conversation_time")

        msg_dicts = _messages_to_dicts(messages)
        if not msg_dicts:
            return {}

        # 从 state 中解析会话时间（由调用方注入，或在后台自动推断）
        conv_time: Optional[datetime] = None
        if conv_time_str:
            try:
                conv_time = datetime.fromisoformat(conv_time_str)
            except ValueError:
                pass

        async def _do_save() -> None:
            try:
                result = await memory_manager.add_from_conversation(
                    user_id=user_id,
                    messages=msg_dicts,
                    conversation_time=conv_time,
                    source_channel=source_channel,
                    extract_profile=extract_profile,
                )
                # 有变更且配置了事件总线，则发布事件
                if event_bus is not None and result.total_changes > 0:
                    from ..core.event_bus import MemoryChangeEvent
                    event = MemoryChangeEvent(
                        user_id=user_id,
                        result=result,
                        channel=source_channel,
                        conversation_time=conv_time or datetime.utcnow(),
                    )
                    await event_bus.publish(event)
                    logger.debug(
                        "Published MemoryChangeEvent for user=%s (changes=%d)",
                        user_id,
                        result.total_changes,
                    )
            except Exception as exc:
                if error_handler:
                    error_handler(exc)
                else:
                    logger.error(
                        "Async memory save failed for user=%s: %s", user_id, exc
                    )

        # 调度后台任务，立即返回不等待
        asyncio.create_task(_do_save())
        return {}

    return save_memories_async


# ------------------------------------------------------------------
# 辅助函数：在构建 prompt 时注入记忆上下文
# ------------------------------------------------------------------

def inject_memories_into_prompt(
    system_prompt: str,
    memory_context: str,
) -> str:
    """
    将记忆上下文插入到 system prompt 的末尾。
    可在 LLM 节点中调用。
    """
    if not memory_context:
        return system_prompt
    return f"{system_prompt}\n\n{memory_context}"


def build_memory_system_message(memory_context: str) -> Any:
    """
    创建包含记忆信息的 SystemMessage（需要 LangChain）。
    """
    if not _LANGGRAPH_AVAILABLE:
        raise ImportError("langchain_core is required for this function")
    return SystemMessage(content=memory_context)


# ------------------------------------------------------------------
# 内部工具
# ------------------------------------------------------------------

def _last_human_content(messages: List[Any]) -> str:
    """从消息列表中提取最后一条用户消息的文本。"""
    for msg in reversed(messages):
        if _LANGGRAPH_AVAILABLE and isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else ""
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _messages_to_dicts(messages: List[Any]) -> List[Dict[str, str]]:
    """将 LangChain Message 对象或 dict 统一转为 dict 格式。"""
    result = []
    for msg in messages:
        if _LANGGRAPH_AVAILABLE:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": str(msg.content)})
            elif isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": str(msg.content)})
        elif isinstance(msg, dict):
            result.append(msg)
    return result


def _format_memory_context(memories: List[str]) -> str:
    if not memories:
        return ""
    lines = ["以下是关于该用户的已知信息，请在回复时参考："]
    for i, m in enumerate(memories, 1):
        lines.append(f"{i}. {m}")
    return "\n".join(lines)
