"""
记忆变更事件总线 + 主动触达决策器。

设计思路
--------
1. MemoryEventBus  —— 发布/订阅模式，记忆更新后异步触发所有已注册回调。
2. ProactiveOutreachEvaluator —— LLM 评估本次记忆变更是否需要主动联系客户，
   返回 OutreachDecision，供回调函数驱动 CRM、消息推送等下游动作。

使用流程
--------
    bus = MemoryEventBus()
    evaluator = ProactiveOutreachEvaluator(llm_client=manager.llm, memory_manager=manager)

    # 注册评估回调
    @bus.subscribe
    async def on_change(event: MemoryChangeEvent):
        decision = await evaluator.evaluate(event)
        if decision.should_reach_out:
            await send_message(event.user_id, decision.suggested_message)

    # 在 langgraph 节点或任意地方发布事件
    await bus.publish(MemoryChangeEvent(user_id="u1", result=add_result))
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ..models import AddMemoryResult, ChannelType

logger = logging.getLogger(__name__)

# 回调签名：async def handler(event: MemoryChangeEvent) -> None
MemoryChangeCallback = Callable[["MemoryChangeEvent"], Awaitable[None]]


# ------------------------------------------------------------------
# 数据模型
# ------------------------------------------------------------------

class MemoryChangeEvent(BaseModel):
    """一次 add_from_conversation 调用产生的变更事件。"""
    user_id: str
    result: AddMemoryResult
    channel: ChannelType = ChannelType.CHAT
    conversation_time: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def has_changes(self) -> bool:
        return self.result.total_changes > 0

    def summary(self) -> str:
        """给 LLM 看的变更摘要文本。"""
        lines: List[str] = []
        for item in self.result.added:
            lines.append(f"[新增] {item.content}")
        for item in self.result.updated:
            lines.append(f"[更新] {item.content}")
        for mid in self.result.deleted:
            lines.append(f"[删除] memory_id={mid}")
        return "\n".join(lines) if lines else "（无变更）"


class OutreachDecision(BaseModel):
    """LLM 对是否主动触达客户的决策结果。"""
    should_reach_out: bool
    action_type: Literal["promotion", "reminder", "follow_up", "none"] = "none"
    urgency: Literal["low", "medium", "high"] = "low"
    suggested_message: str = ""   # 建议发送给客户的话术
    reason: str = ""              # 决策理由（用于日志/审计）


# ------------------------------------------------------------------
# 事件总线
# ------------------------------------------------------------------

class MemoryEventBus:
    """
    轻量级异步事件总线。

    - 所有回调并发执行，互不阻塞。
    - 单个回调异常不影响其他回调。
    - 可通过 subscribe 装饰器或直接调用注册回调。
    """

    def __init__(self) -> None:
        self._callbacks: List[MemoryChangeCallback] = []

    def subscribe(
        self, callback: MemoryChangeCallback
    ) -> MemoryChangeCallback:
        """
        注册事件回调，可作装饰器使用：

            @bus.subscribe
            async def handler(event): ...
        """
        self._callbacks.append(callback)
        return callback

    def unsubscribe(self, callback: MemoryChangeCallback) -> None:
        self._callbacks = [c for c in self._callbacks if c is not callback]

    async def publish(self, event: MemoryChangeEvent) -> None:
        """
        发布事件，并发调用所有订阅者。

        publish 本身不阻塞调用方——可在 asyncio.create_task 中调用。
        """
        if not self._callbacks:
            return

        async def _safe_call(cb: MemoryChangeCallback) -> None:
            try:
                await cb(event)
            except Exception as exc:
                logger.error(
                    "MemoryEventBus callback %s raised: %s", cb.__name__, exc
                )

        await asyncio.gather(*(_safe_call(cb) for cb in self._callbacks))


# ------------------------------------------------------------------
# 主动触达决策器
# ------------------------------------------------------------------

_OUTREACH_SYSTEM = """\
你是一个客户运营助手。根据用户的最新记忆变化和历史记忆，判断是否需要主动联系客户。

【触发主动联系的条件（满足任意一条）】
1. 用户表达了明确的购买意向或需求（ADD）
2. 用户预算、偏好发生了重要升级（UPDATE）
3. 用户提到生活事件（搬家、结婚、生育、换工作等），可推送关联产品
4. 用户合同/订阅即将到期，可提前跟进续约
5. 用户表达了不满或问题未解决，需要跟进

【不触发主动联系的情况】
- 仅是日常闲聊，无商业价值
- 信息轻微调整，不影响用户需求
- 记忆被删除但无替代需求出现

【action_type 说明】
- promotion : 推销/推荐产品或服务
- reminder  : 提醒用户某个行动（续约、预约、领取等）
- follow_up : 问题跟进或关怀回访
- none      : 不需要主动联系

返回严格 JSON（不要包含任何解释）：
{
  "should_reach_out": true/false,
  "action_type": "promotion"|"reminder"|"follow_up"|"none",
  "urgency": "low"|"medium"|"high",
  "suggested_message": "建议发给客户的话术，100字以内，自然友好",
  "reason": "决策理由，30字以内"
}
"""


class ProactiveOutreachEvaluator:
    """
    基于 LLM 的主动触达评估器。

    注册到 MemoryEventBus 后，每次记忆变更时自动评估是否需要主动联系客户。

    Args:
        memory_manager: MemoryManager 实例（用于获取 LLM 客户端和历史记忆）
        outreach_system_prompt: 覆盖默认的 system prompt（可按业务域定制）
        min_changes_to_evaluate: 至少变更几条记忆才触发评估（避免无意义调用）
    """

    def __init__(
        self,
        memory_manager: Any,  # MemoryManager，避免循环引用用 Any
        outreach_system_prompt: Optional[str] = None,
        min_changes_to_evaluate: int = 1,
    ) -> None:
        self._manager = memory_manager
        self._system_prompt = outreach_system_prompt or _OUTREACH_SYSTEM
        self._min_changes = min_changes_to_evaluate

    async def evaluate(self, event: MemoryChangeEvent) -> OutreachDecision:
        """
        评估本次记忆变更是否触发主动触达。

        Returns:
            OutreachDecision，should_reach_out=False 表示无需主动联系。
        """
        if event.result.total_changes < self._min_changes:
            return OutreachDecision(should_reach_out=False, reason="变更量不足阈值")

        # 拉取该用户的近期记忆作为上下文
        from ..models import MemoryQuery, MemoryType
        existing = await self._manager.query(
            MemoryQuery(
                user_id=event.user_id,
                query=event.summary(),
                top_k=10,
            )
        )
        history_block = "\n".join(
            f"- {r.item.content}" for r in existing
        ) or "（暂无历史记忆）"

        user_content = (
            f"【用户ID】{event.user_id}\n"
            f"【变更时间】{event.conversation_time.isoformat()}\n"
            f"【渠道】{event.channel.value}\n\n"
            f"【本次记忆变更】\n{event.summary()}\n\n"
            f"【用户历史记忆（参考）】\n{history_block}\n\n"
            "请判断是否需要主动联系该用户："
        )

        decision = await self._manager.llm.evaluate_outreach(
            system_prompt=self._system_prompt,
            user_content=user_content,
        )
        return decision

    def as_callback(
        self,
        on_decision: Optional[Callable[[str, OutreachDecision], Awaitable[None]]] = None,
    ) -> MemoryChangeCallback:
        """
        将评估器包装为 MemoryEventBus 回调。

        Args:
            on_decision: 收到决策后的处理函数 async def handler(user_id, decision)
                         若为 None，仅打印日志。
        """
        async def _callback(event: MemoryChangeEvent) -> None:
            decision = await self.evaluate(event)
            if decision.should_reach_out:
                logger.info(
                    "Proactive outreach triggered | user=%s action=%s urgency=%s | %s",
                    event.user_id,
                    decision.action_type,
                    decision.urgency,
                    decision.reason,
                )
                if on_decision:
                    await on_decision(event.user_id, decision)
            else:
                logger.debug(
                    "No outreach needed for user=%s: %s",
                    event.user_id,
                    decision.reason,
                )

        _callback.__name__ = "proactive_outreach_callback"
        return _callback
