"""
LLM 驱动的图路由 - 救援服务工作流

核心设计：
1. LLM 输出结构化的路由决策（不是硬编码状态机）
2. LangGraph 根据 LLM 输出决定边的走向
3. 通过 Prompt + 输出校验保证流程合理性
4. 图结构清晰，但路由逻辑由 LLM 动态判断
"""

from typing import Literal, Optional, Annotated, Any
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import json

from .tools import RESCUE_TOOLS, initiate_rescue_order, confirm_order_payment, cancel_rescue_order, query_order_status
from .cards import (
    create_info_collection_card,
    create_payment_pending_card,
    create_order_success_card,
    create_order_status_card,
    render_card_to_text
)


# ==================== 1. 结构化输出定义 ====================

class RouterDecision(BaseModel):
    """LLM 路由决策的结构化输出"""

    # 识别的用户意图
    user_intent: Literal[
        "request_rescue",      # 请求救援服务
        "provide_info",        # 提供信息
        "confirm",             # 确认（信息/支付）
        "reject",              # 拒绝/要修改
        "cancel",              # 取消
        "query",               # 查询（订单/价格等）
        "chitchat",            # 闲聊
        "unclear"              # 不清楚
    ] = Field(description="识别出的用户意图")

    # 下一步动作
    next_action: Literal[
        "greet",               # 问候
        "collect_info",        # 继续收集信息
        "show_summary",        # 展示信息摘要让用户确认
        "create_order",        # 创建订单（调用 API）
        "wait_payment",        # 等待支付确认
        "confirm_payment",     # 确认支付（调用 API）
        "show_status",         # 展示订单状态
        "cancel_order",        # 取消订单
        "answer_question",     # 回答问题
        "end_conversation"     # 结束对话
    ] = Field(description="决定的下一步动作")

    # 需要调用的工具（可选）
    tool_to_call: Optional[Literal[
        "initiate_rescue_order",
        "confirm_order_payment",
        "cancel_rescue_order",
        "query_order_status",
        "none"
    ]] = Field(default="none", description="需要调用的工具")

    # 从用户输入中提取的信息
    extracted_info: dict = Field(default_factory=dict, description="提取的结构化信息")

    # 回复内容
    response: str = Field(description="给用户的回复")

    # 判断理由（用于调试）
    reasoning: str = Field(description="做出此决策的理由")


# ==================== 2. 图状态定义 ====================

class GraphState(BaseModel):
    """图状态"""
    messages: Annotated[list, add_messages] = Field(default_factory=list)

    # 业务数据
    rescue_type: Optional[str] = None
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None
    address: Optional[str] = None
    plate_number: Optional[str] = None

    # 订单数据
    order_id: Optional[str] = None
    price: Optional[float] = None
    payment_confirmed: bool = False
    rescue_worker: Optional[str] = None
    eta: Optional[str] = None

    # 路由相关
    current_action: str = "greet"
    last_decision: Optional[dict] = None

    # UI 卡片
    ui_cards: list = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


# ==================== 3. Router Prompt ====================

ROUTER_PROMPT = """你是道路救援服务的智能路由器。根据对话上下文，判断用户意图并决定下一步动作。

## 当前状态
- 救援类型: {rescue_type}
- 客户电话: {phone}
- 地址: {address}
- 车牌: {plate}
- 订单ID: {order_id}
- 订单金额: {price}
- 支付状态: {payment_status}

## 对话历史
{history}

## 用户最新输入
{user_input}

## 决策规则

### 意图识别
- 用户说"车坏了/没电/爆胎/需要救援"等 → request_rescue
- 用户提供电话/地址/车牌等信息 → provide_info
- 用户说"确认/没问题/可以/下单/支付"等 → confirm
- 用户说"不对/改一下/重新"等 → reject
- 用户说"取消/不要了"等 → cancel
- 用户问"多少钱/多久到/订单状态"等 → query
- 用户闲聊或无关内容 → chitchat

### 动作决策
1. 如果还没开始或用户刚请求救援 → collect_info
2. 如果信息不完整（缺少 rescue_type/phone/address 任一）→ collect_info
3. 如果信息完整但还没确认 → show_summary
4. 如果用户确认了信息且没有订单 → create_order（需要调用 initiate_rescue_order）
5. 如果有订单但未支付，且用户确认支付 → confirm_payment（需要调用 confirm_order_payment）
6. 如果有订单未支付，用户还没确认 → wait_payment
7. 如果用户要取消 → cancel_order（需要调用 cancel_rescue_order）
8. 如果用户查询订单 → show_status（需要调用 query_order_status）
9. 如果是闲聊或问题 → answer_question

### 信息提取
从用户输入中提取以下信息（如果有）：
- rescue_type: 拖车/搭电/换胎/送油/开锁/现场维修/困境救援
- phone: 11位手机号
- address: 地址
- plate_number: 车牌号
- name: 姓名

## 输出要求
返回 JSON 格式，包含：
- user_intent: 识别的意图
- next_action: 下一步动作
- tool_to_call: 需要调用的工具（如果有）
- extracted_info: 提取的信息
- response: 自然、友好的回复（不要机械）
- reasoning: 决策理由

## 回复风格
- 像朋友聊天，不要像客服机器人
- 可以适当表达关心（"别着急"、"马上帮您安排"）
- 收集信息时不要一个个问，尽量自然地融入对话
- 如果用户问问题，先回答再继续流程
"""


# ==================== 4. 核心节点 ====================

def create_router_chain(llm):
    """创建路由链"""

    def router_node(state: GraphState) -> dict:
        """路由节点 - LLM 判断下一步"""

        # 构建历史
        history_lines = []
        for msg in state.messages[-10:]:
            if isinstance(msg, HumanMessage):
                history_lines.append(f"用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                history_lines.append(f"助手: {msg.content[:100]}...")

        # 获取最新用户输入
        user_input = ""
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                user_input = msg.content
                break

        # 构建 prompt
        prompt = ROUTER_PROMPT.format(
            rescue_type=state.rescue_type or "未知",
            phone=state.customer_phone or "未提供",
            address=state.address or "未提供",
            plate=state.plate_number or "未提供",
            order_id=state.order_id or "无",
            price=f"¥{state.price}" if state.price else "未计算",
            payment_status="已支付" if state.payment_confirmed else "未支付",
            history="\n".join(history_lines) or "（新对话）",
            user_input=user_input
        )

        # 调用 LLM（使用 structured output）
        structured_llm = llm.with_structured_output(RouterDecision)
        decision: RouterDecision = structured_llm.invoke([
            SystemMessage(content=prompt)
        ])

        # 更新状态
        updates = {
            "current_action": decision.next_action,
            "last_decision": decision.model_dump(),
            "messages": [AIMessage(content=decision.response)]
        }

        # 合并提取的信息
        for key, value in decision.extracted_info.items():
            if value:
                if key == "rescue_type":
                    updates["rescue_type"] = value
                elif key == "phone":
                    updates["customer_phone"] = value
                elif key == "address":
                    updates["address"] = value
                elif key == "plate_number":
                    updates["plate_number"] = value
                elif key == "name":
                    updates["customer_name"] = value

        return updates

    return router_node


def tool_executor_node(state: GraphState) -> dict:
    """工具执行节点"""
    decision = state.last_decision
    if not decision:
        return {}

    tool_name = decision.get("tool_to_call", "none")
    if tool_name == "none" or not tool_name:
        return {}

    ui_cards = list(state.ui_cards)
    updates = {}

    try:
        if tool_name == "initiate_rescue_order":
            # 调用发起救援 API
            from . import mock_api
            response = mock_api.initiate_rescue(
                rescue_type=state.rescue_type or "救援服务",
                customer_name=state.customer_name or "客户",
                customer_phone=state.customer_phone or "",
                address=state.address or "",
                plate_number=state.plate_number or ""
            )
            updates["order_id"] = response.order_id
            updates["price"] = response.price

            # 生成待支付卡片
            from .state import OrderInfo
            order = OrderInfo(
                order_id=response.order_id,
                price=response.price,
                rescue_type=state.rescue_type,
                status="pending_payment"
            )
            ui_cards.append(create_payment_pending_card(order))
            updates["ui_cards"] = ui_cards

            # 添加订单信息到回复
            updates["messages"] = [AIMessage(
                content=f"订单已创建！\n\n📋 订单号：{response.order_id}\n💰 费用：¥{response.price:.2f}\n\n确认支付后，我们马上安排师傅过去~"
            )]

        elif tool_name == "confirm_order_payment":
            # 调用确认支付 API
            from . import mock_api
            response = mock_api.confirm_order(state.order_id, payment_confirmed=True)
            updates["payment_confirmed"] = True
            updates["rescue_worker"] = response.rescue_worker
            updates["eta"] = response.estimated_arrival

            # 生成成功卡片
            from .state import OrderInfo
            order = OrderInfo(
                order_id=state.order_id,
                price=state.price,
                rescue_type=state.rescue_type,
                status="dispatched",
                rescue_worker=response.rescue_worker,
                estimated_arrival=response.estimated_arrival
            )
            ui_cards.append(create_order_success_card(order))
            updates["ui_cards"] = ui_cards

            updates["messages"] = [AIMessage(
                content=f"🎉 支付成功！{response.rescue_worker} 正在赶来，预计 {response.estimated_arrival} 到达。保持电话畅通哦~"
            )]

        elif tool_name == "cancel_rescue_order":
            from . import mock_api
            response = mock_api.cancel_order(state.order_id, reason="用户取消")
            updates["messages"] = [AIMessage(
                content=f"订单已取消。{'退款 ¥' + str(response.refund_amount) + ' 将原路返回。' if response.refund_amount > 0 else ''}有需要随时找我~"
            )]

        elif tool_name == "query_order_status":
            from . import mock_api
            response = mock_api.query_order_status(state.order_id)
            from .state import OrderInfo
            order = OrderInfo(
                order_id=state.order_id,
                price=state.price,
                rescue_type=state.rescue_type,
                status=response.status,
                rescue_worker=response.rescue_worker,
                estimated_arrival=response.estimated_arrival
            )
            ui_cards.append(create_order_status_card(order, None))
            updates["ui_cards"] = ui_cards

    except Exception as e:
        updates["messages"] = [AIMessage(content=f"操作出现问题：{str(e)}，请稍后重试。")]

    return updates


def should_call_tool(state: GraphState) -> Literal["tool", "respond"]:
    """判断是否需要调用工具"""
    decision = state.last_decision
    if not decision:
        return "respond"

    tool_name = decision.get("tool_to_call", "none")
    if tool_name and tool_name != "none":
        return "tool"
    return "respond"


def should_continue(state: GraphState) -> Literal["continue", "end"]:
    """判断是否继续"""
    action = state.current_action
    if action == "end_conversation":
        return "end"
    return "continue"


# ==================== 5. 构建图 ====================

def build_llm_routed_graph(model_name: str = "gpt-4o-mini"):
    """构建 LLM 路由的图"""

    llm = ChatOpenAI(model=model_name, temperature=0.7)
    router_node = create_router_chain(llm)

    # 创建图
    workflow = StateGraph(GraphState)

    # 添加节点
    workflow.add_node("router", router_node)           # LLM 路由决策
    workflow.add_node("tool_executor", tool_executor_node)  # 工具执行
    workflow.add_node("respond", lambda s: {})         # 响应节点（直接返回）

    # 设置入口
    workflow.set_entry_point("router")

    # 条件边：router 之后判断是否需要调用工具
    workflow.add_conditional_edges(
        "router",
        should_call_tool,
        {
            "tool": "tool_executor",
            "respond": "respond"
        }
    )

    # 工具执行后结束
    workflow.add_edge("tool_executor", END)
    workflow.add_edge("respond", END)

    # 编译
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# ==================== 6. Chatbot 类 ====================

class LLMRoutedChatbot:
    """LLM 路由的聊天机器人"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.graph = build_llm_routed_graph(model_name)
        self.thread_id = "llm-routed-session"
        self.config = {"configurable": {"thread_id": self.thread_id}}
        self.state = GraphState()

    def chat(self, user_input: str) -> tuple[str, list, dict]:
        """
        处理用户输入

        Returns:
            (回复文本, UI卡片列表, 路由决策详情)
        """
        # 添加用户消息
        input_state = self.state.model_copy()
        input_state.messages = list(self.state.messages) + [HumanMessage(content=user_input)]

        # 记录之前的卡片数量
        prev_cards = len(input_state.ui_cards)

        # 运行图
        result = self.graph.invoke(input_state.model_dump(), self.config)

        # 更新状态
        self.state = GraphState(**result)

        # 获取回复
        response = ""
        for msg in reversed(self.state.messages):
            if isinstance(msg, AIMessage):
                response = msg.content
                break

        # 获取新卡片
        new_cards = self.state.ui_cards[prev_cards:]

        # 获取决策详情
        decision = self.state.last_decision or {}

        return response, new_cards, decision

    def get_state(self) -> dict:
        """获取当前状态"""
        return {
            "rescue_type": self.state.rescue_type,
            "phone": self.state.customer_phone,
            "address": self.state.address,
            "order_id": self.state.order_id,
            "price": self.state.price,
            "current_action": self.state.current_action,
        }

    def reset(self):
        """重置"""
        self.state = GraphState()
        self.thread_id = f"llm-routed-{id(self)}"
        self.config = {"configurable": {"thread_id": self.thread_id}}


# ==================== 7. 图结构可视化 ====================

"""
                    ┌─────────────────┐
                    │   User Input    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │     Router      │  ← LLM 判断意图 + 决定下一步
                    │  (LLM Decision) │    输出：next_action, tool_to_call
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
        tool_to_call?                  tool_to_call?
           = "xxx"                        = "none"
              │                             │
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │  Tool Executor  │           │     Respond     │
    │  (调用 API)      │           │   (直接返回)    │
    └────────┬────────┘           └────────┬────────┘
             │                             │
             └──────────────┬──────────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │      END        │
                    └─────────────────┘


关键点：
1. Router 是 LLM，输出结构化的 RouterDecision
2. 图的边走向由 LLM 输出的 tool_to_call 决定
3. 不是硬编码的 if-else，而是 LLM 动态判断
4. 通过 Prompt 引导 LLM 做出合理决策
5. 通过 Pydantic 结构化输出保证格式正确
"""
