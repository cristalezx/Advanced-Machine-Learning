"""
LangGraph 工作流图定义 - 救援服务 Agentic Workflow
"""

from typing import Literal, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .state import RescueState, CustomerInfo, LocationInfo, VehicleInfo, OrderInfo, UICard
from .tools import RESCUE_TOOLS
from .cards import (
    create_info_collection_card,
    create_payment_pending_card,
    create_order_success_card,
    create_order_status_card,
    render_card_to_text
)


# 系统提示词
SYSTEM_PROMPT = """你是一个专业的道路救援服务助手。你的职责是帮助客户完成救援下单或解决咨询问题。

## 救援服务流程：
1. 确认客户需要发起救援
2. 收集必要信息（救援类型、联系方式、位置、车辆信息）
3. 调用发起救援接口获取价格
4. 等待客户确认支付
5. 确认下单并派单
6. 跟踪订单状态直到救援完成

## 支持的救援类型：
- 拖车：车辆无法行驶时拖车服务（基础价格 ¥200）
- 搭电：电瓶没电无法启动（基础价格 ¥80）
- 换胎：轮胎爆胎或漏气（基础价格 ¥100）
- 送油：燃油耗尽（基础价格 ¥150）
- 开锁：钥匙锁车内（基础价格 ¥120）
- 现场维修：简单故障现场修理（基础价格 ¥180）
- 困境救援：车辆陷入泥地等困境（基础价格 ¥250）

## 工作原则：
1. 热情友好，专业高效
2. 主动询问缺失的必要信息
3. 清晰解释每个步骤和费用
4. 在关键节点（信息收集、支付确认、下单成功）时通知系统生成可视化卡片
5. 如果客户只是咨询问题，耐心解答

## 当前状态：
- 阶段：{current_stage}
- 救援类型：{rescue_type}
- 订单信息：{order_info}

请根据用户的输入和当前状态，决定下一步操作。如果需要调用 API，请使用相应的工具。
"""


def get_llm(model_name: str = "gpt-4o-mini"):
    """获取 LLM 实例"""
    if model_name.startswith("gpt"):
        return ChatOpenAI(model=model_name, temperature=0.7)
    elif model_name.startswith("claude"):
        return ChatAnthropic(model=model_name, temperature=0.7)
    else:
        # 默认使用 OpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def create_rescue_agent(model_name: str = "gpt-4o-mini"):
    """创建救援服务 Agent"""
    llm = get_llm(model_name)
    return llm.bind_tools(RESCUE_TOOLS)


def agent_node(state: RescueState, agent) -> dict:
    """Agent 节点 - 处理用户输入并决定下一步"""
    messages = state.get("messages", [])
    current_stage = state.get("current_stage", "greeting")
    rescue_type = state.get("rescue_type", "未选择")
    order_info = state.get("order_info")

    # 构建系统消息
    order_info_str = "无" if not order_info else f"订单号: {order_info.order_id}, 状态: {order_info.status}"
    system_message = SystemMessage(content=SYSTEM_PROMPT.format(
        current_stage=current_stage,
        rescue_type=rescue_type,
        order_info=order_info_str
    ))

    # 调用 LLM
    full_messages = [system_message] + messages
    response = agent.invoke(full_messages)

    return {"messages": [response]}


def should_continue(state: RescueState) -> Literal["tools", "update_state", "human_interrupt", "end"]:
    """路由决策 - 决定下一步走向"""
    messages = state.get("messages", [])
    current_stage = state.get("current_stage", "greeting")

    if not messages:
        return "end"

    last_message = messages[-1]

    # 检查是否有工具调用
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    # 检查是否需要人工干预（等待支付确认等）
    if state.get("need_human_interrupt", False):
        return "human_interrupt"

    # 检查是否结束
    if current_stage in ["completed", "cancelled"]:
        return "end"

    return "update_state"


def update_state_node(state: RescueState) -> dict:
    """更新状态节点 - 根据对话内容更新状态"""
    messages = state.get("messages", [])
    updates = {}

    # 获取最新的用户消息和AI回复
    user_messages = [m for m in messages if isinstance(m, HumanMessage)]
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]

    if user_messages:
        last_user_msg = user_messages[-1].content

        # 尝试从消息中提取信息
        from .nodes import extract_info_from_message
        extracted = extract_info_from_message(last_user_msg)

        if "rescue_type" in extracted:
            updates["rescue_type"] = extracted["rescue_type"]

        if "phone" in extracted:
            customer_info = state.get("customer_info") or CustomerInfo()
            customer_info.phone = extracted["phone"]
            updates["customer_info"] = customer_info

        if "plate_number" in extracted:
            vehicle_info = state.get("vehicle_info") or VehicleInfo()
            vehicle_info.plate_number = extracted["plate_number"]
            updates["vehicle_info"] = vehicle_info

    return updates


def process_tool_result(state: RescueState) -> dict:
    """处理工具调用结果"""
    from .nodes import process_tool_response
    return process_tool_response(state)


def human_interrupt_node(state: RescueState) -> dict:
    """人工干预节点 - 等待用户输入"""
    return {"need_human_interrupt": False}


def build_rescue_graph(model_name: str = "gpt-4o-mini"):
    """构建救援服务工作流图"""

    # 创建 Agent
    agent = create_rescue_agent(model_name)

    # 定义 agent 节点的包装函数
    def call_agent(state: RescueState):
        return agent_node(state, agent)

    # 创建状态图
    workflow = StateGraph(RescueState)

    # 添加节点
    workflow.add_node("agent", call_agent)
    workflow.add_node("tools", ToolNode(RESCUE_TOOLS))
    workflow.add_node("update_state", update_state_node)
    workflow.add_node("process_tool_result", process_tool_result)
    workflow.add_node("human_interrupt", human_interrupt_node)

    # 设置入口
    workflow.set_entry_point("agent")

    # 添加条件边
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "update_state": "update_state",
            "human_interrupt": "human_interrupt",
            "end": END
        }
    )

    # 工具执行后处理结果
    workflow.add_edge("tools", "process_tool_result")
    workflow.add_edge("process_tool_result", "agent")

    # 状态更新后返回等待用户输入
    workflow.add_edge("update_state", END)

    # 人工干预后返回等待用户输入
    workflow.add_edge("human_interrupt", END)

    # 编译图
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


class RescueChatbot:
    """救援服务聊天机器人类"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.graph = build_rescue_graph(model_name)
        self.thread_id = "rescue-session-1"
        self.config = {"configurable": {"thread_id": self.thread_id}}

    def get_initial_state(self) -> RescueState:
        """获取初始状态"""
        return {
            "messages": [],
            "customer_info": None,
            "location_info": None,
            "vehicle_info": None,
            "order_info": None,
            "rescue_type": None,
            "current_stage": "greeting",
            "ui_cards": [],
            "need_human_interrupt": False,
            "context": {}
        }

    def chat(self, user_input: str, state: Optional[RescueState] = None) -> tuple[str, RescueState, list[UICard]]:
        """
        处理用户输入

        Args:
            user_input: 用户输入文本
            state: 当前状态（如果为None则使用初始状态）

        Returns:
            (AI回复, 更新后的状态, 新生成的卡片列表)
        """
        if state is None:
            state = self.get_initial_state()

        # 添加用户消息
        state["messages"].append(HumanMessage(content=user_input))

        # 清空之前的卡片
        previous_cards_count = len(state.get("ui_cards", []))

        # 运行图
        result = self.graph.invoke(state, self.config)

        # 获取AI回复
        ai_response = ""
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                ai_response = msg.content
                break

        # 获取新生成的卡片
        all_cards = result.get("ui_cards", [])
        new_cards = all_cards[previous_cards_count:]

        return ai_response, result, new_cards

    def reset(self):
        """重置会话"""
        self.thread_id = f"rescue-session-{id(self)}"
        self.config = {"configurable": {"thread_id": self.thread_id}}


def create_simple_rescue_graph():
    """
    创建简化版救援服务图（不依赖外部 LLM，用于演示流程）
    """

    def greeting_node(state: RescueState) -> dict:
        """问候节点"""
        return {
            "messages": [AIMessage(content="您好！我是道路救援服务助手。请问您需要什么帮助？\n\n我们提供以下服务：\n1. 拖车服务\n2. 搭电服务\n3. 换胎服务\n4. 送油服务\n5. 开锁服务\n6. 现场维修\n7. 困境救援\n\n您也可以查询现有订单状态。")],
            "current_stage": "confirm_rescue"
        }

    def collect_info_node(state: RescueState) -> dict:
        """收集信息节点"""
        ui_cards = state.get("ui_cards", [])

        # 生成信息收集卡片
        card = create_info_collection_card(
            customer_info=state.get("customer_info"),
            location_info=state.get("location_info"),
            vehicle_info=state.get("vehicle_info"),
            rescue_type=state.get("rescue_type"),
            missing_fields=["救援类型", "联系电话", "救援地址"]
        )
        ui_cards.append(card)

        return {
            "messages": [AIMessage(content="好的，为了发起救援，我需要收集一些信息。请提供以下内容：\n1. 您的联系电话\n2. 当前位置（详细地址）\n3. 车辆信息（车牌号、车型）")],
            "ui_cards": ui_cards,
            "current_stage": "collect_info"
        }

    def router(state: RescueState) -> str:
        """路由决策"""
        stage = state.get("current_stage", "greeting")

        if stage == "greeting":
            return "greeting"
        elif stage in ["confirm_rescue", "collect_info"]:
            return "collect_info"
        elif stage == "payment_pending":
            return "payment"
        elif stage in ["completed", "cancelled"]:
            return "end"
        else:
            return "greeting"

    # 创建图
    workflow = StateGraph(RescueState)

    workflow.add_node("greeting", greeting_node)
    workflow.add_node("collect_info", collect_info_node)

    workflow.set_entry_point("greeting")

    workflow.add_conditional_edges(
        "greeting",
        lambda s: "collect" if s.get("current_stage") == "confirm_rescue" else "end",
        {"collect": "collect_info", "end": END}
    )

    workflow.add_edge("collect_info", END)

    return workflow.compile()
