"""
严格状态机版本的救援服务工作流

核心设计原则：
1. 状态驱动路由 - 根据 current_stage 决定下一步，而非 LLM 判断
2. 分阶段工具限制 - 每个阶段只暴露允许的工具
3. 强制校验 - 关键节点必须校验数据完整性
4. Human-in-the-loop - 支付确认等关键操作必须等待用户确认
"""

from typing import Literal, Optional, Annotated
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

from .state import RescueState, CustomerInfo, LocationInfo, VehicleInfo, OrderInfo, UICard
from .tools import (
    initiate_rescue_order,
    confirm_order_payment,
    cancel_rescue_order,
    query_order_status,
    get_available_rescue_types,
)
from .cards import (
    create_info_collection_card,
    create_payment_pending_card,
    create_order_success_card,
    create_order_status_card,
    create_rescue_complete_card,
    render_card_to_text
)


# ==================== 阶段定义 ====================
class WorkflowStage:
    """工作流阶段常量"""
    GREETING = "greeting"                    # 初始问候
    INTENT_RECOGNITION = "intent_recognition"  # 意图识别
    COLLECT_INFO = "collect_info"            # 收集信息
    CONFIRM_INFO = "confirm_info"            # 确认信息
    CREATE_ORDER = "create_order"            # 创建订单
    PENDING_PAYMENT = "pending_payment"      # 等待支付
    CONFIRM_PAYMENT = "confirm_payment"      # 确认支付
    DISPATCHING = "dispatching"              # 派单中
    DISPATCHED = "dispatched"                # 已派单
    RESCUE_IN_PROGRESS = "rescue_in_progress"  # 救援中
    COMPLETED = "completed"                  # 完成
    CANCELLED = "cancelled"                  # 已取消
    CONSULTING = "consulting"                # 咨询模式


# ==================== 阶段允许的工具 ====================
STAGE_ALLOWED_TOOLS = {
    WorkflowStage.GREETING: [get_available_rescue_types],
    WorkflowStage.INTENT_RECOGNITION: [get_available_rescue_types],
    WorkflowStage.COLLECT_INFO: [get_available_rescue_types],
    WorkflowStage.CONFIRM_INFO: [initiate_rescue_order],  # 只能发起订单
    WorkflowStage.CREATE_ORDER: [initiate_rescue_order],
    WorkflowStage.PENDING_PAYMENT: [],  # 等待支付时不允许任何工具调用
    WorkflowStage.CONFIRM_PAYMENT: [confirm_order_payment, cancel_rescue_order],
    WorkflowStage.DISPATCHING: [query_order_status],
    WorkflowStage.DISPATCHED: [query_order_status, cancel_rescue_order],
    WorkflowStage.RESCUE_IN_PROGRESS: [query_order_status],
    WorkflowStage.COMPLETED: [query_order_status],
    WorkflowStage.CANCELLED: [],
    WorkflowStage.CONSULTING: [get_available_rescue_types, query_order_status],
}


# ==================== 阶段转换规则 ====================
VALID_TRANSITIONS = {
    WorkflowStage.GREETING: [WorkflowStage.INTENT_RECOGNITION, WorkflowStage.CONSULTING],
    WorkflowStage.INTENT_RECOGNITION: [WorkflowStage.COLLECT_INFO, WorkflowStage.CONSULTING],
    WorkflowStage.COLLECT_INFO: [WorkflowStage.CONFIRM_INFO, WorkflowStage.CANCELLED],
    WorkflowStage.CONFIRM_INFO: [WorkflowStage.CREATE_ORDER, WorkflowStage.COLLECT_INFO],
    WorkflowStage.CREATE_ORDER: [WorkflowStage.PENDING_PAYMENT],
    WorkflowStage.PENDING_PAYMENT: [WorkflowStage.CONFIRM_PAYMENT, WorkflowStage.CANCELLED],
    WorkflowStage.CONFIRM_PAYMENT: [WorkflowStage.DISPATCHING, WorkflowStage.CANCELLED],
    WorkflowStage.DISPATCHING: [WorkflowStage.DISPATCHED],
    WorkflowStage.DISPATCHED: [WorkflowStage.RESCUE_IN_PROGRESS, WorkflowStage.CANCELLED],
    WorkflowStage.RESCUE_IN_PROGRESS: [WorkflowStage.COMPLETED],
    WorkflowStage.COMPLETED: [],
    WorkflowStage.CANCELLED: [],
    WorkflowStage.CONSULTING: [WorkflowStage.INTENT_RECOGNITION, WorkflowStage.GREETING],
}


# ==================== 阶段特定的 Prompt ====================
STAGE_PROMPTS = {
    WorkflowStage.GREETING: """你是道路救援服务助手。请友好地问候用户，并询问需要什么帮助。
可以简单介绍我们提供的救援服务：拖车、搭电、换胎、送油、开锁、现场维修、困境救援。""",

    WorkflowStage.INTENT_RECOGNITION: """根据用户的回复，判断用户的意图：
- 如果用户需要救援服务，识别救援类型
- 如果用户是咨询问题，切换到咨询模式
- 如果用户要查询订单，询问订单号""",

    WorkflowStage.COLLECT_INFO: """你正在收集救援所需信息。当前已收集：
- 救援类型：{rescue_type}
- 客户姓名：{customer_name}
- 联系电话：{customer_phone}
- 救援地址：{address}
- 车牌号：{plate_number}

缺失信息：{missing_fields}

请友好地询问缺失的信息。只询问缺失的部分，不要重复已收集的信息。""",

    WorkflowStage.CONFIRM_INFO: """信息收集完毕，请向用户确认以下信息是否正确：
- 救援类型：{rescue_type}
- 联系方式：{customer_name} {customer_phone}
- 救援地址：{address}
- 车牌号：{plate_number}

询问用户是否确认提交。""",

    WorkflowStage.PENDING_PAYMENT: """订单已创建，等待用户支付确认。
订单信息：
- 订单号：{order_id}
- 费用：¥{price}

请告知用户订单详情，等待用户确认支付。
【重要】在用户明确表示"确认支付"或"支付"之前，不要进行任何操作。""",

    WorkflowStage.DISPATCHED: """订单已派单，救援人员正在赶往现场。
- 救援人员：{rescue_worker}
- 预计到达：{estimated_arrival}

告知用户派单成功的信息，并表示可以随时查询进度。""",
}


def get_llm(model_name: str = "gpt-4o-mini"):
    """获取 LLM 实例"""
    if model_name.startswith("gpt"):
        return ChatOpenAI(model=model_name, temperature=0.3)  # 降低温度增加确定性
    elif model_name.startswith("claude"):
        return ChatAnthropic(model=model_name, temperature=0.3)
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


def get_stage_tools(stage: str) -> list:
    """获取当前阶段允许的工具"""
    return STAGE_ALLOWED_TOOLS.get(stage, [])


def validate_transition(current_stage: str, target_stage: str) -> bool:
    """验证状态转换是否合法"""
    valid_targets = VALID_TRANSITIONS.get(current_stage, [])
    return target_stage in valid_targets


def check_required_info(state: RescueState) -> list[str]:
    """检查缺失的必要信息"""
    missing = []

    if not state.get("rescue_type"):
        missing.append("救援类型")

    customer = state.get("customer_info")
    if not customer or not customer.name:
        missing.append("客户姓名")
    if not customer or not customer.phone:
        missing.append("联系电话")

    location = state.get("location_info")
    if not location or not location.address:
        missing.append("救援地址")

    return missing


# ==================== 节点定义 ====================

def greeting_node(state: RescueState) -> dict:
    """问候节点 - 固定输出"""
    greeting_msg = """您好！我是道路救援服务助手 🚗

我们提供以下救援服务：
1. 🚛 拖车服务 - 车辆无法行驶
2. 🔋 搭电服务 - 电瓶没电
3. 🛞 换胎服务 - 轮胎爆胎
4. ⛽ 送油服务 - 燃油耗尽
5. 🔑 开锁服务 - 钥匙锁车内
6. 🔧 现场维修 - 简单故障
7. 🆘 困境救援 - 车辆被困

请问您需要什么帮助？"""

    return {
        "messages": [AIMessage(content=greeting_msg)],
        "current_stage": WorkflowStage.INTENT_RECOGNITION
    }


def intent_recognition_node(state: RescueState, llm) -> dict:
    """意图识别节点 - LLM 判断用户意图"""
    messages = state.get("messages", [])
    last_user_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    # 使用 LLM 进行意图识别
    intent_prompt = f"""分析用户消息，判断意图类型。只返回 JSON 格式：

用户消息："{last_user_msg}"

可能的救援类型：拖车、搭电、换胎、送油、开锁、现场维修、困境救援

返回格式：
{{"intent": "rescue|consult|query_order", "rescue_type": "类型或null", "extracted_info": {{}}}}

只返回 JSON，不要其他内容。"""

    response = llm.invoke([HumanMessage(content=intent_prompt)])

    # 解析意图
    import json
    try:
        intent_data = json.loads(response.content)
        intent = intent_data.get("intent", "consult")
        rescue_type = intent_data.get("rescue_type")
    except:
        intent = "rescue" if any(kw in last_user_msg for kw in
            ["救援", "拖车", "搭电", "换胎", "送油", "开锁", "维修", "困境", "没电", "爆胎", "没油"]) else "consult"
        rescue_type = None
        # 简单关键词匹配
        for rt in ["拖车", "搭电", "换胎", "送油", "开锁", "现场维修", "困境救援"]:
            if rt in last_user_msg:
                rescue_type = rt
                break

    if intent == "rescue":
        next_stage = WorkflowStage.COLLECT_INFO
        reply = f"好的，我来帮您安排{'救援' if not rescue_type else rescue_type}服务。"
    else:
        next_stage = WorkflowStage.CONSULTING
        reply = "好的，请问您想咨询什么问题？"

    updates = {
        "current_stage": next_stage,
        "messages": [AIMessage(content=reply)]
    }

    if rescue_type:
        updates["rescue_type"] = rescue_type

    return updates


def collect_info_node(state: RescueState, llm) -> dict:
    """信息收集节点 - 收集必要信息"""
    messages = state.get("messages", [])
    missing_fields = check_required_info(state)

    # 先尝试从最新消息提取信息
    updates = extract_info_from_messages(state)

    # 重新检查缺失字段
    temp_state = {**state, **updates}
    missing_fields = check_required_info(temp_state)

    ui_cards = state.get("ui_cards", []).copy()

    if not missing_fields:
        # 信息收集完成，生成确认卡片
        card = create_info_collection_card(
            customer_info=updates.get("customer_info") or state.get("customer_info"),
            location_info=updates.get("location_info") or state.get("location_info"),
            vehicle_info=updates.get("vehicle_info") or state.get("vehicle_info"),
            rescue_type=updates.get("rescue_type") or state.get("rescue_type"),
            missing_fields=[]
        )
        ui_cards.append(card)

        updates["current_stage"] = WorkflowStage.CONFIRM_INFO
        updates["ui_cards"] = ui_cards
        updates["messages"] = [AIMessage(content="信息已收集完毕，请确认以上信息是否正确？")]
    else:
        # 继续收集信息
        prompt = STAGE_PROMPTS[WorkflowStage.COLLECT_INFO].format(
            rescue_type=state.get("rescue_type") or "未选择",
            customer_name=state.get("customer_info", CustomerInfo()).name if state.get("customer_info") else "未提供",
            customer_phone=state.get("customer_info", CustomerInfo()).phone if state.get("customer_info") else "未提供",
            address=state.get("location_info", LocationInfo()).address if state.get("location_info") else "未提供",
            plate_number=state.get("vehicle_info", VehicleInfo()).plate_number if state.get("vehicle_info") else "未提供",
            missing_fields="、".join(missing_fields)
        )

        response = llm.invoke([
            SystemMessage(content=prompt),
            *messages[-4:]  # 只用最近几条消息
        ])

        updates["messages"] = [response]
        updates["current_stage"] = WorkflowStage.COLLECT_INFO

    return updates


def confirm_info_node(state: RescueState, llm) -> dict:
    """确认信息节点 - 等待用户确认"""
    messages = state.get("messages", [])
    last_user_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content.lower()
            break

    # 判断用户是否确认
    confirm_keywords = ["确认", "正确", "没问题", "是的", "对", "好的", "可以", "提交"]
    reject_keywords = ["不对", "不正确", "修改", "错了", "改", "重新"]

    if any(kw in last_user_msg for kw in confirm_keywords):
        # 用户确认，进入创建订单
        return {
            "current_stage": WorkflowStage.CREATE_ORDER,
            "messages": [AIMessage(content="好的，正在为您创建订单...")]
        }
    elif any(kw in last_user_msg for kw in reject_keywords):
        # 用户要求修改，返回收集信息
        return {
            "current_stage": WorkflowStage.COLLECT_INFO,
            "messages": [AIMessage(content="好的，请告诉我需要修改哪些信息？")]
        }
    else:
        # 不确定，再次询问
        return {
            "messages": [AIMessage(content="请确认以上信息是否正确？回复"确认"继续下单，或告诉我需要修改的内容。")]
        }


def create_order_node(state: RescueState) -> dict:
    """创建订单节点 - 调用 API 创建订单"""
    customer = state.get("customer_info", CustomerInfo())
    location = state.get("location_info", LocationInfo())
    vehicle = state.get("vehicle_info", VehicleInfo())
    rescue_type = state.get("rescue_type", "救援服务")

    # 调用 API
    from . import mock_api
    response = mock_api.initiate_rescue(
        rescue_type=rescue_type,
        customer_name=customer.name or "客户",
        customer_phone=customer.phone or "",
        address=location.address or "",
        plate_number=vehicle.plate_number if vehicle else "",
        latitude=location.latitude if location else None,
        longitude=location.longitude if location else None
    )

    # 创建订单信息
    order_info = OrderInfo(
        order_id=response.order_id,
        price=response.price,
        rescue_type=rescue_type,
        status="pending_payment"
    )

    # 生成待支付卡片
    ui_cards = state.get("ui_cards", []).copy()
    payment_card = create_payment_pending_card(order_info)
    ui_cards.append(payment_card)

    return {
        "order_info": order_info,
        "current_stage": WorkflowStage.PENDING_PAYMENT,
        "ui_cards": ui_cards,
        "messages": [AIMessage(content=f"订单创建成功！\n\n订单号：{response.order_id}\n费用：¥{response.price:.2f}\n\n请确认支付以继续。")]
    }


def pending_payment_node(state: RescueState) -> dict:
    """等待支付节点 - Human-in-the-loop"""
    messages = state.get("messages", [])
    last_user_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content.lower()
            break

    # 判断用户意图
    pay_keywords = ["支付", "付款", "确认", "付", "交钱", "给钱"]
    cancel_keywords = ["取消", "不要了", "算了", "不需要"]

    if any(kw in last_user_msg for kw in pay_keywords):
        return {
            "current_stage": WorkflowStage.CONFIRM_PAYMENT,
            "messages": [AIMessage(content="好的，正在确认支付...")]
        }
    elif any(kw in last_user_msg for kw in cancel_keywords):
        return {
            "current_stage": WorkflowStage.CANCELLED,
            "messages": [AIMessage(content="好的，订单已取消。如需帮助请随时联系我们。")]
        }
    else:
        order = state.get("order_info")
        return {
            "messages": [AIMessage(content=f"订单 {order.order_id} 等待支付，费用 ¥{order.price:.2f}。\n\n请回复"确认支付"继续，或"取消订单"取消。")]
        }


def confirm_payment_node(state: RescueState) -> dict:
    """确认支付节点 - 调用支付确认 API"""
    order = state.get("order_info")

    from . import mock_api
    response = mock_api.confirm_order(order.order_id, payment_confirmed=True)

    if response.success:
        # 更新订单信息
        order.status = "dispatched"
        order.rescue_worker = response.rescue_worker
        order.estimated_arrival = response.estimated_arrival

        # 生成下单成功卡片
        ui_cards = state.get("ui_cards", []).copy()
        success_card = create_order_success_card(order)
        ui_cards.append(success_card)

        return {
            "order_info": order,
            "current_stage": WorkflowStage.DISPATCHED,
            "ui_cards": ui_cards,
            "messages": [AIMessage(content=f"🎉 支付成功！订单已派单。\n\n救援人员：{response.rescue_worker}\n预计到达：{response.estimated_arrival}\n\n请保持电话畅通，救援人员会尽快与您联系。")]
        }
    else:
        return {
            "messages": [AIMessage(content=f"支付确认失败：{response.message}，请重试。")]
        }


def dispatched_node(state: RescueState) -> dict:
    """已派单节点 - 等待救援完成"""
    messages = state.get("messages", [])
    last_user_msg = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content.lower()
            break

    order = state.get("order_info")

    # 查询状态
    if any(kw in last_user_msg for kw in ["状态", "进度", "到哪了", "多久"]):
        from . import mock_api
        status_response = mock_api.query_order_status(order.order_id)

        ui_cards = state.get("ui_cards", []).copy()
        status_card = create_order_status_card(order, state.get("location_info"))
        ui_cards.append(status_card)

        return {
            "ui_cards": ui_cards,
            "messages": [AIMessage(content=f"订单状态：{status_response.status}\n救援人员：{order.rescue_worker}\n预计到达：{order.estimated_arrival}")]
        }

    # 取消订单
    if any(kw in last_user_msg for kw in ["取消"]):
        from . import mock_api
        cancel_response = mock_api.cancel_order(order.order_id, reason="用户取消")
        if cancel_response.success:
            return {
                "current_stage": WorkflowStage.CANCELLED,
                "messages": [AIMessage(content=f"订单已取消，退款金额 ¥{cancel_response.refund_amount:.2f}")]
            }

    return {
        "messages": [AIMessage(content="救援人员正在赶往现场，请耐心等待。您可以随时询问订单状态。")]
    }


def consulting_node(state: RescueState, llm) -> dict:
    """咨询节点 - 回答用户问题"""
    messages = state.get("messages", [])

    # 允许使用查询工具
    tools = [get_available_rescue_types, query_order_status]
    llm_with_tools = llm.bind_tools(tools)

    system_msg = """你是道路救援服务的客服助手。回答用户的咨询问题。
可以介绍服务内容、价格、流程等。如果用户想发起救援，告知用户说"我需要救援"即可开始。"""

    response = llm_with_tools.invoke([
        SystemMessage(content=system_msg),
        *messages[-6:]
    ])

    return {"messages": [response]}


def extract_info_from_messages(state: RescueState) -> dict:
    """从消息中提取信息"""
    import re
    messages = state.get("messages", [])
    updates = {}

    for msg in messages:
        if not isinstance(msg, HumanMessage):
            continue
        text = msg.content

        # 提取电话
        phone_match = re.search(r'1[3-9]\d{9}', text)
        if phone_match:
            customer = state.get("customer_info") or CustomerInfo()
            customer.phone = phone_match.group()
            updates["customer_info"] = customer

        # 提取车牌
        plate_match = re.search(r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领][A-Z][A-Z0-9]{5,6}', text)
        if plate_match:
            vehicle = state.get("vehicle_info") or VehicleInfo()
            vehicle.plate_number = plate_match.group()
            updates["vehicle_info"] = vehicle

        # 提取地址（简单匹配包含"路"、"街"、"区"等关键词的内容）
        addr_patterns = [
            r'在(.{2,30}(?:路|街|区|大厦|广场|小区|公司|酒店|商场|医院|学校|站))',
            r'地址[是：:]*(.{5,50})',
            r'位置[是：:]*(.{5,50})',
        ]
        for pattern in addr_patterns:
            addr_match = re.search(pattern, text)
            if addr_match:
                location = state.get("location_info") or LocationInfo()
                location.address = addr_match.group(1).strip()
                updates["location_info"] = location
                break

        # 提取姓名
        name_patterns = [
            r'我叫(\w{2,4})',
            r'姓名[是：:]*(\w{2,4})',
            r'我是(\w{2,4})',
        ]
        for pattern in name_patterns:
            name_match = re.search(pattern, text)
            if name_match:
                customer = updates.get("customer_info") or state.get("customer_info") or CustomerInfo()
                customer.name = name_match.group(1)
                updates["customer_info"] = customer
                break

        # 提取救援类型
        rescue_types = ["拖车", "搭电", "换胎", "送油", "开锁", "现场维修", "困境救援"]
        for rt in rescue_types:
            if rt in text:
                updates["rescue_type"] = rt
                break

    return updates


# ==================== 路由函数 ====================

def route_by_stage(state: RescueState) -> str:
    """根据当前阶段路由"""
    stage = state.get("current_stage", WorkflowStage.GREETING)

    routing_map = {
        WorkflowStage.GREETING: "greeting",
        WorkflowStage.INTENT_RECOGNITION: "intent_recognition",
        WorkflowStage.COLLECT_INFO: "collect_info",
        WorkflowStage.CONFIRM_INFO: "confirm_info",
        WorkflowStage.CREATE_ORDER: "create_order",
        WorkflowStage.PENDING_PAYMENT: "pending_payment",
        WorkflowStage.CONFIRM_PAYMENT: "confirm_payment",
        WorkflowStage.DISPATCHING: "dispatched",  # 合并到 dispatched
        WorkflowStage.DISPATCHED: "dispatched",
        WorkflowStage.RESCUE_IN_PROGRESS: "dispatched",
        WorkflowStage.COMPLETED: "end",
        WorkflowStage.CANCELLED: "end",
        WorkflowStage.CONSULTING: "consulting",
    }

    return routing_map.get(stage, "greeting")


# ==================== 构建状态机图 ====================

def build_strict_rescue_graph(model_name: str = "gpt-4o-mini"):
    """构建严格状态机版本的救援服务工作流"""

    llm = get_llm(model_name)

    # 创建节点包装函数
    def _intent_recognition(state: RescueState):
        return intent_recognition_node(state, llm)

    def _collect_info(state: RescueState):
        return collect_info_node(state, llm)

    def _confirm_info(state: RescueState):
        return confirm_info_node(state, llm)

    def _consulting(state: RescueState):
        return consulting_node(state, llm)

    # 创建状态图
    workflow = StateGraph(RescueState)

    # 添加节点
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("intent_recognition", _intent_recognition)
    workflow.add_node("collect_info", _collect_info)
    workflow.add_node("confirm_info", _confirm_info)
    workflow.add_node("create_order", create_order_node)
    workflow.add_node("pending_payment", pending_payment_node)
    workflow.add_node("confirm_payment", confirm_payment_node)
    workflow.add_node("dispatched", dispatched_node)
    workflow.add_node("consulting", _consulting)

    # 设置入口点
    workflow.set_entry_point("greeting")

    # 添加条件边 - 从 greeting 出发
    workflow.add_edge("greeting", END)  # greeting 后等待用户输入

    # 条件路由 - 这是关键！根据状态决定下一步
    def route_after_input(state: RescueState) -> str:
        """用户输入后的路由"""
        return route_by_stage(state)

    # 从各节点出发的边
    workflow.add_edge("intent_recognition", END)
    workflow.add_edge("collect_info", END)
    workflow.add_edge("confirm_info", END)
    workflow.add_edge("create_order", END)
    workflow.add_edge("pending_payment", END)
    workflow.add_edge("confirm_payment", END)
    workflow.add_edge("dispatched", END)
    workflow.add_edge("consulting", END)

    # 编译
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


class StrictRescueChatbot:
    """严格状态机版本的救援聊天机器人"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.llm = get_llm(model_name)
        self.thread_id = "strict-rescue-session"
        self.state = self._get_initial_state()

    def _get_initial_state(self) -> RescueState:
        return {
            "messages": [],
            "customer_info": None,
            "location_info": None,
            "vehicle_info": None,
            "order_info": None,
            "rescue_type": None,
            "current_stage": WorkflowStage.GREETING,
            "ui_cards": [],
            "need_human_interrupt": False,
            "context": {}
        }

    def _get_node_handler(self, stage: str):
        """根据阶段获取处理函数"""
        handlers = {
            WorkflowStage.GREETING: lambda s: greeting_node(s),
            WorkflowStage.INTENT_RECOGNITION: lambda s: intent_recognition_node(s, self.llm),
            WorkflowStage.COLLECT_INFO: lambda s: collect_info_node(s, self.llm),
            WorkflowStage.CONFIRM_INFO: lambda s: confirm_info_node(s, self.llm),
            WorkflowStage.CREATE_ORDER: lambda s: create_order_node(s),
            WorkflowStage.PENDING_PAYMENT: lambda s: pending_payment_node(s),
            WorkflowStage.CONFIRM_PAYMENT: lambda s: confirm_payment_node(s),
            WorkflowStage.DISPATCHED: lambda s: dispatched_node(s),
            WorkflowStage.CONSULTING: lambda s: consulting_node(s, self.llm),
        }
        return handlers.get(stage, handlers[WorkflowStage.GREETING])

    def chat(self, user_input: str) -> tuple[str, list[UICard]]:
        """处理用户输入"""

        # 添加用户消息
        self.state["messages"].append(HumanMessage(content=user_input))

        # 提取信息更新状态
        extracted = extract_info_from_messages(self.state)
        for key, value in extracted.items():
            self.state[key] = value

        # 获取当前阶段的处理函数
        current_stage = self.state.get("current_stage", WorkflowStage.GREETING)
        handler = self._get_node_handler(current_stage)

        # 执行处理
        previous_cards_count = len(self.state.get("ui_cards", []))
        updates = handler(self.state)

        # 更新状态
        for key, value in updates.items():
            if key == "messages":
                self.state["messages"].extend(value)
            else:
                self.state[key] = value

        # 获取 AI 回复
        ai_response = ""
        for msg in reversed(self.state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                ai_response = msg.content
                break

        # 获取新卡片
        new_cards = self.state.get("ui_cards", [])[previous_cards_count:]

        return ai_response, new_cards

    def get_current_stage(self) -> str:
        return self.state.get("current_stage", WorkflowStage.GREETING)

    def reset(self):
        self.state = self._get_initial_state()
