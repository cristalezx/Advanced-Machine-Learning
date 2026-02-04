"""
LangGraph 节点定义 - 工作流中的各个处理节点
"""

from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from .state import (
    RescueState, UICard, CustomerInfo, LocationInfo,
    VehicleInfo, OrderInfo
)
from .cards import (
    create_info_collection_card,
    create_payment_pending_card,
    create_order_success_card,
    create_order_status_card,
    create_rescue_complete_card,
    render_card_to_text
)
from .tools import RESCUE_TOOLS


def create_tool_node():
    """创建工具执行节点"""
    return ToolNode(RESCUE_TOOLS)


def should_use_tools(state: RescueState) -> Literal["tools", "process_response"]:
    """判断是否需要调用工具"""
    messages = state.get("messages", [])
    if not messages:
        return "process_response"

    last_message = messages[-1]

    # 如果最后一条消息是AI消息且有工具调用
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return "process_response"


def process_tool_response(state: RescueState) -> dict:
    """处理工具调用结果，更新状态和生成卡片"""
    messages = state.get("messages", [])
    ui_cards = state.get("ui_cards", [])
    order_info = state.get("order_info")
    current_stage = state.get("current_stage", "greeting")

    updates = {}

    # 查找最近的工具消息
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            tool_name = msg.name
            try:
                # 解析工具返回结果
                import json
                if isinstance(msg.content, str):
                    result = json.loads(msg.content)
                else:
                    result = msg.content

                # 根据工具类型处理结果
                if tool_name == "initiate_rescue_order" and result.get("success"):
                    # 创建订单成功，生成待支付卡片
                    new_order = OrderInfo(
                        order_id=result["order_id"],
                        price=result["price"],
                        rescue_type=state.get("rescue_type", ""),
                        status="pending_payment"
                    )
                    updates["order_info"] = new_order
                    updates["current_stage"] = "payment_pending"

                    # 添加待支付卡片
                    payment_card = create_payment_pending_card(new_order)
                    ui_cards.append(payment_card)
                    updates["ui_cards"] = ui_cards

                elif tool_name == "confirm_order_payment" and result.get("success"):
                    # 支付确认成功，更新订单状态
                    if order_info:
                        order_info.status = result["status"]
                        order_info.rescue_worker = result.get("rescue_worker")
                        order_info.estimated_arrival = result.get("estimated_arrival")
                        updates["order_info"] = order_info
                    updates["current_stage"] = "dispatched"

                    # 添加下单成功卡片
                    if order_info:
                        success_card = create_order_success_card(order_info)
                        ui_cards.append(success_card)
                        updates["ui_cards"] = ui_cards

                elif tool_name == "cancel_rescue_order" and result.get("success"):
                    # 取消订单成功
                    updates["current_stage"] = "cancelled"
                    if order_info:
                        order_info.status = "cancelled"
                        updates["order_info"] = order_info

                elif tool_name == "query_order_status" and result.get("success"):
                    # 查询订单状态成功
                    if order_info:
                        order_info.status = result["status"]
                        order_info.rescue_worker = result.get("rescue_worker")
                        order_info.estimated_arrival = result.get("estimated_arrival")
                        updates["order_info"] = order_info

                        # 添加订单状态卡片
                        status_card = create_order_status_card(
                            order_info,
                            state.get("location_info")
                        )
                        ui_cards.append(status_card)
                        updates["ui_cards"] = ui_cards

                        # 如果订单完成，更新阶段
                        if result["status"] == "completed":
                            updates["current_stage"] = "completed"
                            complete_card = create_rescue_complete_card(order_info)
                            ui_cards.append(complete_card)

            except (json.JSONDecodeError, KeyError, TypeError):
                pass

            break

    return updates


def determine_next_action(state: RescueState) -> Literal[
    "collect_info",
    "confirm_order",
    "process_payment",
    "check_status",
    "handle_consultation",
    "end"
]:
    """确定下一步操作"""
    current_stage = state.get("current_stage", "greeting")
    messages = state.get("messages", [])

    # 根据当前阶段决定下一步
    stage_mapping = {
        "greeting": "collect_info",
        "confirm_rescue": "collect_info",
        "collect_info": "confirm_order",
        "confirm_order": "process_payment",
        "payment_pending": "process_payment",
        "order_placed": "check_status",
        "dispatched": "check_status",
        "rescue_arrived": "check_status",
        "completed": "end",
        "cancelled": "end",
        "consulting": "handle_consultation"
    }

    return stage_mapping.get(current_stage, "handle_consultation")


def create_info_collection_node(state: RescueState) -> dict:
    """信息收集节点 - 检查缺失信息并生成收集卡片"""
    customer_info = state.get("customer_info")
    location_info = state.get("location_info")
    vehicle_info = state.get("vehicle_info")
    rescue_type = state.get("rescue_type")
    ui_cards = state.get("ui_cards", [])

    missing_fields = []

    # 检查必要信息
    if not rescue_type:
        missing_fields.append("救援类型")

    if not customer_info or not customer_info.name:
        missing_fields.append("客户姓名")
    if not customer_info or not customer_info.phone:
        missing_fields.append("联系电话")

    if not location_info or not location_info.address:
        missing_fields.append("救援地址")

    # 如果有缺失信息，生成信息收集卡片
    if missing_fields:
        collection_card = create_info_collection_card(
            customer_info=customer_info,
            location_info=location_info,
            vehicle_info=vehicle_info,
            rescue_type=rescue_type,
            missing_fields=missing_fields
        )
        ui_cards.append(collection_card)

        return {
            "ui_cards": ui_cards,
            "current_stage": "collect_info",
            "need_human_interrupt": True
        }

    # 信息收集完成
    return {
        "current_stage": "confirm_order",
        "need_human_interrupt": False
    }


def extract_info_from_message(message: str) -> dict:
    """从用户消息中提取信息（简单的关键词匹配）"""
    extracted = {}

    # 救援类型识别
    rescue_keywords = {
        "拖车": ["拖车", "拖走", "运走", "车坏了动不了"],
        "搭电": ["搭电", "电瓶", "没电", "打不着火", "启动不了"],
        "换胎": ["换胎", "爆胎", "轮胎", "胎破了", "漏气"],
        "送油": ["送油", "没油", "油用完了", "加油"],
        "开锁": ["开锁", "钥匙锁车里", "锁车里了", "打不开门"],
        "现场维修": ["维修", "修车", "故障"],
        "困境救援": ["陷车", "泥地", "困住", "出不来"]
    }

    for rescue_type, keywords in rescue_keywords.items():
        if any(kw in message for kw in keywords):
            extracted["rescue_type"] = rescue_type
            break

    # 电话号码识别（简单的11位数字）
    import re
    phone_match = re.search(r'1[3-9]\d{9}', message)
    if phone_match:
        extracted["phone"] = phone_match.group()

    # 车牌号识别
    plate_match = re.search(r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][A-Z0-9]{5}', message)
    if plate_match:
        extracted["plate_number"] = plate_match.group()

    return extracted
