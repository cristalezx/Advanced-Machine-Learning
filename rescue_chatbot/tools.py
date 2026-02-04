"""
LangGraph 工具定义 - 将 API 包装成 Agent 可调用的工具
"""

from typing import Optional
from langchain_core.tools import tool

from . import mock_api


@tool
def initiate_rescue_order(
    rescue_type: str,
    customer_name: str,
    customer_phone: str,
    address: str,
    plate_number: str = "",
    latitude: Optional[float] = None,
    longitude: Optional[float] = None
) -> dict:
    """
    发起救援订单。

    在收集完客户信息后调用此工具创建救援订单，返回订单价格供客户确认。

    Args:
        rescue_type: 救援类型，可选值：拖车、搭电、换胎、送油、开锁、现场维修、困境救援
        customer_name: 客户姓名
        customer_phone: 客户联系电话
        address: 救援地址
        plate_number: 车牌号（可选）
        latitude: 纬度（可选）
        longitude: 经度（可选）

    Returns:
        包含订单ID、价格和消息的字典
    """
    response = mock_api.initiate_rescue(
        rescue_type=rescue_type,
        customer_name=customer_name,
        customer_phone=customer_phone,
        address=address,
        plate_number=plate_number,
        latitude=latitude,
        longitude=longitude
    )
    return response.model_dump()


@tool
def confirm_order_payment(order_id: str, payment_confirmed: bool = True) -> dict:
    """
    确认订单支付。

    当客户确认支付后调用此工具，系统将分配救援人员并开始派单。

    Args:
        order_id: 订单ID
        payment_confirmed: 是否确认支付，默认为True

    Returns:
        包含订单状态、救援人员信息和预计到达时间的字典
    """
    response = mock_api.confirm_order(
        order_id=order_id,
        payment_confirmed=payment_confirmed
    )
    return response.model_dump()


@tool
def cancel_rescue_order(order_id: str, reason: str = "") -> dict:
    """
    取消救援订单。

    当客户要求取消订单时调用此工具。注意：已到达或正在救援的订单无法取消。

    Args:
        order_id: 订单ID
        reason: 取消原因（可选）

    Returns:
        包含取消结果和退款金额的字典
    """
    response = mock_api.cancel_order(
        order_id=order_id,
        reason=reason
    )
    return response.model_dump()


@tool
def query_order_status(order_id: str) -> dict:
    """
    查询订单状态。

    查询指定订单的当前状态，包括救援人员位置和预计到达时间。

    Args:
        order_id: 订单ID

    Returns:
        包含订单详细状态的字典
    """
    response = mock_api.query_order_status(order_id=order_id)
    return response.model_dump()


@tool
def get_available_rescue_types() -> dict:
    """
    获取可用的救援类型列表。

    返回所有支持的救援类型及其基础价格。

    Returns:
        救援类型和价格的字典
    """
    return mock_api.get_rescue_types()


# 所有工具列表
RESCUE_TOOLS = [
    initiate_rescue_order,
    confirm_order_payment,
    cancel_rescue_order,
    query_order_status,
    get_available_rescue_types,
]
