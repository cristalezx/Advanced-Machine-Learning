"""
状态定义 - 救援服务工作流的状态模式
"""

from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class LocationInfo(BaseModel):
    """客户位置信息"""
    address: str = Field(default="", description="详细地址")
    latitude: Optional[float] = Field(default=None, description="纬度")
    longitude: Optional[float] = Field(default=None, description="经度")
    landmark: str = Field(default="", description="附近地标")


class VehicleInfo(BaseModel):
    """车辆信息"""
    plate_number: str = Field(default="", description="车牌号")
    brand: str = Field(default="", description="品牌")
    model: str = Field(default="", description="型号")
    color: str = Field(default="", description="颜色")


class CustomerInfo(BaseModel):
    """客户信息"""
    name: str = Field(default="", description="客户姓名")
    phone: str = Field(default="", description="联系电话")


class OrderInfo(BaseModel):
    """订单信息"""
    order_id: str = Field(default="", description="订单ID")
    price: float = Field(default=0.0, description="订单价格")
    rescue_type: str = Field(default="", description="救援类型")
    status: str = Field(default="", description="订单状态")
    rescue_worker: Optional[str] = Field(default=None, description="救援人员")
    estimated_arrival: Optional[str] = Field(default=None, description="预计到达时间")


class UICard(BaseModel):
    """可视化卡片"""
    card_type: Literal[
        "info_collection",      # 信息收集卡片
        "payment_pending",      # 订单待支付
        "order_success",        # 下单成功
        "order_status",         # 订单状态
        "rescue_complete"       # 救援完成
    ]
    title: str
    content: dict
    actions: list[str] = Field(default_factory=list)


class RescueState(TypedDict):
    """救援服务工作流状态"""
    # 消息历史
    messages: Annotated[list, add_messages]

    # 客户信息
    customer_info: Optional[CustomerInfo]

    # 位置信息
    location_info: Optional[LocationInfo]

    # 车辆信息
    vehicle_info: Optional[VehicleInfo]

    # 订单信息
    order_info: Optional[OrderInfo]

    # 救援类型
    rescue_type: Optional[str]  # 拖车、搭电、换胎、送油、开锁等

    # 当前阶段
    current_stage: Literal[
        "greeting",              # 问候/初始
        "confirm_rescue",        # 确认发起救援
        "collect_info",          # 收集信息
        "confirm_order",         # 确认下单
        "payment_pending",       # 等待支付
        "order_placed",          # 已下单
        "dispatched",            # 已派单
        "rescue_arrived",        # 救援到达
        "completed",             # 完成
        "cancelled",             # 已取消
        "consulting"             # 咨询中
    ]

    # UI卡片队列
    ui_cards: list[UICard]

    # 是否需要人工干预
    need_human_interrupt: bool

    # 上下文信息
    context: dict
