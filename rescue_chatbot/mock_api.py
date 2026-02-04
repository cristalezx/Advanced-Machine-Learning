"""
模拟 API 服务 - 救援服务后端接口
"""

import random
import string
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel


class RescueOrderResponse(BaseModel):
    """发起救援响应"""
    success: bool
    order_id: str
    price: float
    message: str


class ConfirmOrderResponse(BaseModel):
    """确认下单响应"""
    success: bool
    order_id: str
    status: str
    rescue_worker: Optional[str]
    estimated_arrival: Optional[str]
    message: str


class CancelOrderResponse(BaseModel):
    """取消订单响应"""
    success: bool
    order_id: str
    refund_amount: float
    message: str


class OrderStatusResponse(BaseModel):
    """查询订单状态响应"""
    success: bool
    order_id: str
    status: str
    rescue_type: str
    price: float
    rescue_worker: Optional[str]
    estimated_arrival: Optional[str]
    current_location: Optional[str]
    message: str


# 模拟订单数据库
_order_db = {}

# 救援类型价格表
RESCUE_PRICES = {
    "拖车": 200.0,
    "搭电": 80.0,
    "换胎": 100.0,
    "送油": 150.0,
    "开锁": 120.0,
    "现场维修": 180.0,
    "困境救援": 250.0,
}

# 模拟救援人员
RESCUE_WORKERS = [
    "张师傅 (工号: R001)",
    "李师傅 (工号: R002)",
    "王师傅 (工号: R003)",
    "赵师傅 (工号: R004)",
    "刘师傅 (工号: R005)",
]


def _generate_order_id() -> str:
    """生成订单ID"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_suffix = ''.join(random.choices(string.digits, k=4))
    return f"RSC{timestamp}{random_suffix}"


def _calculate_price(rescue_type: str, distance_km: float = 5.0) -> float:
    """计算救援价格"""
    base_price = RESCUE_PRICES.get(rescue_type, 150.0)
    # 超过5公里按每公里10元收费
    extra_distance_fee = max(0, (distance_km - 5)) * 10
    return base_price + extra_distance_fee


def _estimate_arrival_time(distance_km: float = 5.0) -> str:
    """预估到达时间"""
    # 假设平均速度30km/h
    minutes = int((distance_km / 30) * 60) + random.randint(10, 20)
    arrival_time = datetime.now() + timedelta(minutes=minutes)
    return arrival_time.strftime("%H:%M")


def initiate_rescue(
    rescue_type: str,
    customer_name: str,
    customer_phone: str,
    address: str,
    plate_number: str = "",
    latitude: Optional[float] = None,
    longitude: Optional[float] = None
) -> RescueOrderResponse:
    """
    发起救援接口
    - 创建救援订单
    - 返回订单价格
    """
    order_id = _generate_order_id()

    # 模拟距离计算
    distance_km = random.uniform(3.0, 15.0)
    price = _calculate_price(rescue_type, distance_km)

    # 存储订单
    _order_db[order_id] = {
        "order_id": order_id,
        "rescue_type": rescue_type,
        "customer_name": customer_name,
        "customer_phone": customer_phone,
        "address": address,
        "plate_number": plate_number,
        "latitude": latitude,
        "longitude": longitude,
        "price": price,
        "distance_km": distance_km,
        "status": "pending_payment",
        "rescue_worker": None,
        "estimated_arrival": None,
        "created_at": datetime.now().isoformat()
    }

    return RescueOrderResponse(
        success=True,
        order_id=order_id,
        price=price,
        message=f"救援订单已创建，预估费用 ¥{price:.2f}，请确认支付"
    )


def confirm_order(order_id: str, payment_confirmed: bool = True) -> ConfirmOrderResponse:
    """
    确认下单接口
    - 确认支付
    - 分配救援人员
    - 开始派单
    """
    if order_id not in _order_db:
        return ConfirmOrderResponse(
            success=False,
            order_id=order_id,
            status="not_found",
            rescue_worker=None,
            estimated_arrival=None,
            message="订单不存在"
        )

    order = _order_db[order_id]

    if not payment_confirmed:
        return ConfirmOrderResponse(
            success=False,
            order_id=order_id,
            status=order["status"],
            rescue_worker=None,
            estimated_arrival=None,
            message="支付未确认"
        )

    # 分配救援人员
    rescue_worker = random.choice(RESCUE_WORKERS)
    estimated_arrival = _estimate_arrival_time(order["distance_km"])

    # 更新订单状态
    order["status"] = "dispatched"
    order["rescue_worker"] = rescue_worker
    order["estimated_arrival"] = estimated_arrival

    return ConfirmOrderResponse(
        success=True,
        order_id=order_id,
        status="dispatched",
        rescue_worker=rescue_worker,
        estimated_arrival=estimated_arrival,
        message=f"订单已确认，{rescue_worker} 正在赶往现场，预计 {estimated_arrival} 到达"
    )


def cancel_order(order_id: str, reason: str = "") -> CancelOrderResponse:
    """
    取消订单接口
    """
    if order_id not in _order_db:
        return CancelOrderResponse(
            success=False,
            order_id=order_id,
            refund_amount=0.0,
            message="订单不存在"
        )

    order = _order_db[order_id]

    # 检查订单状态是否可取消
    non_cancellable_statuses = ["arrived", "in_progress", "completed"]
    if order["status"] in non_cancellable_statuses:
        return CancelOrderResponse(
            success=False,
            order_id=order_id,
            refund_amount=0.0,
            message=f"订单状态为{order['status']}，无法取消"
        )

    # 计算退款金额
    refund_amount = order["price"] if order["status"] == "pending_payment" else order["price"] * 0.8

    # 更新订单状态
    order["status"] = "cancelled"
    order["cancel_reason"] = reason
    order["cancelled_at"] = datetime.now().isoformat()

    return CancelOrderResponse(
        success=True,
        order_id=order_id,
        refund_amount=refund_amount,
        message=f"订单已取消，退款金额 ¥{refund_amount:.2f}"
    )


def query_order_status(order_id: str) -> OrderStatusResponse:
    """
    查询订单状态接口
    """
    if order_id not in _order_db:
        return OrderStatusResponse(
            success=False,
            order_id=order_id,
            status="not_found",
            rescue_type="",
            price=0.0,
            rescue_worker=None,
            estimated_arrival=None,
            current_location=None,
            message="订单不存在"
        )

    order = _order_db[order_id]

    # 模拟救援人员位置（如果已派单）
    current_location = None
    if order["status"] in ["dispatched", "en_route"]:
        current_location = f"距离您约 {random.uniform(1.0, order['distance_km']):.1f} 公里"

    return OrderStatusResponse(
        success=True,
        order_id=order_id,
        status=order["status"],
        rescue_type=order["rescue_type"],
        price=order["price"],
        rescue_worker=order.get("rescue_worker"),
        estimated_arrival=order.get("estimated_arrival"),
        current_location=current_location,
        message="查询成功"
    )


def simulate_order_progress(order_id: str, new_status: str) -> bool:
    """
    模拟订单进度更新（用于测试）
    """
    if order_id not in _order_db:
        return False

    valid_statuses = [
        "pending_payment", "paid", "dispatching", "dispatched",
        "en_route", "arrived", "in_progress", "completed", "cancelled"
    ]

    if new_status not in valid_statuses:
        return False

    _order_db[order_id]["status"] = new_status
    return True


def get_rescue_types() -> dict:
    """获取支持的救援类型和价格"""
    return RESCUE_PRICES.copy()
