"""
可视化卡片定义 - 用于前端展示的卡片模板
"""

from typing import Optional
from .state import UICard, LocationInfo, VehicleInfo, CustomerInfo, OrderInfo


def create_info_collection_card(
    customer_info: Optional[CustomerInfo] = None,
    location_info: Optional[LocationInfo] = None,
    vehicle_info: Optional[VehicleInfo] = None,
    rescue_type: Optional[str] = None,
    missing_fields: list[str] = None
) -> UICard:
    """创建信息收集卡片"""
    content = {
        "rescue_type": rescue_type or "未选择",
        "customer": {
            "name": customer_info.name if customer_info else "",
            "phone": customer_info.phone if customer_info else "",
        } if customer_info else {},
        "location": {
            "address": location_info.address if location_info else "",
            "landmark": location_info.landmark if location_info else "",
            "coordinates": f"{location_info.latitude}, {location_info.longitude}"
                if location_info and location_info.latitude else ""
        } if location_info else {},
        "vehicle": {
            "plate_number": vehicle_info.plate_number if vehicle_info else "",
            "brand": vehicle_info.brand if vehicle_info else "",
            "model": vehicle_info.model if vehicle_info else "",
            "color": vehicle_info.color if vehicle_info else "",
        } if vehicle_info else {},
        "missing_fields": missing_fields or []
    }

    return UICard(
        card_type="info_collection",
        title="📋 救援信息收集",
        content=content,
        actions=["提交信息", "取消救援"]
    )


def create_payment_pending_card(order_info: OrderInfo) -> UICard:
    """创建待支付卡片"""
    content = {
        "order_id": order_info.order_id,
        "rescue_type": order_info.rescue_type,
        "price": f"¥{order_info.price:.2f}",
        "status": "待支付",
        "payment_methods": ["微信支付", "支付宝", "银行卡"]
    }

    return UICard(
        card_type="payment_pending",
        title="💳 订单待支付",
        content=content,
        actions=["确认支付", "取消订单"]
    )


def create_order_success_card(order_info: OrderInfo) -> UICard:
    """创建下单成功卡片"""
    content = {
        "order_id": order_info.order_id,
        "rescue_type": order_info.rescue_type,
        "price": f"¥{order_info.price:.2f}",
        "status": "已下单",
        "rescue_worker": order_info.rescue_worker or "正在分配中...",
        "estimated_arrival": order_info.estimated_arrival or "计算中..."
    }

    return UICard(
        card_type="order_success",
        title="✅ 下单成功",
        content=content,
        actions=["查看订单详情", "联系救援人员", "取消订单"]
    )


def create_order_status_card(order_info: OrderInfo, location_info: Optional[LocationInfo] = None) -> UICard:
    """创建订单状态卡片"""

    status_map = {
        "pending_payment": "⏳ 待支付",
        "paid": "💰 已支付",
        "dispatching": "🔄 派单中",
        "dispatched": "🚗 已派单",
        "en_route": "🚙 救援途中",
        "arrived": "📍 已到达",
        "in_progress": "🔧 救援中",
        "completed": "✅ 已完成",
        "cancelled": "❌ 已取消"
    }

    content = {
        "order_id": order_info.order_id,
        "rescue_type": order_info.rescue_type,
        "price": f"¥{order_info.price:.2f}",
        "status": status_map.get(order_info.status, order_info.status),
        "rescue_worker": order_info.rescue_worker or "未分配",
        "estimated_arrival": order_info.estimated_arrival or "-",
        "location": location_info.address if location_info else "-"
    }

    # 根据状态设置可用操作
    actions = ["刷新状态"]
    if order_info.status in ["pending_payment", "paid", "dispatching", "dispatched"]:
        actions.append("取消订单")
    if order_info.status == "dispatched" or order_info.status == "en_route":
        actions.append("联系救援人员")
    if order_info.status == "completed":
        actions.append("评价服务")

    return UICard(
        card_type="order_status",
        title="📦 订单状态",
        content=content,
        actions=actions
    )


def create_rescue_complete_card(order_info: OrderInfo) -> UICard:
    """创建救援完成卡片"""
    content = {
        "order_id": order_info.order_id,
        "rescue_type": order_info.rescue_type,
        "price": f"¥{order_info.price:.2f}",
        "status": "✅ 救援完成",
        "rescue_worker": order_info.rescue_worker,
        "message": "感谢您使用我们的救援服务！"
    }

    return UICard(
        card_type="rescue_complete",
        title="🎉 救援完成",
        content=content,
        actions=["评价服务", "再次下单", "返回首页"]
    )


def render_card_to_text(card: UICard) -> str:
    """将卡片渲染为文本格式（用于CLI展示）"""
    lines = [
        "┌" + "─" * 50 + "┐",
        f"│ {card.title:^48} │",
        "├" + "─" * 50 + "┤",
    ]

    for key, value in card.content.items():
        if isinstance(value, dict):
            lines.append(f"│ {key}: │")
            for k, v in value.items():
                if v:  # 只显示非空值
                    lines.append(f"│   {k}: {v:<40} │")
        elif isinstance(value, list):
            if value:
                lines.append(f"│ {key}: {', '.join(value):<36} │")
        else:
            lines.append(f"│ {key}: {str(value):<40} │")

    if card.actions:
        lines.append("├" + "─" * 50 + "┤")
        lines.append(f"│ 可用操作: {' | '.join(card.actions):<36} │")

    lines.append("└" + "─" * 50 + "┘")

    return "\n".join(lines)
