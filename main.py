#!/usr/bin/env python3
"""
救援服务聊天机器人 - 主程序入口

基于 LangGraph 的 Agentic Workflow 实现
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def print_banner():
    """打印启动横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║        🚗  道路救援服务智能助手  🚗                          ║
║                                                              ║
║        基于 LangGraph 的 Agentic Workflow                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_help():
    """打印帮助信息"""
    help_text = """
命令帮助：
  /help     - 显示此帮助信息
  /reset    - 重置会话
  /status   - 显示当前状态
  /demo     - 运行演示流程
  /quit     - 退出程序

示例对话：
  - "我需要拖车服务"
  - "我的车打不着火了"
  - "查询订单状态"
  - "取消订单"
    """
    print(help_text)


def run_demo():
    """运行演示流程"""
    print("\n" + "=" * 60)
    print("🎬 开始演示流程")
    print("=" * 60)

    from rescue_chatbot.state import CustomerInfo, LocationInfo, VehicleInfo, OrderInfo
    from rescue_chatbot.cards import (
        create_info_collection_card,
        create_payment_pending_card,
        create_order_success_card,
        create_order_status_card,
        create_rescue_complete_card,
        render_card_to_text
    )
    from rescue_chatbot.mock_api import (
        initiate_rescue,
        confirm_order,
        query_order_status,
        simulate_order_progress
    )

    # 模拟用户信息
    customer = CustomerInfo(name="张三", phone="13800138000")
    location = LocationInfo(
        address="北京市朝阳区望京SOHO T1",
        latitude=39.9876,
        longitude=116.4567,
        landmark="望京地铁站附近"
    )
    vehicle = VehicleInfo(
        plate_number="京A12345",
        brand="大众",
        model="帕萨特",
        color="黑色"
    )

    # 步骤1：信息收集卡片
    print("\n📋 步骤1：信息收集")
    print("-" * 40)
    info_card = create_info_collection_card(
        customer_info=customer,
        location_info=location,
        vehicle_info=vehicle,
        rescue_type="搭电",
        missing_fields=[]
    )
    print(render_card_to_text(info_card))

    # 步骤2：发起救援
    print("\n🚀 步骤2：发起救援订单")
    print("-" * 40)
    response = initiate_rescue(
        rescue_type="搭电",
        customer_name=customer.name,
        customer_phone=customer.phone,
        address=location.address,
        plate_number=vehicle.plate_number,
        latitude=location.latitude,
        longitude=location.longitude
    )
    print(f"API响应: {response.message}")
    print(f"订单ID: {response.order_id}")
    print(f"价格: ¥{response.price:.2f}")

    # 步骤3：待支付卡片
    print("\n💳 步骤3：订单待支付")
    print("-" * 40)
    order = OrderInfo(
        order_id=response.order_id,
        price=response.price,
        rescue_type="搭电",
        status="pending_payment"
    )
    payment_card = create_payment_pending_card(order)
    print(render_card_to_text(payment_card))

    # 步骤4：确认支付
    print("\n✅ 步骤4：确认支付并派单")
    print("-" * 40)
    confirm_response = confirm_order(response.order_id, payment_confirmed=True)
    print(f"API响应: {confirm_response.message}")

    # 步骤5：下单成功卡片
    print("\n🎉 步骤5：下单成功")
    print("-" * 40)
    order.status = confirm_response.status
    order.rescue_worker = confirm_response.rescue_worker
    order.estimated_arrival = confirm_response.estimated_arrival
    success_card = create_order_success_card(order)
    print(render_card_to_text(success_card))

    # 步骤6：查询订单状态
    print("\n📦 步骤6：查询订单状态")
    print("-" * 40)
    status_response = query_order_status(response.order_id)
    print(f"API响应: 订单状态 - {status_response.status}")

    status_card = create_order_status_card(order, location)
    print(render_card_to_text(status_card))

    # 步骤7：模拟救援完成
    print("\n🏁 步骤7：救援完成")
    print("-" * 40)
    simulate_order_progress(response.order_id, "completed")
    order.status = "completed"
    complete_card = create_rescue_complete_card(order)
    print(render_card_to_text(complete_card))

    print("\n" + "=" * 60)
    print("🎬 演示流程结束")
    print("=" * 60)


def run_interactive(model_name: str = "gpt-4o-mini"):
    """运行交互式聊天"""
    try:
        from rescue_chatbot.graph import RescueChatbot
        from rescue_chatbot.cards import render_card_to_text

        print(f"\n正在初始化聊天机器人 (模型: {model_name})...")
        chatbot = RescueChatbot(model_name=model_name)
        state = chatbot.get_initial_state()

        print("\n✅ 初始化完成！开始对话吧~")
        print("(输入 /help 查看命令帮助)\n")

        # 发送初始问候
        response, state, cards = chatbot.chat("你好")
        print(f"\n🤖 助手: {response}\n")

        for card in cards:
            print(render_card_to_text(card))
            print()

        while True:
            try:
                user_input = input("👤 您: ").strip()

                if not user_input:
                    continue

                # 处理命令
                if user_input.startswith("/"):
                    cmd = user_input.lower()
                    if cmd == "/quit" or cmd == "/exit":
                        print("\n感谢使用，再见！👋")
                        break
                    elif cmd == "/help":
                        print_help()
                        continue
                    elif cmd == "/reset":
                        state = chatbot.get_initial_state()
                        chatbot.reset()
                        print("\n🔄 会话已重置\n")
                        continue
                    elif cmd == "/status":
                        print(f"\n当前状态: {state.get('current_stage', 'unknown')}")
                        print(f"救援类型: {state.get('rescue_type', '未选择')}")
                        if state.get('order_info'):
                            print(f"订单ID: {state['order_info'].order_id}")
                        print()
                        continue
                    elif cmd == "/demo":
                        run_demo()
                        continue

                # 处理对话
                response, state, cards = chatbot.chat(user_input, state)

                print(f"\n🤖 助手: {response}\n")

                # 显示卡片
                for card in cards:
                    print(render_card_to_text(card))
                    print()

            except KeyboardInterrupt:
                print("\n\n感谢使用，再见！👋")
                break

    except ImportError as e:
        print(f"\n⚠️ 导入错误: {e}")
        print("请确保已安装所有依赖: pip install -r requirements.txt")
        print("\n您仍可以运行演示模式: python main.py --demo")
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        print("请检查 API 密钥配置是否正确")
        print("\n您仍可以运行演示模式: python main.py --demo")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="道路救援服务智能助手",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="运行演示流程（不需要 API 密钥）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="使用的模型 (默认: gpt-4o-mini)"
    )

    args = parser.parse_args()

    print_banner()

    if args.demo:
        run_demo()
    else:
        # 检查 API 密钥
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            print("\n⚠️ 未检测到 API 密钥")
            print("请设置 OPENAI_API_KEY 或 ANTHROPIC_API_KEY 环境变量")
            print("\n或者运行演示模式: python main.py --demo")
            print("\n是否运行演示模式？[Y/n]: ", end="")
            choice = input().strip().lower()
            if choice != "n":
                run_demo()
            return

        run_interactive(args.model)


if __name__ == "__main__":
    main()
