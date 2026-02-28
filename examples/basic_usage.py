"""
基础用法示例：不依赖任何外部服务，使用内存向量库。
运行：  python -m examples.basic_usage
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from memory import (
    ChannelType,
    MemoryQuery,
    MemoryType,
    create_openai_memory_manager,
)


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "sk-YOUR-KEY")

    # ① 创建 MemoryManager（使用内存向量库，适合测试）
    manager = create_openai_memory_manager(api_key=api_key)

    user_id = "customer_001"

    # ② 从对话中提取记忆（mem0 风格）
    conversation = [
        {"role": "user", "content": "你好，我叫李明，住在上海，我对智能家居产品很感兴趣。"},
        {"role": "assistant", "content": "您好李明！很高兴认识您，请问有什么可以帮您的？"},
        {"role": "user", "content": "我最近想买一款智能音箱，预算大概 500 元左右，喜欢白色外观。"},
        {"role": "assistant", "content": "好的，我们有几款符合您预算和偏好的产品，稍后为您推荐。"},
    ]
    memories = await manager.add_from_conversation(
        user_id=user_id,
        messages=conversation,
        extract_profile=True,
    )
    print(f"\n[对话提取] 保存了 {len(memories)} 条记忆：")
    for m in memories:
        print(f"  - [{m.memory_type.value}] {m.content}")

    # ③ 直接插入一条记忆
    from memory import MemoryItem

    item = await manager.insert(
        MemoryItem(
            user_id=user_id,
            memory_type=MemoryType.CONVERSATIONAL,
            content="用户曾购买过我们的智能门锁产品，评价为好评。",
            metadata={"order_id": "ORD20240101"},
        )
    )
    print(f"\n[直接插入] ID={item.id}")

    # ④ 写入外部渠道数据（CRM）
    crm_item = await manager.add_external_channel_data(
        user_id=user_id,
        data={
            "crm_id": "CRM_88888",
            "vip_level": "gold",
            "total_purchase": 3200,
            "last_contact": "2024-12-01",
        },
        channel=ChannelType.CRM,
        content_text="用户为 CRM 黄金会员，累计消费 3200 元，上次联系日期 2024-12-01",
    )
    print(f"\n[CRM渠道] ID={crm_item.id}")

    # ⑤ 更新客户画像
    profile_item = await manager.update_customer_profile(
        user_id=user_id,
        profile_updates={
            "name": "李明",
            "location": "上海",
            "preferences": {"category": "智能家居", "color": "白色"},
            "tags": ["vip", "smart_home_fan"],
        },
        channel=ChannelType.CRM,
    )
    print(f"\n[客户画像更新] {profile_item.content}")

    # ⑥ 获取画像
    profile = await manager.get_customer_profile(user_id)
    print(f"\n[客户画像] name={profile.name}, location={profile.location}, tags={profile.tags}")

    # ⑦ 语义检索
    results = await manager.query(
        MemoryQuery(
            user_id=user_id,
            query="用户对哪类产品感兴趣",
            top_k=3,
        )
    )
    print(f"\n[语义检索] 查询「用户对哪类产品感兴趣」，Top-{len(results)} 结果：")
    for r in results:
        print(f"  score={r.score:.3f}  [{r.item.memory_type.value}] {r.item.content}")

    # ⑧ 按类型检索
    profile_results = await manager.query(
        MemoryQuery(
            user_id=user_id,
            query="用户基本信息",
            memory_types=[MemoryType.CUSTOMER_PROFILE],
            top_k=1,
        )
    )
    print(f"\n[画像检索] {profile_results[0].item.content if profile_results else '无'}")

    # ⑨ 更新一条记忆
    updated = await manager.update(
        item.id,
        {"content": "用户曾购买过智能门锁产品，并给出好评，推荐过给朋友。"},
    )
    print(f"\n[更新记忆] {updated.content}")

    # ⑩ 列出所有记忆
    all_memories = await manager.list_all(user_id)
    print(f"\n[全部记忆] 共 {len(all_memories)} 条：")
    for m in all_memories:
        print(f"  [{m.memory_type.value}] {m.content[:60]}...")


if __name__ == "__main__":
    asyncio.run(main())
