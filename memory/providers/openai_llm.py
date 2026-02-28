"""
OpenAI / 兼容 OpenAI 接口的 LLM 客户端。
base_url 可以指向任意 OpenAI 兼容的第三方服务（如 DeepSeek、Qwen、本地 vLLM 等）。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from openai import AsyncOpenAI

from ..base import BaseLLMClient

logger = logging.getLogger(__name__)

_EXTRACT_SYSTEM = """\
你是一个记忆提取助手。请从对话中提取关于用户的重要事实和信息，这些信息将来会用于个性化服务。
返回 JSON 格式：{"memories": [{"content": "...", "metadata": {...}}, ...]}

规则：
- 只提取对用户有持久意义的信息（偏好、目标、重要事件等）
- 不要重复已有记忆中存在的内容
- 每条 content 应是一句清晰的陈述句
- 如没有值得记录的信息，返回 {"memories": []}
"""

_PROFILE_SYSTEM = """\
从对话中提取客户画像信息，返回 JSON 格式，字段仅包含以下键（缺失的直接省略）：
name, email, phone, age, gender, location, occupation, preferences (dict), tags (list), custom_fields (dict)

例：{"name": "张三", "age": 28, "preferences": {"颜色": "蓝色"}}
如果没有任何信息，返回 {}
"""

_MERGE_SYSTEM = """\
将「已有记忆」和「新信息」合并为一条更完整、准确的记忆描述（一句话）。
直接返回合并后的文本，不要解释。
"""

_DECIDE_SYSTEM = """\
你是一个精准的记忆管理助手。你的任务是：分析新对话，决定如何更新用户的记忆库。

【已有记忆格式】
每条格式为：[id: <memory_id>] <内容>

【操作类型】
- ADD    : 对话中出现了记忆库中没有的新信息 → 新增一条记忆
- UPDATE : 对话中的信息与某条已有记忆相关（补充/修正/与旧内容冲突） → 用合并后的完整内容覆盖旧记忆
- DELETE : 某条已有记忆被对话明确否定或已完全过时 → 删除旧记忆
- NONE   : 所有信息已被现有记忆覆盖，无需任何操作

【返回格式】严格返回 JSON，不要包含任何解释文字：
{
  "operations": [
    {"action": "ADD",    "content": "完整的新记忆陈述句"},
    {"action": "UPDATE", "memory_id": "<id>", "content": "合并后的完整记忆内容"},
    {"action": "DELETE", "memory_id": "<id>", "reason": "删除原因"},
    {"action": "NONE"}
  ]
}

【规则】
1. content 必须是完整的陈述句，不能是补丁描述（如"增加了..."）
2. UPDATE 的 content 是合并后的最终版本，不是追加内容
3. 同一信息只产生一条操作，不要重复
4. 若无任何有价值的新信息，仅返回 {"operations": [{"action": "NONE"}]}
5. 只记录对用户有持久价值的信息（偏好、习惯、重要事实等），忽略临时闲聊
"""


class OpenAILLMClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )
        self.model = model

    async def extract_memories(
        self,
        messages: List[Dict[str, str]],
        existing_memories: List[str],
    ) -> List[Dict[str, Any]]:
        existing_text = (
            "\n".join(f"- {m}" for m in existing_memories)
            if existing_memories
            else "（暂无）"
        )
        user_content = (
            f"已有记忆：\n{existing_text}\n\n"
            f"对话内容：\n{json.dumps(messages, ensure_ascii=False)}\n\n"
            "请提取新记忆："
        )
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        try:
            result = json.loads(resp.choices[0].message.content)
            return result.get("memories", [])
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning("Failed to parse extract_memories response: %s", e)
            return []

    async def extract_profile(
        self,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _PROFILE_SYSTEM},
                {
                    "role": "user",
                    "content": f"对话：\n{json.dumps(messages, ensure_ascii=False)}",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        try:
            return json.loads(resp.choices[0].message.content) or {}
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning("Failed to parse extract_profile response: %s", e)
            return {}

    async def decide_memory_operations(
        self,
        messages: List[Dict[str, Any]],
        existing_memories: List[Dict[str, Any]],
        conversation_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        分析对话，输出 ADD / UPDATE / DELETE / NONE 操作列表。

        Args:
            messages:          对话消息列表，每条含 role/content，可附带 timestamp
            existing_memories: 已检索到的相关记忆，每条含 {"id": ..., "content": ...}
            conversation_time: 本次会话时间（ISO 字符串），给 LLM 时序参考
        """
        # ---- 构建已有记忆文本块 ----
        if existing_memories:
            mem_lines = "\n".join(
                f"[id: {m['id']}] {m['content']}" for m in existing_memories
            )
            existing_block = f"【已有记忆】\n{mem_lines}"
        else:
            existing_block = "【已有记忆】（暂无）"

        # ---- 构建对话文本块（含消息级时间戳） ----
        dialog_lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            ts = msg.get("timestamp")
            prefix = f"[{ts}] " if ts else ""
            dialog_lines.append(f"{prefix}{role}: {content}")
        dialog_block = "\n".join(dialog_lines)

        time_hint = f"（会话时间: {conversation_time}）" if conversation_time else ""

        user_content = (
            f"{existing_block}\n\n"
            f"【新对话】{time_hint}\n{dialog_block}\n\n"
            "请输出操作列表："
        )

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _DECIDE_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        try:
            result = json.loads(resp.choices[0].message.content)
            return result.get("operations", [])
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning("Failed to parse decide_memory_operations response: %s", e)
            return []

    async def merge_memory(self, existing: str, new_info: str) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _MERGE_SYSTEM},
                {
                    "role": "user",
                    "content": f"已有记忆：{existing}\n新信息：{new_info}",
                },
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
