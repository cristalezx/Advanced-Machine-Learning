"""
混合工作流：状态机骨架 + LLM 血肉

设计原则：
1. 状态机只控制"关键节点"的转换
2. 每个阶段内部，LLM 可以自由对话
3. 支持跑题、闲聊、咨询等"软性"交互
4. 关键操作（下单、支付）必须显式确认
"""

from typing import Optional, Literal
from enum import Enum
from dataclasses import dataclass
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .state import RescueState, CustomerInfo, LocationInfo, VehicleInfo, OrderInfo
from .cards import create_info_collection_card, create_payment_pending_card, create_order_success_card


class Stage(str, Enum):
    """关键阶段（状态机控制）"""
    GREETING = "greeting"
    COLLECTING = "collecting"      # 收集信息阶段
    CONFIRMING = "confirming"      # 确认信息阶段
    PAYMENT = "payment"            # 支付阶段
    TRACKING = "tracking"          # 跟踪阶段
    DONE = "done"


@dataclass
class StageTransitionCondition:
    """阶段转换条件"""
    stage: Stage
    required_fields: list[str]
    trigger_intents: list[str]


# 只定义关键转换点
STAGE_TRANSITIONS = {
    Stage.GREETING: StageTransitionCondition(
        stage=Stage.COLLECTING,
        required_fields=[],
        trigger_intents=["need_rescue", "request_service"]
    ),
    Stage.COLLECTING: StageTransitionCondition(
        stage=Stage.CONFIRMING,
        required_fields=["rescue_type", "phone", "address"],  # 必须收集完
        trigger_intents=[]  # 自动触发，无需特定意图
    ),
    Stage.CONFIRMING: StageTransitionCondition(
        stage=Stage.PAYMENT,
        required_fields=["order_id"],
        trigger_intents=["confirm", "yes", "ok"]  # 必须明确确认
    ),
    Stage.PAYMENT: StageTransitionCondition(
        stage=Stage.TRACKING,
        required_fields=["payment_confirmed"],
        trigger_intents=["paid", "confirm_payment"]  # 必须明确确认支付
    ),
}


# ==================== 阶段内 Prompt（允许 LLM 自由发挥）====================

STAGE_PROMPTS = {
    Stage.GREETING: """你是道路救援服务助手，正在问候用户。

职责：
- 热情友好地问候
- 简单介绍服务内容
- 引导用户说出需求

风格：
- 亲切自然，像朋友聊天
- 可以适当使用表情
- 如果用户闲聊，可以简单回应再引导回主题

【不要】直接问"请问您需要什么服务"这种死板的话
【要】自然地聊天，比如"您好呀！是车子遇到什么问题了吗？"
""",

    Stage.COLLECTING: """你正在帮用户收集救援信息。

当前已收集：
- 救援类型：{rescue_type}
- 联系电话：{phone}
- 地址：{address}
- 车牌：{plate}

还需要：{missing}

【核心任务】收集缺失信息
【交互风格】
- 不要像表单一样一个个问，要自然对话
- 用户说的信息可能不完整，需要追问细节
- 用户可能跑题或问问题，简单回答后拉回来
- 用户着急时要先安抚情绪

【示例对话】
用户：我车打不着火了
助手：别着急，这种情况很常见，一般搭个电就能解决。您现在在什么位置呢？我帮您安排师傅过去。

用户：在望京
助手：好的望京。能告诉我更具体的位置吗？比如在哪条路上、附近有什么标志性建筑？这样师傅能更快找到您。

用户：SOHO 附近，对了你们大概多少钱啊？
助手：搭电服务基础费用是 80 元，超过 5 公里会有一点里程费。您在望京 SOHO 这边对吧？方便留个电话吗？师傅到了好联系您。

【可以做】
- 回答价格、时间等咨询问题
- 安抚用户情绪
- 解释服务内容
- 闲聊几句

【不要做】
- 跳过信息收集直接下单
- 忽视用户的问题
- 机械式地一问一答
""",

    Stage.CONFIRMING: """信息收集完毕，需要用户确认。

救援信息：
- 类型：{rescue_type}
- 联系人：{name} {phone}
- 地址：{address}
- 车牌：{plate}
- 预估费用：{price}

【任务】让用户确认信息是否正确

【风格】
- 清晰列出信息让用户核对
- 如果用户说要改，问清楚改哪里
- 如果用户有疑问，耐心解答

【关键】只有用户明确说"确认"、"没问题"、"可以"等肯定词时才能进入下一步
用户说"嗯"、"好"可能只是在听，要追问"那我帮您下单了哈？"
""",

    Stage.PAYMENT: """订单已创建，等待用户确认支付。

订单号：{order_id}
费用：¥{price}

【任务】引导用户完成支付

【风格】
- 清晰告知费用
- 解答费用相关问题
- 如果用户犹豫，可以解释费用构成

【关键】必须等用户明确说"支付"、"付款"、"确认"才能进入下一步
不要自己假设用户已支付
""",

    Stage.TRACKING: """订单已派单，救援人员正在路上。

救援人员：{worker}
预计到达：{eta}

【任务】
- 告知派单成功
- 回答用户关于等待时间、进度的问题
- 安抚用户情绪

【可以做】
- 主动告知可以查看进度
- 提醒保持电话畅通
- 回答"还要多久"等问题
""",
}


class HybridWorkflow:
    """混合工作流：状态机 + LLM"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)  # 保持一定创造性
        self.stage = Stage.GREETING
        self.context = {
            "rescue_type": None,
            "name": None,
            "phone": None,
            "address": None,
            "plate": None,
            "order_id": None,
            "price": None,
            "payment_confirmed": False,
            "worker": None,
            "eta": None,
        }
        self.messages = []

    def _get_stage_prompt(self) -> str:
        """获取当前阶段的 prompt"""
        template = STAGE_PROMPTS.get(self.stage, "")
        missing = self._get_missing_fields()
        return template.format(
            rescue_type=self.context.get("rescue_type") or "未知",
            name=self.context.get("name") or "未知",
            phone=self.context.get("phone") or "未提供",
            address=self.context.get("address") or "未提供",
            plate=self.context.get("plate") or "未提供",
            price=self.context.get("price") or "待计算",
            order_id=self.context.get("order_id") or "",
            worker=self.context.get("worker") or "",
            eta=self.context.get("eta") or "",
            missing="、".join(missing) if missing else "无"
        )

    def _get_missing_fields(self) -> list[str]:
        """获取缺失字段"""
        required = {
            Stage.COLLECTING: ["rescue_type", "phone", "address"],
            Stage.CONFIRMING: [],
            Stage.PAYMENT: [],
        }
        fields = required.get(self.stage, [])
        return [f for f in fields if not self.context.get(f)]

    def _extract_info(self, text: str):
        """从用户输入提取信息（LLM 辅助）"""
        import re

        # 电话
        phone = re.search(r'1[3-9]\d{9}', text)
        if phone:
            self.context["phone"] = phone.group()

        # 车牌
        plate = re.search(r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z][A-Z0-9]{5,6}', text)
        if plate:
            self.context["plate"] = plate.group()

        # 救援类型
        type_keywords = {
            "拖车": ["拖车", "拖走", "运走"],
            "搭电": ["搭电", "没电", "打不着", "电瓶", "启动不了"],
            "换胎": ["换胎", "爆胎", "轮胎", "漏气"],
            "送油": ["送油", "没油", "加油"],
            "开锁": ["开锁", "钥匙", "锁车里"],
        }
        for rtype, keywords in type_keywords.items():
            if any(kw in text for kw in keywords):
                self.context["rescue_type"] = rtype
                break

        # 地址（简单匹配）
        addr_patterns = [
            r'在(.{2,30}(?:路|街|区|大厦|广场|小区|SOHO|soho|商场|医院|学校|站|门口|附近))',
            r'位置[是在：:]*(.{5,40})',
        ]
        for pattern in addr_patterns:
            match = re.search(pattern, text)
            if match:
                self.context["address"] = match.group(1).strip()
                break

    def _check_stage_transition(self, user_intent: str) -> bool:
        """检查是否满足阶段转换条件"""
        condition = STAGE_TRANSITIONS.get(self.stage)
        if not condition:
            return False

        # 检查必要字段
        for field in condition.required_fields:
            if not self.context.get(field):
                return False

        # 检查触发意图（如果有要求）
        if condition.trigger_intents:
            if user_intent not in condition.trigger_intents:
                return False

        return True

    def _detect_intent(self, text: str) -> str:
        """检测用户意图（简化版）"""
        text = text.lower()

        # 确认类
        if any(w in text for w in ["确认", "没问题", "可以", "好的", "对", "是的", "下单", "ok", "行"]):
            return "confirm"

        # 否定类
        if any(w in text for w in ["不对", "不是", "改", "修改", "错了", "重新"]):
            return "reject"

        # 取消类
        if any(w in text for w in ["取消", "不要了", "算了"]):
            return "cancel"

        # 支付类
        if any(w in text for w in ["支付", "付款", "付了", "已支付", "微信", "支付宝"]):
            return "paid"

        # 救援请求
        if any(w in text for w in ["救援", "帮忙", "拖车", "搭电", "换胎", "送油", "开锁", "没电", "爆胎"]):
            return "need_rescue"

        return "other"

    def _do_stage_transition(self):
        """执行阶段转换"""
        transitions = {
            Stage.GREETING: Stage.COLLECTING,
            Stage.COLLECTING: Stage.CONFIRMING,
            Stage.CONFIRMING: Stage.PAYMENT,
            Stage.PAYMENT: Stage.TRACKING,
            Stage.TRACKING: Stage.DONE,
        }
        next_stage = transitions.get(self.stage)
        if next_stage:
            print(f"  [状态转换] {self.stage.value} → {next_stage.value}")
            self.stage = next_stage

            # 阶段转换时的特殊处理
            if next_stage == Stage.CONFIRMING:
                # 计算价格
                self.context["price"] = self._calculate_price()

            elif next_stage == Stage.PAYMENT:
                # 创建订单
                self._create_order()

            elif next_stage == Stage.TRACKING:
                # 确认支付并派单
                self._dispatch_order()

    def _calculate_price(self) -> float:
        """计算价格"""
        prices = {"拖车": 200, "搭电": 80, "换胎": 100, "送油": 150, "开锁": 120}
        return prices.get(self.context.get("rescue_type", ""), 150)

    def _create_order(self):
        """创建订单"""
        import random, string
        self.context["order_id"] = "RSC" + "".join(random.choices(string.digits, k=10))

    def _dispatch_order(self):
        """派单"""
        import random
        workers = ["张师傅", "李师傅", "王师傅"]
        self.context["worker"] = random.choice(workers)
        self.context["eta"] = f"{random.randint(15, 30)}分钟"
        self.context["payment_confirmed"] = True

    def chat(self, user_input: str) -> tuple[str, Optional[dict]]:
        """
        处理用户输入

        Returns:
            (AI回复, 可选的UI卡片)
        """
        # 1. 提取信息
        self._extract_info(user_input)

        # 2. 检测意图
        intent = self._detect_intent(user_input)

        # 3. 检查是否满足阶段转换条件
        if self._check_stage_transition(intent):
            self._do_stage_transition()

        # 4. 构建消息并调用 LLM
        self.messages.append(HumanMessage(content=user_input))

        system_prompt = self._get_stage_prompt()
        llm_messages = [SystemMessage(content=system_prompt)] + self.messages[-10:]  # 保留最近10轮

        response = self.llm.invoke(llm_messages)
        self.messages.append(response)

        # 5. 生成卡片（关键节点）
        card = None
        if self.stage == Stage.CONFIRMING and self.context.get("price"):
            card = {
                "type": "info_confirmation",
                "data": {
                    "rescue_type": self.context["rescue_type"],
                    "phone": self.context["phone"],
                    "address": self.context["address"],
                    "price": self.context["price"],
                }
            }
        elif self.stage == Stage.PAYMENT and self.context.get("order_id"):
            card = {
                "type": "payment_pending",
                "data": {
                    "order_id": self.context["order_id"],
                    "price": self.context["price"],
                }
            }
        elif self.stage == Stage.TRACKING and self.context.get("worker"):
            card = {
                "type": "order_dispatched",
                "data": {
                    "order_id": self.context["order_id"],
                    "worker": self.context["worker"],
                    "eta": self.context["eta"],
                }
            }

        return response.content, card

    def get_status(self) -> dict:
        """获取当前状态"""
        return {
            "stage": self.stage.value,
            "context": self.context,
            "missing_fields": self._get_missing_fields()
        }


# ==================== 交互体验优化示例 ====================

"""
【死板的交互】❌
用户：我车打不着火了
助手：请问您需要什么救援服务？
用户：搭电
助手：请提供您的联系电话。
用户：13800138000
助手：请提供您的地址。
用户：望京SOHO
助手：请确认信息...

【自然的交互】✅
用户：我车打不着火了
助手：别着急！这种情况一般是电瓶没电了，搭个电就能解决。您现在在哪儿呢？我帮您安排师傅过去~

用户：在望京这边，SOHO旁边
助手：好的望京SOHO，那边我们师傅很熟悉。方便留个手机号吗？师傅到了好联系您。

用户：13800138000，对了大概多久能到啊？
助手：这个时间段望京不太堵，估计 20 分钟左右能到。我帮您确认下信息哈：
     - 搭电服务
     - 位置：望京SOHO附近
     - 电话：138****0000
     费用是 80 元，确认的话我就帮您下单了？

用户：行
助手：好的！订单已提交，张师傅正在赶来，预计 18 分钟到。他会提前给您打电话的，保持手机畅通哈~有问题随时找我！

【关键区别】
1. 不是一问一答的表单模式
2. 主动安抚情绪
3. 顺带回答用户的问题
4. 自然地收集信息
5. 确认时用口语化表达
"""
