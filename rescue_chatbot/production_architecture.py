"""
生产环境推荐架构 - 救援服务工作流

核心原则：
1. 状态机控制流程，LLM 只做语言理解/生成
2. 所有状态转换必须显式、可审计
3. 关键操作需要幂等性保证
4. 完善的错误处理和降级机制
"""

import logging
import json
from enum import Enum
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ==================== 1. 状态定义（枚举，不可篡改）====================

class WorkflowStage(str, Enum):
    """工作流阶段 - 使用枚举保证类型安全"""
    INIT = "init"
    COLLECTING_INFO = "collecting_info"
    CONFIRMING_INFO = "confirming_info"
    CREATING_ORDER = "creating_order"
    PENDING_PAYMENT = "pending_payment"
    PROCESSING_PAYMENT = "processing_payment"
    DISPATCHING = "dispatching"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


class UserIntent(str, Enum):
    """用户意图 - LLM 输出的结构化结果"""
    REQUEST_RESCUE = "request_rescue"
    PROVIDE_INFO = "provide_info"
    CONFIRM = "confirm"
    REJECT = "reject"
    CANCEL = "cancel"
    QUERY_STATUS = "query_status"
    CONSULT = "consult"
    UNKNOWN = "unknown"


# ==================== 2. 状态转换规则（显式定义）====================

@dataclass
class TransitionRule:
    """状态转换规则"""
    from_stage: WorkflowStage
    to_stage: WorkflowStage
    required_intent: Optional[UserIntent] = None
    required_data: list[str] = field(default_factory=list)
    guard_condition: Optional[str] = None  # 额外校验条件


# 显式定义所有合法转换
TRANSITION_RULES: list[TransitionRule] = [
    # 初始 → 收集信息
    TransitionRule(
        from_stage=WorkflowStage.INIT,
        to_stage=WorkflowStage.COLLECTING_INFO,
        required_intent=UserIntent.REQUEST_RESCUE
    ),

    # 收集信息 → 确认信息（需要所有必填字段）
    TransitionRule(
        from_stage=WorkflowStage.COLLECTING_INFO,
        to_stage=WorkflowStage.CONFIRMING_INFO,
        required_data=["rescue_type", "customer_phone", "address"]
    ),

    # 确认信息 → 创建订单
    TransitionRule(
        from_stage=WorkflowStage.CONFIRMING_INFO,
        to_stage=WorkflowStage.CREATING_ORDER,
        required_intent=UserIntent.CONFIRM
    ),

    # 确认信息 → 返回收集（用户要修改）
    TransitionRule(
        from_stage=WorkflowStage.CONFIRMING_INFO,
        to_stage=WorkflowStage.COLLECTING_INFO,
        required_intent=UserIntent.REJECT
    ),

    # 创建订单 → 等待支付
    TransitionRule(
        from_stage=WorkflowStage.CREATING_ORDER,
        to_stage=WorkflowStage.PENDING_PAYMENT,
        required_data=["order_id", "price"]
    ),

    # 等待支付 → 处理支付
    TransitionRule(
        from_stage=WorkflowStage.PENDING_PAYMENT,
        to_stage=WorkflowStage.PROCESSING_PAYMENT,
        required_intent=UserIntent.CONFIRM,
        guard_condition="order_not_expired"
    ),

    # 等待支付 → 取消
    TransitionRule(
        from_stage=WorkflowStage.PENDING_PAYMENT,
        to_stage=WorkflowStage.CANCELLED,
        required_intent=UserIntent.CANCEL
    ),

    # 处理支付 → 派单中
    TransitionRule(
        from_stage=WorkflowStage.PROCESSING_PAYMENT,
        to_stage=WorkflowStage.DISPATCHING,
        required_data=["payment_id"]
    ),

    # 派单中 → 进行中
    TransitionRule(
        from_stage=WorkflowStage.DISPATCHING,
        to_stage=WorkflowStage.IN_PROGRESS,
        required_data=["rescue_worker_id"]
    ),

    # 进行中 → 完成
    TransitionRule(
        from_stage=WorkflowStage.IN_PROGRESS,
        to_stage=WorkflowStage.COMPLETED,
        guard_condition="rescue_confirmed"
    ),
]


# ==================== 3. 状态机核心 ====================

class StateMachineError(Exception):
    """状态机错误"""
    pass


class InvalidTransitionError(StateMachineError):
    """非法状态转换"""
    pass


class WorkflowContext(BaseModel):
    """工作流上下文 - 持久化存储"""
    session_id: str
    current_stage: WorkflowStage = WorkflowStage.INIT
    rescue_type: Optional[str] = None
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None
    address: Optional[str] = None
    plate_number: Optional[str] = None
    order_id: Optional[str] = None
    price: Optional[float] = None
    payment_id: Optional[str] = None
    rescue_worker_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    stage_history: list[dict] = Field(default_factory=list)
    error_count: int = 0

    def has_required_data(self, fields: list[str]) -> bool:
        """检查是否有必要数据"""
        for field in fields:
            if getattr(self, field, None) is None:
                return False
        return True

    def record_transition(self, from_stage: WorkflowStage, to_stage: WorkflowStage, reason: str):
        """记录状态转换（审计日志）"""
        self.stage_history.append({
            "from": from_stage.value,
            "to": to_stage.value,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        })
        self.updated_at = datetime.now()


class WorkflowStateMachine:
    """工作流状态机"""

    def __init__(self, context: WorkflowContext):
        self.context = context
        self._build_transition_map()

    def _build_transition_map(self):
        """构建转换映射"""
        self.transitions: dict[WorkflowStage, list[TransitionRule]] = {}
        for rule in TRANSITION_RULES:
            if rule.from_stage not in self.transitions:
                self.transitions[rule.from_stage] = []
            self.transitions[rule.from_stage].append(rule)

    def can_transition(self, to_stage: WorkflowStage, intent: Optional[UserIntent] = None) -> tuple[bool, str]:
        """检查是否可以转换到目标状态"""
        current = self.context.current_stage
        rules = self.transitions.get(current, [])

        for rule in rules:
            if rule.to_stage != to_stage:
                continue

            # 检查意图
            if rule.required_intent and rule.required_intent != intent:
                continue

            # 检查必要数据
            if not self.context.has_required_data(rule.required_data):
                missing = [f for f in rule.required_data if getattr(self.context, f, None) is None]
                return False, f"缺少必要数据: {missing}"

            # 检查守卫条件
            if rule.guard_condition:
                if not self._check_guard(rule.guard_condition):
                    return False, f"条件不满足: {rule.guard_condition}"

            return True, "OK"

        return False, f"不允许从 {current.value} 转换到 {to_stage.value}"

    def _check_guard(self, condition: str) -> bool:
        """检查守卫条件"""
        guards = {
            "order_not_expired": self._check_order_not_expired,
            "rescue_confirmed": self._check_rescue_confirmed,
        }
        checker = guards.get(condition)
        return checker() if checker else True

    def _check_order_not_expired(self) -> bool:
        """检查订单是否过期"""
        # 实际实现：检查订单创建时间是否超过30分钟
        return True

    def _check_rescue_confirmed(self) -> bool:
        """检查救援是否确认完成"""
        # 实际实现：检查救援人员是否确认完成
        return True

    def transition(self, to_stage: WorkflowStage, intent: Optional[UserIntent] = None, reason: str = "") -> bool:
        """执行状态转换"""
        can, message = self.can_transition(to_stage, intent)

        if not can:
            logger.warning(f"状态转换被拒绝: {message}")
            raise InvalidTransitionError(message)

        # 记录转换
        from_stage = self.context.current_stage
        self.context.record_transition(from_stage, to_stage, reason)
        self.context.current_stage = to_stage

        logger.info(f"状态转换: {from_stage.value} → {to_stage.value}, 原因: {reason}")
        return True

    def get_available_transitions(self) -> list[WorkflowStage]:
        """获取当前可用的转换目标"""
        current = self.context.current_stage
        rules = self.transitions.get(current, [])
        return [rule.to_stage for rule in rules]


# ==================== 4. LLM 接口（只做 NLU/NLG）====================

class LLMResponse(BaseModel):
    """LLM 响应结构"""
    intent: UserIntent
    extracted_data: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    reply_text: str = ""


class LLMService(ABC):
    """LLM 服务抽象接口"""

    @abstractmethod
    def understand_intent(self, message: str, context: WorkflowContext) -> LLMResponse:
        """理解用户意图（NLU）"""
        pass

    @abstractmethod
    def generate_reply(self, stage: WorkflowStage, context: WorkflowContext, data: dict) -> str:
        """生成回复（NLG）"""
        pass


class OpenAILLMService(LLMService):
    """OpenAI LLM 服务实现"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model

    def understand_intent(self, message: str, context: WorkflowContext) -> LLMResponse:
        """使用 LLM 理解意图并提取信息"""

        # 构造结构化提取 prompt
        prompt = f"""分析用户消息，返回 JSON 格式：

当前阶段: {context.current_stage.value}
用户消息: "{message}"

返回格式:
{{
  "intent": "request_rescue|provide_info|confirm|reject|cancel|query_status|consult|unknown",
  "extracted_data": {{
    "rescue_type": "拖车|搭电|换胎|送油|开锁|现场维修|困境救援|null",
    "customer_name": "姓名或null",
    "customer_phone": "手机号或null",
    "address": "地址或null",
    "plate_number": "车牌或null"
  }},
  "confidence": 0.0-1.0
}}

只返回 JSON，不要其他内容。"""

        # 这里调用实际的 OpenAI API
        # response = openai.chat.completions.create(...)

        # 示例返回
        return LLMResponse(
            intent=UserIntent.PROVIDE_INFO,
            extracted_data={},
            confidence=0.9,
            reply_text=""
        )

    def generate_reply(self, stage: WorkflowStage, context: WorkflowContext, data: dict) -> str:
        """生成自然语言回复"""

        # 根据阶段使用模板 + LLM 润色
        templates = {
            WorkflowStage.COLLECTING_INFO: "请提供以下信息：{missing_fields}",
            WorkflowStage.CONFIRMING_INFO: "请确认信息：{info_summary}",
            WorkflowStage.PENDING_PAYMENT: "订单已创建，金额 ¥{price}，请确认支付",
        }

        template = templates.get(stage, "请问还有什么可以帮您？")
        return template.format(**data)


# ==================== 5. 业务服务层 ====================

class RescueAPIService:
    """救援 API 服务（与后端对接）"""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key

    def create_order(self, context: WorkflowContext) -> dict:
        """创建订单 - 需要幂等性"""
        # 幂等键：session_id + 信息哈希
        idempotency_key = f"{context.session_id}:{hash((context.rescue_type, context.customer_phone, context.address))}"

        # 调用后端 API
        # response = requests.post(f"{self.base_url}/orders", headers={"Idempotency-Key": idempotency_key}, ...)

        return {"order_id": "RSC123456", "price": 150.0}

    def confirm_payment(self, order_id: str, payment_method: str) -> dict:
        """确认支付 - 需要幂等性"""
        return {"payment_id": "PAY123456", "status": "success"}

    def cancel_order(self, order_id: str, reason: str) -> dict:
        """取消订单"""
        return {"success": True, "refund_amount": 0.0}

    def query_status(self, order_id: str) -> dict:
        """查询状态"""
        return {"status": "dispatched", "rescue_worker": "张师傅", "eta": "15分钟"}


# ==================== 6. 工作流编排器 ====================

class WorkflowOrchestrator:
    """工作流编排器 - 整合状态机、LLM、业务服务"""

    def __init__(
        self,
        llm_service: LLMService,
        api_service: RescueAPIService,
        context_store: Any  # Redis/数据库
    ):
        self.llm = llm_service
        self.api = api_service
        self.store = context_store

    def process_message(self, session_id: str, message: str) -> dict:
        """处理用户消息"""

        # 1. 加载上下文
        context = self._load_context(session_id)
        state_machine = WorkflowStateMachine(context)

        # 2. LLM 理解意图
        llm_response = self.llm.understand_intent(message, context)

        # 3. 更新上下文数据
        for key, value in llm_response.extracted_data.items():
            if value is not None:
                setattr(context, key, value)

        # 4. 根据当前阶段执行业务逻辑
        result = self._execute_stage_logic(state_machine, llm_response.intent)

        # 5. 生成回复
        reply = self.llm.generate_reply(context.current_stage, context, result.get("data", {}))

        # 6. 保存上下文
        self._save_context(context)

        return {
            "reply": reply,
            "stage": context.current_stage.value,
            "ui_card": result.get("ui_card"),
            "actions": state_machine.get_available_transitions()
        }

    def _execute_stage_logic(self, sm: WorkflowStateMachine, intent: UserIntent) -> dict:
        """执行阶段业务逻辑"""
        context = sm.context
        stage = context.current_stage

        # 根据阶段和意图执行不同逻辑
        if stage == WorkflowStage.INIT and intent == UserIntent.REQUEST_RESCUE:
            sm.transition(WorkflowStage.COLLECTING_INFO, intent, "用户发起救援请求")
            return {"data": {"missing_fields": self._get_missing_fields(context)}}

        elif stage == WorkflowStage.COLLECTING_INFO:
            missing = self._get_missing_fields(context)
            if not missing:
                sm.transition(WorkflowStage.CONFIRMING_INFO, reason="信息收集完成")
                return {"ui_card": self._create_info_card(context)}
            return {"data": {"missing_fields": missing}}

        elif stage == WorkflowStage.CONFIRMING_INFO:
            if intent == UserIntent.CONFIRM:
                sm.transition(WorkflowStage.CREATING_ORDER, intent, "用户确认信息")
                # 调用 API 创建订单
                order = self.api.create_order(context)
                context.order_id = order["order_id"]
                context.price = order["price"]
                sm.transition(WorkflowStage.PENDING_PAYMENT, reason="订单创建成功")
                return {"ui_card": self._create_payment_card(context)}
            elif intent == UserIntent.REJECT:
                sm.transition(WorkflowStage.COLLECTING_INFO, intent, "用户要求修改信息")

        elif stage == WorkflowStage.PENDING_PAYMENT:
            if intent == UserIntent.CONFIRM:
                sm.transition(WorkflowStage.PROCESSING_PAYMENT, intent, "用户确认支付")
                # 调用支付 API
                payment = self.api.confirm_payment(context.order_id, "wechat")
                context.payment_id = payment["payment_id"]
                sm.transition(WorkflowStage.DISPATCHING, reason="支付成功")
                return {"ui_card": self._create_success_card(context)}
            elif intent == UserIntent.CANCEL:
                self.api.cancel_order(context.order_id, "用户取消")
                sm.transition(WorkflowStage.CANCELLED, intent, "用户取消订单")

        return {}

    def _get_missing_fields(self, context: WorkflowContext) -> list[str]:
        """获取缺失字段"""
        required = ["rescue_type", "customer_phone", "address"]
        return [f for f in required if getattr(context, f, None) is None]

    def _create_info_card(self, context: WorkflowContext) -> dict:
        return {"type": "info_collection", "data": context.dict()}

    def _create_payment_card(self, context: WorkflowContext) -> dict:
        return {"type": "payment_pending", "order_id": context.order_id, "price": context.price}

    def _create_success_card(self, context: WorkflowContext) -> dict:
        return {"type": "order_success", "order_id": context.order_id}

    def _load_context(self, session_id: str) -> WorkflowContext:
        """从存储加载上下文"""
        # data = self.store.get(f"workflow:{session_id}")
        # if data:
        #     return WorkflowContext.parse_raw(data)
        return WorkflowContext(session_id=session_id)

    def _save_context(self, context: WorkflowContext):
        """保存上下文到存储"""
        # self.store.set(f"workflow:{context.session_id}", context.json(), ex=3600)
        pass


# ==================== 7. 生产环境额外考虑 ====================

"""
生产环境 Checklist：

1. 【持久化】
   - 使用 Redis/PostgreSQL 存储会话状态
   - 状态变更需要事务保证
   - 实现会话过期清理

2. 【幂等性】
   - 所有写操作使用幂等键
   - 订单创建、支付确认必须幂等
   - 防止重复下单/重复支付

3. 【监控告警】
   - 状态转换埋点
   - LLM 调用延迟/成功率监控
   - 异常状态告警（长时间停留在某阶段）

4. 【降级策略】
   - LLM 不可用时：使用规则引擎 fallback
   - 支付失败时：自动重试 + 人工介入
   - API 超时：熔断 + 降级

5. 【安全】
   - 敏感信息脱敏存储
   - 操作审计日志
   - 权限控制（取消订单需验证身份）

6. 【测试】
   - 状态机全路径覆盖测试
   - LLM Mock 测试
   - 压力测试
"""
