# 道路救援服务智能助手

基于 LangGraph 的 Agentic Workflow 实现的救援服务聊天机器人。

## 功能特性

- 🚗 **智能救援下单**：自动引导客户完成救援服务下单流程
- 💬 **自然语言交互**：支持自然语言对话，智能理解用户意图
- 📋 **信息收集**：自动收集救援所需的客户信息、位置信息和车辆信息
- 💳 **支付流程**：支持订单创建、支付确认、派单等完整流程
- 📊 **可视化卡片**：在关键节点展示信息收集卡片、待支付卡片、下单成功卡片等
- 🔍 **订单查询**：支持查询订单状态和救援进度

## 救援服务流程

```
确认发起救援 → 收集客户信息 → 发起订单（获取价格）→ 客户确认支付
    → 确认下单 → 派单 → 救援人员到达 → 完成救援
```

## 支持的救援类型

| 类型 | 基础价格 | 说明 |
|------|---------|------|
| 拖车 | ¥200 | 车辆无法行驶时拖车服务 |
| 搭电 | ¥80 | 电瓶没电无法启动 |
| 换胎 | ¥100 | 轮胎爆胎或漏气 |
| 送油 | ¥150 | 燃油耗尽 |
| 开锁 | ¥120 | 钥匙锁车内 |
| 现场维修 | ¥180 | 简单故障现场修理 |
| 困境救援 | ¥250 | 车辆陷入泥地等困境 |

## 项目结构

```
.
├── main.py                     # 主程序入口
├── requirements.txt            # Python 依赖
├── .env.example               # 环境变量示例
└── rescue_chatbot/            # 核心模块
    ├── __init__.py
    ├── state.py               # 状态定义
    ├── cards.py               # 可视化卡片
    ├── mock_api.py            # 模拟 API 服务
    ├── tools.py               # LangGraph 工具
    ├── nodes.py               # 工作流节点
    ├── graph.py               # LangGraph 工作流图
    └── prompts.py             # 提示词模板
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/your-repo/rescue-chatbot.git
cd rescue-chatbot

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 API 密钥
```

## 使用方法

### 运行演示（无需 API 密钥）

```bash
python main.py --demo
```

### 运行交互式聊天

```bash
# 使用默认模型 (gpt-4o-mini)
python main.py

# 指定模型
python main.py --model gpt-4o
python main.py --model claude-3-sonnet-20240229
```

### 命令帮助

在交互模式下，支持以下命令：

- `/help` - 显示帮助信息
- `/reset` - 重置会话
- `/status` - 显示当前状态
- `/demo` - 运行演示流程
- `/quit` - 退出程序

## API 接口

### 1. 发起救援 `initiate_rescue_order`

创建救援订单，返回订单价格。

```python
initiate_rescue_order(
    rescue_type="搭电",
    customer_name="张三",
    customer_phone="13800138000",
    address="北京市朝阳区望京SOHO",
    plate_number="京A12345"
)
```

### 2. 确认下单 `confirm_order_payment`

确认支付，分配救援人员。

```python
confirm_order_payment(order_id="RSC202401010001", payment_confirmed=True)
```

### 3. 取消订单 `cancel_rescue_order`

取消救援订单。

```python
cancel_rescue_order(order_id="RSC202401010001", reason="不需要了")
```

### 4. 查询订单状态 `query_order_status`

查询订单当前状态。

```python
query_order_status(order_id="RSC202401010001")
```

## 可视化卡片

系统在关键节点会生成可视化卡片：

1. **信息收集卡片** - 收集客户信息时展示
2. **订单待支付卡片** - 订单创建后等待支付
3. **下单成功卡片** - 支付确认后展示派单信息
4. **订单状态卡片** - 查询订单时展示当前状态
5. **救援完成卡片** - 救援完成后展示

## 技术架构

- **LangGraph**: 构建 Agentic Workflow
- **LangChain**: LLM 集成和工具调用
- **Pydantic**: 数据模型和验证
- **Rich**: 终端美化输出

## 工作流图

```
                    ┌─────────────┐
                    │   开始      │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Agent     │◄────────────────┐
                    └──────┬──────┘                 │
                           │                        │
              ┌────────────┼────────────┐           │
              │            │            │           │
       ┌──────▼──────┐     │     ┌──────▼──────┐    │
       │   Tools     │     │     │ Update State│    │
       └──────┬──────┘     │     └──────┬──────┘    │
              │            │            │           │
       ┌──────▼──────┐     │            │           │
       │Process Result│    │            │           │
       └──────┬──────┘     │            │           │
              │            │            │           │
              └────────────┴────────────┴───────────┘
                           │
                    ┌──────▼──────┐
                    │    结束     │
                    └─────────────┘
```

## License

MIT License
