"""图状态（State）：在 Agent 各节点之间流转的"共享上下文"。

状态贯穿整条调用链：每个节点读它、改它、传给下一个节点。

LangGraph 的约定：
- 状态是一个 TypedDict（带类型的字典）；
- 字段可以声明"归并方式"（reducer）：messages 用 add_messages 表示
  节点返回的新消息是【追加】而不是【覆盖】——这就是多轮记忆的状态基础。
"""
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """客服 Agent 的状态。

    total=False 表示字段都可以缺省，节点只需返回自己要更新的键。
    """
    # 对话消息列表。Annotated[..., add_messages] = 追加式更新
    messages: Annotated[list, add_messages]

    # 意图分类结果：order（查订单）/ knowledge（政策咨询）/ refund（退款）/ chat（闲聊）
    intent: str
