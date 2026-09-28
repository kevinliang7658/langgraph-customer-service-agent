"""图编排：把"意图路由 → 工具/生成节点 → 人工审批"串成一张有向图。

LangGraph 核心概念速览：
- 图（Graph）= 流程编排定义，相当于一份 Activiti/Flowable 流程图
- 节点（Node）= 一个普通 Python 函数：读状态 → 干活 → 返回要更新的状态字段
- 边（Edge）= 节点间的流转关系；条件边 = if/else 路由
- 检查点（Checkpointer）= 每步执行完把状态存盘，所以能"记忆"、能"中断后恢复"
- interrupt() = 流程挂起等人工输入，类似工作流引擎的"人工任务节点"
"""
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from openai import OpenAI

from agent import config
from agent.state import AgentState
from agent.tools import (
    extract_order_no,
    query_order,
    search_knowledge,
    update_order_status,
)

# ---------- 大模型客户端（阿里，OpenAI 兼容协议） ----------
_llm = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)


def _chat(system: str, user: str) -> str:
    """最简 LLM 调用封装：单轮一问一答。"""
    resp = _llm.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def _last_user_text(state: AgentState) -> str:
    """取最近一条用户消息的文本。"""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get("role") == "user"):
            return msg.content if isinstance(msg, HumanMessage) else msg["content"]
    return ""


# 滑动窗口：只带最近 10 条历史给模型，控制 token 成本（太久远的上下文通常没帮助）
HISTORY_WINDOW = 10


def _chat_with_history(state: AgentState, system: str) -> str:
    """带多轮记忆的对话：把 checkpointer 持久化的历史消息一起发给模型。

    state["messages"] 是追加式的（add_messages reducer）：旧历史在前，
    本轮 HumanMessage 在最后。之前只传当前一句话，导致"刚才那个订单到哪了？"
    这类指代追问答不上来——状态里明明有记忆，模型却看不到。
    """
    history = [m for m in state["messages"][:-1] if isinstance(m, (HumanMessage, AIMessage))]
    history = history[-HISTORY_WINDOW:]
    messages = [{"role": "system", "content": system}]
    for m in history:
        messages.append({"role": "user" if isinstance(m, HumanMessage) else "assistant",
                         "content": m.content})
    messages.append({"role": "user", "content": _last_user_text(state)})
    resp = _llm.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def _order_no_from_state(state: AgentState) -> str | None:
    """提取订单号：先看本轮输入；没有再回溯会话历史里最近出现过的单号。

    这就是"多轮对话不用重复单号"的实现——回溯是确定性代码，
    绝不让大模型猜单号（模型猜的单号不可信）。
    """
    order_no = extract_order_no(_last_user_text(state))
    if order_no:
        return order_no
    for m in reversed(state["messages"][:-1]):
        if isinstance(m, (HumanMessage, AIMessage)):
            found = extract_order_no(m.content)
            if found:
                return found
    return None


# ---------- 节点1：意图分类（路由的大脑） ----------

CLASSIFY_PROMPT = """你是客服系统的意图分类器。把用户最后一句话分成以下四类，只输出类别单词：
order     —— 查询订单状态、物流进度（通常含订单号或"我的订单"）
refund    —— 明确要求退款、退货退款（涉及钱款退回的操作）
knowledge —— 咨询售后政策：退货规则、运费、保修、发票、发货时效等
chat      —— 打招呼、寒暄、其他无法归类的闲聊
complaint —— 用户表达不满、抱怨服务或商品、要求投诉/索赔/升级处理
只输出一个单词，不要解释。"""


def classify_node(state: AgentState) -> dict:
    """意图分类节点：输出写入 state["intent"]，供条件边读取。"""
    intent = _chat(CLASSIFY_PROMPT, _last_user_text(state)).lower()
    # 防御性处理：模型可能输出多余字符，只认四个合法值
    for label in ( "complaint", "refund", "order", "knowledge", "chat"):
        if label in intent:
            return {"intent": label}
    return {"intent": "chat"}


def route_by_intent(state: AgentState) -> str:
    """条件边的路由函数：把 intent 映射成下一个节点名。"""
    return {
        "order": "order_node",
        "refund": "refund_node",
        "knowledge": "knowledge_node",
        "complaint": "complaint",
    }.get(state.get("intent", "chat"), "chat_node")


# ---------- 节点2：查订单 ----------

def order_node(state: AgentState) -> dict:
    """查订单节点：提取单号 → SQL 查询 → 模板化回复（确定性输出，不让模型编单号）。"""
    order_no = _order_no_from_state(state)
    if not order_no:
        reply = "请提供您的订单号（格式如 E2025001），我马上帮您查询。"
        return {"messages": [AIMessage(content=reply)]}

    order = query_order(order_no)
    if not order:
        reply = f"未查询到订单 {order_no}，请核对单号是否正确，或联系人工客服核实。"
    else:
        reply = (
            f"为您查到订单 {order['order_no']}：\n"
            f"- 商品：{order['product']}\n"
            f"- 金额：￥{order['amount']}\n"
            f"- 状态：{order['status']}\n"
            f"- 物流：{order['logistics']}\n"
            f"- 下单时间：{order['created_at']}"
        )
    return {"messages": [AIMessage(content=reply)]}


# ---------- 节点3：查知识库（RAG） ----------

KB_PROMPT = """你是电商售后客服。只能根据【参考资料】回答，资料里没有的就说
"这个问题超出了我的知识范围，为您转接人工客服"，禁止编造。
回答亲切简洁，200 字以内。

【参考资料】
{context}"""


def knowledge_node(state: AgentState) -> dict:
    """知识库问答节点：Chroma 检索 + LLM 组织语言（带会话历史，支持指代追问）。"""
    question = _last_user_text(state)
    docs = search_knowledge(question)
    answer = _chat_with_history(state, KB_PROMPT.format(context="\n\n".join(docs)))
    return {"messages": [AIMessage(content=answer)]}


# ---------- 节点4：退款（人工兜底，本项目的灵魂） ----------

def refund_node(state: AgentState) -> dict:
    """退款节点：涉及资金操作，必须人工审批。

    流程：提取单号（支持从会话历史回溯） → 查订单 → interrupt() 挂起等审批 → 按审批结果执行。
    interrupt() 的机制：
    - 第一次执行到 interrupt() 时抛出特殊异常，图【暂停】，状态已存进 checkpointer；
    - 人工审批后用 Command(resume="...") 再次 invoke，节点会【从头重跑】，
      但 interrupt() 不再抛异常，而是返回 resume 的值。
    所以 interrupt() 之前的代码必须是幂等的（重跑无副作用）——查订单满足，写库不满足。
    """
    order_no = _order_no_from_state(state)
    if not order_no:
        return {"messages": [AIMessage(content="请提供需要退款的订单号（如 E2025001）。")]}

    order = query_order(order_no)  # 幂等的只读操作，重跑安全
    if not order:
        return {"messages": [AIMessage(content=f"未查询到订单 {order_no}，无法办理退款。")]}
    if order["status"] in ("退款中", "已退款"):
        return {"messages": [AIMessage(content=f"订单 {order_no} 当前状态为「{order['status']}」，无需重复申请。")]}

    # 挂起：把订单信息抛给人工审批者（interrupt 的参数会原样传给调用方展示）
    decision = interrupt({
        "type": "refund_approval",
        "order_no": order["order_no"],
        "product": order["product"],
        "amount": order["amount"],
        "prompt": "检测到退款申请，请人工审批（回复 同意 / 拒绝）：",
    })

    # 恢复执行：decision 就是 Command(resume=...) 里人工给的答复
    if "同意" in str(decision):
        update_order_status(order_no, "退款中")  # 写操作放在 interrupt 之后，保证只执行一次
        reply = (
            f"退款申请已审批通过 √ 订单 {order_no}（￥{order['amount']}）"
            f"已转入退款流程，款项将于 1-3 个工作日原路退回。"
        )
    else:
        reply = f"很抱歉，订单 {order_no} 的退款申请未通过审批。如有疑问可回复「转人工」进一步沟通。"
    return {"messages": [AIMessage(content=reply)]}


# ---------- 节点5：闲聊 ----------

def chat_node(state: AgentState) -> dict:
    """闲聊节点：人设回复。如需转人工引导也在这里处理。"""
    text = _last_user_text(state)
    if "人工" in text:
        reply = "已为您记录转人工请求，客服专员工作时间 9:00-21:00 内会尽快接入，请稍候。"
    else:
        reply = _chat_with_history(state, "你是电商售后客服助手，亲切简洁地回应用户，并引导用户咨询订单或售后问题。")
    return {"messages": [AIMessage(content=reply)]}

# ---------- 节点6：投诉 ----------
def complaint(state: AgentState) -> dict:
    """投诉节点：处理用户投诉。"""
    text = _last_user_text(state)
    reply = _chat_with_history(state, "你是电商售后客服助手，用严肃安抚话术回复用户的投诉问题。")
    return {"messages": [AIMessage(content=reply)]}


# ---------- 组装图 ----------

def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """构建并编译客服 Agent 图。

    结构：
        START → classify →（条件边）→ order/refund/knowledge/chat 四个节点 → END
    传入 checkpointer 后，每个 thread_id 的对话状态独立持久化：
    多轮记忆和中断恢复都靠它。
    """
    builder = StateGraph(AgentState)
    builder.add_node("classify", classify_node)
    builder.add_node("order_node", order_node)
    builder.add_node("refund_node", refund_node)
    builder.add_node("knowledge_node", knowledge_node)
    builder.add_node("chat_node", chat_node)
    builder.add_node("complaint", complaint)

    builder.add_edge(START, "classify")
    # 条件边：classify 之后按 route_by_intent 的返回值决定去向
    builder.add_conditional_edges("classify", route_by_intent)
    for node in ("order_node", "refund_node", "knowledge_node", "chat_node", "complaint"):
        builder.add_edge(node, END)

    return builder.compile(checkpointer=checkpointer)
