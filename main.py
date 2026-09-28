"""客服 Agent 交互入口（命令行版）。

使用前：
    1. pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    2. 设置环境变量 DEEPSEEK_API_KEY（以及 SILICONFLOW_API_KEY，知识库功能需要）
    3. python scripts/build_mock_db.py     # 初始化订单库 + 知识库
    4. python main.py                      # 开始对话

试试这些话：
    你好
    帮我查一下订单 E2025002            → 意图路由到"查订单"
    退货运费谁承担？                    → 意图路由到"查知识库"
    我要退款，订单号 E2025001          → 触发人工审批（程序暂停等你输入 同意/拒绝）
    年假有几天？                        → 知识库没有，触发兜底话术
"""
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agent import config
from agent.graph import build_graph


def run_once(graph, thread_config: dict, user_text: str) -> None:
    """执行一轮对话；若图在 interrupt 处暂停，则进入人工审批循环。"""
    # 以用户消息作为输入推进图；thread_id 相同的多次调用共享记忆
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_text)]},
        config=thread_config,
    )

    # 图可能在退款节点挂起。注意 langgraph 0.2.x 里 invoke 的返回值【不带】__interrupt__ 键
    # （那是更新版本的写法），要靠 get_state 快照判断：任务列表里挂着 interrupt，
    # 说明图停在人工审批点，需要人工决策后用 Command(resume=...) 恢复。
    while True:
        snapshot = graph.get_state(thread_config)
        pending = [intr for task in snapshot.tasks for intr in task.interrupts]
        if not pending:
            break
        for intr in pending:
            info = intr.value  # interrupt() 传入的订单信息字典
            print("\n" + "=" * 50)
            print(f"【人工审批】{info.get('prompt')}")
            print(f"  订单号：{info.get('order_no')}  商品：{info.get('product')}  金额：￥{info.get('amount')}")
            print("=" * 50)
        decision = input("审批意见（同意 / 拒绝）：").strip() or "拒绝"
        # Command(resume=...) 把人工决策送回图里，从挂起点继续执行
        result = graph.invoke(Command(resume=decision), config=thread_config)

    # 打印本轮 AI 的最后一条回复
    last = result["messages"][-1]
    print(f"\n客服：{last.content}\n")


def main() -> None:
    if not config.LLM_API_KEY:
        raise SystemExit("请先设置环境变量 DEEPSEEK_API_KEY")
    config.ensure_dirs()

    # SqliteSaver：把每一步的图状态持久化到 sqlite。
    # 即使程序退出重开，只要 thread_id 不变，记忆就还在 —— 这就是"检查点"。
    with SqliteSaver.from_conn_string(str(config.CHECKPOINT_DB)) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)

        thread_id = input("会话ID（直接回车用默认 demo-user；同一ID共享记忆）：").strip() or "demo-user"
        thread_config = {"configurable": {"thread_id": thread_id}}
        print(f"\n已进入会话 [{thread_id}]，输入 exit 退出。\n")

        while True:
            try:
                user_text = input("你：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break
            if not user_text:
                continue
            if user_text.lower() in ("exit", "quit"):
                print("再见！")
                break
            run_once(graph, thread_config, user_text)


if __name__ == "__main__":
    main()
