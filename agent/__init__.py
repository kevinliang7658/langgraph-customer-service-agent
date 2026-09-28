"""agent 包：LangGraph 客服 Agent 的核心模块。

- config.py  配置（等价 application.yml）
- state.py   图状态定义（等价在节点间传递的上下文对象）
- tools.py   工具层：查订单（sqlite）、查知识库（Chroma）
- graph.py   图编排：意图路由 + 工具节点 + 人工审批中断
"""
