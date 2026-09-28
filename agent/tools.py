"""工具层：Agent 的"手脚" —— 查订单（sqlite mock 库）和查知识库（Chroma）。

大模型负责"决策调哪个工具"，
工具本身必须是确定性代码 —— 查单号就是 SQL 查询，绝不让模型猜。
"""
import re
import sqlite3

import chromadb
from openai import OpenAI

from agent import config

# 订单号格式：E + 数字（如 E2025001），用正则从用户话里抠出来
ORDER_NO_PATTERN = re.compile(r"E\d{6,}")


def extract_order_no(text: str) -> str | None:
    """从用户输入中提取订单号，没有则返回 None。"""
    match = ORDER_NO_PATTERN.search(text.upper())
    return match.group(0) if match else None


def query_order(order_no: str) -> dict | None:
    """查订单工具：按单号查 mock 订单库。

    返回订单字典；查无此单返回 None。真实项目里这里是调订单中台的 REST API。
    """
    conn = sqlite3.connect(str(config.ORDERS_DB))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT order_no, user_name, product, amount, status, logistics, created_at "
            "FROM orders WHERE order_no = ?",
            (order_no,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_order_status(order_no: str, new_status: str) -> bool:
    """更新订单状态（退款审批通过后调用）。返回是否更新成功。"""
    conn = sqlite3.connect(str(config.ORDERS_DB))
    try:
        cursor = conn.execute(
            "UPDATE orders SET status = ? WHERE order_no = ?", (new_status, order_no)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ---------- 知识库检索（Chroma 向量检索） ----------

_chroma_collection = None
_embed_client = None


def _get_collection():
    """懒加载 Chroma 集合（相当于 Spring 的懒初始化 Bean）。"""
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _chroma_collection = client.get_or_create_collection(
            name="after_sales_kb", metadata={"hnsw:space": "cosine"}
        )
    return _chroma_collection


def _get_embed_client() -> OpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = OpenAI(
            base_url=config.SILICONFLOW_BASE_URL, api_key=config.SILICONFLOW_API_KEY
        )
    return _embed_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量 Embedding。建库脚本也复用这个函数，保证入库和检索用同一个模型。"""
    resp = _get_embed_client().embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]


def search_knowledge(query: str) -> list[str]:
    """查知识库工具：检索最相关的政策段落，返回纯文本列表。"""
    collection = _get_collection()
    if collection.count() == 0:
        return ["（知识库为空，请先运行 scripts/build_mock_db.py 初始化）"]
    query_vec = embed_texts([query])[0]
    result = collection.query(query_embeddings=[query_vec], n_results=config.TOP_K)
    return result["documents"][0] if result["documents"] else []
