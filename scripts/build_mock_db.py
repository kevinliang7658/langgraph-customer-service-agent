"""建库脚本：初始化 mock 订单库（sqlite）+ 把示例知识文档灌进 Chroma。

运行（项目根目录下）：
    python scripts/build_mock_db.py

重复运行是安全的：订单表先删后建，Chroma 集合先清空再写入（幂等）。
"""
import sqlite3
import sys
from pathlib import Path

# 让脚本在根目录外运行也能 import agent 包（把项目根目录加入模块搜索路径）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import config  # noqa: E402
from agent.tools import embed_texts  # noqa: E402

# 示例订单：覆盖常见状态，方便演示不同查询结果
MOCK_ORDERS = [
    ("E2025001", "张三", "无线蓝牙耳机", 199.00, "已签收", "中通 7654321：已签收（2025-07-20）", "2025-07-15 10:23:00"),
    ("E2025002", "张三", "机械键盘", 349.00, "运输中", "圆通 8822334：到达杭州转运中心", "2025-07-23 16:45:00"),
    ("E2025003", "李四", "保温杯", 89.00, "待发货", "暂未揽收", "2025-07-24 09:12:00"),
    ("E2025004", "王五", "人体工学椅", 1299.00, "已签收", "专线物流：已签收（2025-07-10）", "2025-07-05 14:30:00"),
    ("E2025005", "李四", "手机壳", 29.90, "已退款", "退货已验收，款项已原路退回", "2025-07-01 11:05:00"),
]


def build_orders_db() -> None:
    """创建订单表并写入 mock 数据。"""
    conn = sqlite3.connect(str(config.ORDERS_DB))
    try:
        conn.execute("DROP TABLE IF EXISTS orders")
        conn.execute("""
            CREATE TABLE orders (
                order_no   TEXT PRIMARY KEY,   -- 订单号
                user_name  TEXT NOT NULL,      -- 下单人
                product    TEXT NOT NULL,      -- 商品名
                amount     REAL NOT NULL,      -- 金额
                status     TEXT NOT NULL,      -- 待发货/运输中/已签收/退款中/已退款
                logistics  TEXT NOT NULL,      -- 物流描述
                created_at TEXT NOT NULL       -- 下单时间
            )
        """)
        conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)", MOCK_ORDERS)
        conn.commit()
        print(f"[OK] 订单库已初始化：{config.ORDERS_DB}（{len(MOCK_ORDERS)} 条示例订单）")
    finally:
        conn.close()


def build_knowledge_base() -> None:
    """把 docs/售后知识库示例.md 按小节切分后写入 Chroma。"""
    if not config.SILICONFLOW_API_KEY:
        print("[跳过] 未设置 SILICONFLOW_API_KEY，知识库未初始化（查知识库功能将不可用）")
        return

    import chromadb

    doc_path = config.BASE_DIR / "docs" / "售后知识库示例.md"
    text = doc_path.read_text(encoding="utf-8")

    # 按 "## " 标题切小节：政策类文档一节一个主题，语义天然完整
    chunks = [
        section.strip()
        for section in text.split("## ")
        if section.strip() and not section.startswith("#")  # 排除文件开头的一级标题块
    ]

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    # 幂等：先删旧集合再重建，重复运行不会累积重复数据
    client.delete_collection("after_sales_kb") if "after_sales_kb" in [
        c.name for c in client.list_collections()
    ] else None
    collection = client.create_collection(
        name="after_sales_kb", metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[f"kb-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embed_texts(chunks),
    )
    print(f"[OK] 知识库已初始化：{len(chunks)} 个小节写入 Chroma")


if __name__ == "__main__":
    config.ensure_dirs()
    build_orders_db()
    build_knowledge_base()
    print("\n初始化完成，现在可以运行：python main.py")
