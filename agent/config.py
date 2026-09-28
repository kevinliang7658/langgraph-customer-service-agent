"""全局配置：环境变量读取，密钥绝不写进代码。"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("AGENT_DATA_DIR", BASE_DIR / "data"))

# 订单 mock 库（sqlite）与记忆 checkpointer 的文件路径
ORDERS_DB = DATA_DIR / "orders.db"
CHECKPOINT_DB = DATA_DIR / "checkpoints.db"
CHROMA_DIR = DATA_DIR / "chroma"

# 对话模型：默认 DeepSeek（OpenAI 兼容协议），key 一律走环境变量
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# Embedding（知识库检索用）：DashScope text-embedding-v4（DeepSeek 没有 embedding 接口）
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")

TOP_K = int(os.getenv("TOP_K", "3"))  # 知识库检索切片数


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
