import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOG_DIR = os.path.join(BASE_DIR, "data", "logs")
    DOCS_DIR = os.path.join(BASE_DIR, "data", "source_docs")
    VECTOR_DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

    PRIMARY_MODEL_NAME = "xiaomi/mimo-v2-flash:free"

    FALLBACK_MODEL_NAME = "qwen2.5:1.5b"

    CRITIC_MODEL_NAME = "qwen2.5:1.5b"

    EMBEDDING_MODEL_NAME = "nomic-embed-text"

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_RETRIES = 2
