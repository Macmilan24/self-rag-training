import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from src.config import Config


def get_retriever():
    if not os.path.exists(os.path.join(Config.BASE_DIR, "data", "faiss_index")):
        raise FileNotFoundError("FAISS index not found. Run ingest.py first.")

    embeddings = OllamaEmbeddings(model=Config.EMBEDDING_MODEL_NAME)

    vectorstore = FAISS.load_local(
        os.path.join(Config.BASE_DIR, "data", "faiss_index"),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore.as_retriever(search_kwargs={"k": 5})
