import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from rich.console import Console
from src.config import Config

console = Console()


def ingest_documents():
    if not os.path.exists(Config.DOCS_DIR) or not os.listdir(Config.DOCS_DIR):
        console.print(
            f"[bold red]Error:[/bold red] No documents found in {Config.DOCS_DIR}"
        )
        return False

    with console.status(
        "[bold green]Loading PDF Documents...[/bold green]", spinner="dots"
    ):
        # Load PDF
        loader = DirectoryLoader(Config.DOCS_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()
        console.log(f"Loaded [cyan]{len(docs)}[/cyan] pages from source.")

    with console.status(
        "[bold green]Splitting Text into Chunks...[/bold green]", spinner="dots"
    ):
        # Split Text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            add_start_index=True,
        )
        splits = text_splitter.split_documents(docs)
        console.log(f"Created [cyan]{len(splits)}[/cyan] document chunks.")

    with console.status(
        "[bold green]Embedding & Indexing (FAISS)...[/bold green]", spinner="dots"
    ):
        # Initialize Embedding Model
        embeddings = OllamaEmbeddings(model=Config.EMBEDDING_MODEL_NAME)

        # Create Vector Store
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

        save_path = os.path.join(Config.BASE_DIR, "data", "faiss_index")
        vectorstore.save_local(save_path)
        console.log(f"Index saved to [cyan]{save_path}[/cyan]")

    console.print(
        f"\n[bold green]Success![/bold green] Knowledge Base Ready. ({len(splits)} chunks)"
    )
    return True


if __name__ == "__main__":
    ingest_documents()
