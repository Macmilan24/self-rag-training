# Aura-RAG: Adaptive User-Reflective Agent

**A Production-Grade, Self-Reflective RAG System for the Commercial Code of Ethiopia.**

Aura-RAG is an advanced AI assistant that doesn't just retrieve documents—it *thinks* about them. Using a **LangGraph** state machine and a **Cognitive Loop**, it evaluates retrieved evidence, detects hallucinations, and rewrites its own search queries if it fails to find relevant legal articles.

## 🧠 Key Features

*   **Hybrid Intelligence:** Uses **OpenRouter** (Cloud) for high-quality generation and **Ollama** (Local) for fast, free critique and fallback generation.
*   **Self-Reflection (Self-RAG):**
    *   **Retrieval Grader:** Checks if found docs are actually relevant.
    *   **Hallucination Grader:** Ensures answers are grounded in facts.
    *   **Query Transformation:** Automatically rewrites bad queries based on *why* the previous search failed.
*   **Stateful Memory:** Remembers feedback from previous iterations to improve results.
*   **Resilient Architecture:** Handles JSON parsing errors and API failures gracefully.

## 🏗 Architecture

The system operates as a Cyclic Graph:

1.  **Retrieve:** Fetch legal articles using FAISS (Vector DB).
2.  **Grade:** Local LLM (`qwen2.5`) scores relevance (1-5) and gives reasoning.
3.  **Reflect:**
    *   *If irrelevant:* Store reasoning -> Rewrite Query -> Retry.
    *   *If relevant:* Generate Answer.
4.  **Verify:** Check for hallucinations.
    *   *If hallucinated:* Retry Generation.
    *   *If good:* Output to User.

## 🚀 Installation

### Prerequisites
*   Python 3.10+
*   [Poetry](https://python-poetry.org/) (Dependency Manager)
*   [Ollama](https://ollama.com/) (For local models)

### 1. Setup Local Models
Run this in your terminal to pull the necessary small models:
```bash
ollama pull gemma3:1b       # For Fallback Generation
ollama pull qwen2.5:1.5b    # For The Critic (The Brain)
ollama pull nomic-embed-text # For Embeddings
```

### 2. Install Dependencies
```bash
poetry install
```

### 3. Environment Setup
Create a `.env` file in the root directory:
```ini
OPENROUTER_API_KEY=sk-your-key-here
```

## 📖 Usage

### Phase 1: Ingest Data
Load your PDF documents (e.g., Ethiopian Commercial Code) into `data/source_docs/` and index them:

```bash
poetry run python -m src.ingest
```

### Phase 2: Run the Agent
Start the interactive CLI:

```bash
poetry run python main.py
```

## 📂 Project Structure

```text
Aura-RAG/
├── src/
│   ├── chains/       # LLM Logic (Critics, Retrieval)
│   ├── graph/        # LangGraph State Machine (Nodes, Workflow)
│   ├── utils/        # Prompts & Wrappers
│   └── config.py     # Settings
├── data/             # Logs and Vector Database
└── main.py           # CLI Entrypoint
```

## 🛠 Tech Stack
*   **Orchestration:** LangGraph, LangChain
*   **LLMs:** Google Gemini (via OpenRouter) + Gemma/Qwen (Local)
*   **Database:** FAISS
*   **UI:** Rich (Terminal)
