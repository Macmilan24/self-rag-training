from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import Config
from src.logger_setup import logger


class HybridLLM:

    def __init__(self):
        self.primary = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=Config.OPENROUTER_API_KEY,
            model=Config.PRIMARY_MODEL_NAME,
            temperature=0,
            default_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Aura-RAG-CLI",
            },
        )

        self.fallback = ChatOllama(
            model=Config.FALLBACK_MODEL_NAME,
            temperature=0,
            base_url="http://localhost:11434",
        )

    def invoke(self, messages):
        try:
            logger.info(
                "Generating with Primary Model...",
                extra={"details": {"model": Config.PRIMARY_MODEL_NAME}},
            )

            response = self.primary.invoke(messages)
            return response
        except Exception as e:
            error_msg = f"Primary model failed: {str(e)}. Switching to Fallback."
            logger.error(error_msg)
            print(
                f"\n Connection Stability Issue. Switching to Local Model ({Config.FALLBACK_MODEL_NAME})..."
            )

            # Execute Fallback
            try:
                response = self.fallback.invoke(messages)
                logger.info("Fallback generation successful.")
                return response
            except Exception as e_local:
                logger.critical(f"Both models failed. Local Error: {str(e_local)}")
                raise e_local


if __name__ == "__main__":
    print("Testing Hybrid Generator...")
    llm = HybridLLM()

    msg = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello! Briefly explain what a Commercial Code is."),
    ]

    result = llm.invoke(msg)
    print(f"\nResult:\n{result.content}")
