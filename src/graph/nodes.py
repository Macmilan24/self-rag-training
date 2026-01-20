from langchain_core.messages import HumanMessage, SystemMessage
from src.graph.state import GraphState
from src.chains.retrieval import get_retriever
from src.chains.critics import question_rewriter, retrieval_grader
from src.utils.llm_wrapper import HybridLLM
from src.logger_setup import logger


def retrieve(state: GraphState):
    logger.info("--- NODE: RETRIEVE ---")
    question = state["question"]
    retriever = get_retriever()
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state: GraphState):
    logger.info("--- NODE: GENERATE ---")
    question = state["question"]
    documents = state["documents"]

    context = "\n\n".join([doc.page_content for doc in documents])
    sys_msg = SystemMessage(
        content="You are an expert on the Commercial Code of Ethiopia. Answer based strictly on the context."
    )
    human_msg = HumanMessage(content=f"Question: {question}\n\nContext: {context}")

    llm = HybridLLM()
    generation = llm.invoke([sys_msg, human_msg])

    return {"generation": generation.content}


def grade_documents(state: GraphState):
    """
    Analytic Grading: Stores WHY documents failed.
    """
    logger.info("--- NODE: GRADE DOCUMENTS ---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    feedback_list = []

    SCORE_THRESHOLD = 3

    for d in documents:
        result = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        score = result.get("score", 0)
        reason = result.get("reasoning", "No reason provided")

        logger.info(f"Doc Score: {score}/5 | Reason: {reason[:50]}...")

        if score >= SCORE_THRESHOLD:
            filtered_docs.append(d)
        else:
            feedback_list.append(reason)

    updates = {"documents": filtered_docs, "question": question}

    if feedback_list:
        summary = f"Docs rejected because: {feedback_list[0]}"
        updates["feedback_history"] = [summary]
    if not filtered_docs:
        logger.warning("--- ALL DOCS FILTERED: Keeping originals for context ---")
        updates["documents"] = documents

    return updates


def transform_query(state: GraphState):
    """
    Uses the FEEDBACK HISTORY to rewrite the query.
    """
    logger.info("--- NODE: TRANSFORM QUERY ---")
    question = state["question"]
    feedback = state.get("feedback_history", ["No specific feedback"])

    # Flatten feedback list to string
    feedback_str = " ".join(feedback)

    # Rewrite based on feedback
    better_question = question_rewriter.invoke(
        {"question": question, "feedback": feedback_str}
    )

    logger.info(f"--- OLD QUERY: {question}")
    logger.info(f"--- CRITIC FEEDBACK: {feedback_str}")
    logger.info(f"--- NEW QUERY: {better_question}")

    current_retries = state.get("retry_count", 0)
    return {
        "question": better_question,
        "retry_count": current_retries + 1,
        "feedback_history": [],  # Clear feedback after using it
    }
