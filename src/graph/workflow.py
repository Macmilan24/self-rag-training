from langgraph.graph import END, StateGraph
from src.graph.state import GraphState
from src.graph.nodes import retrieve, grade_documents, generate, transform_query
from src.chains.critics import hallucination_grader, answer_grader
from src.logger_setup import logger
from src.config import Config


def decide_to_generate(state: GraphState):
    filtered_documents = state["documents"]
    current_retries = state.get("retry_count", 0)

    # HARD STOP to prevent RecursionError
    if current_retries >= Config.MAX_RETRIES:
        logger.warning("--- MAX RETRIES REACHED -> FORCING GENERATION ---")
        return "generate"

    if not filtered_documents:
        return "transform_query"
    else:
        return "generate"


def grade_generation_v_documents_and_question(state: GraphState):
    """
    Decides if the generation is good enough.
    """
    current_retries = state.get("retry_count", 0)

    if current_retries >= Config.MAX_RETRIES:

        logger.warning("--- DECISION: MAX RETRIES REACHED -> ACCEPTING ANSWER ---")
        return "useful"

    res = hallucination_grader.invoke(
        {"documents": state["documents"], "generation": state["generation"]}
    )
    if res["score"] >= 4:
        res_util = answer_grader.invoke(
            {"question": state["question"], "generation": state["generation"]}
        )
        if res_util["score"] >= 4:
            return "useful"
        else:
            return "not useful"  # Goes to transform_query
    else:
        return "not supported"


workflow = StateGraph(GraphState)

# Nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

# Entry Point
workflow.set_entry_point("retrieve")

# Edges
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)

workflow.add_edge("transform_query", "retrieve")

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

app = workflow.compile()
