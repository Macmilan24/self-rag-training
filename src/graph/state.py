from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's original (or rewritten) question.
        generation: The LLM's current generated answer.
        documents: List of retrieved documents.
        retry_count: Safety counter to stop infinite loops.
        gen_attempts: Safety counter to stop infinite generation retries.
        feedback_history: A list of strings containing specific advice from the Critic
                          on what was missing in previous steps.
    """

    question: str
    generation: str
    documents: List[str]
    retrival_retries: int
    generation_retries: int
    feedback_history: List[str]
