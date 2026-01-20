from langchain_core.prompts import ChatPromptTemplate


REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """You are a Strategic Search Optimizer.
    
    User's Goal: {question}
    
    Problem: The previous search results were rejected.
    Critic's Feedback: {feedback}
    
    Task: meaningful specific search query that addresses the feedback. 
    Focus on specific Article numbers or legal terminology mentioned in the feedback.
    
    Return ONLY the new query text."""
)

GRADE_DOCS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Legal Research Auditor.
    
    Analyze if the document contains the specific legal information needed to answer the question.
    
    Output JSON format:
    {{
        "score": (int) 1 to 5, where 5 is a perfect match and 1 is irrelevant,
        "reasoning": "Explain concisely what is present or missing."
    }}
    """,
        ),
        ("human", "User Question: {question} \n\n Retrieved Document: {document}"),
    ]
)


GRADE_HALLUCINATIONS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Fact Checker.
    
    Compare the Answer to the Facts.
    
    Output JSON format:
    {{
        "score": (int) 1 to 5 (5 = fully supported, 1 = hallucination),
        "reasoning": "List any claims in the answer that are not in the facts."
    }}
    """,
        ),
        ("human", "Facts: {documents} \n\n Answer: {generation}"),
    ]
)


GRADE_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Quality Control Agent.
    
    Does the answer directly resolve the user's question?
    
    Output JSON format:
    {{
        "score": (int) 1 to 5,
        "reasoning": "Explain if the answer is too vague or misses the point."
    }}
    """,
        ),
        ("human", "User Question: {question} \n\n Answer: {generation}"),
    ]
)
