import json
import re
from functools import partial
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from src.config import Config
from src.logger_setup import logger
from src.utils.prompts import (
    REWRITE_PROMPT,
    GRADE_DOCS_PROMPT,
    GRADE_HALLUCINATIONS_PROMPT,
    GRADE_ANSWER_PROMPT,
)

llm = ChatOllama(model=Config.CRITIC_MODEL_NAME, temperature=0, format="json")
llm_text = ChatOllama(model=Config.CRITIC_MODEL_NAME, temperature=0)


def parse_output(ai_message):
    """
    Tries to parse JSON. If fails, uses Regex to find the score.
    Returns a dict with 'score' and 'reasoning'.
    """
    text = ai_message.content
    try:
        return json.loads(text)
    except:
        score_match = re.search(r'"score"\s*:\s*(\d)', text)
        score = int(score_match.group(1)) if score_match else 3

        return {"score": score, "reasoning": "Parsing failed, but score extracted."}


def parse_output_and_log(name, ai_message):
    """
    Parses output and logs the score/reasoning.
    Used for partial application in chains.
    """
    res = parse_output(ai_message)
    logger.info(
        f"--- {name} ---: Score {res.get('score')} | Reasoning: {res.get('reasoning')}"
    )
    return res


question_rewriter = REWRITE_PROMPT | llm_text | StrOutputParser()

retrieval_grader = GRADE_DOCS_PROMPT | llm | parse_output
hallucination_grader = (
    GRADE_HALLUCINATIONS_PROMPT
    | llm
    | partial(parse_output_and_log, "HALLUCINATION GRADER")
)
answer_grader = (
    GRADE_ANSWER_PROMPT | llm | partial(parse_output_and_log, "ANSWER GRADER")
)
