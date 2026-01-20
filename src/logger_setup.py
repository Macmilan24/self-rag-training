import logging
import os
import datetime
import pprint
from src.config import Config

os.makedirs(Config.LOG_DIR, exist_ok=True)


class ReadableFormatter(logging.Formatter):
    def format(self, record):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname
        message = record.getMessage()

        log_str = f"[{timestamp}] [{level}] :: {message}"
        if hasattr(record, "details"):
            details_str = pprint.pformat(record.details, indent=4, width=120)
            indented_details = "\n".join(
                ["\t|  " + line for line in details_str.split("\n")]
            )
            log_str += f"\n{indented_details}"

        return log_str


def setup_logger(name="AuraLogger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(Config.LOG_DIR, f"execution_trace_{timestamp}.txt")

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(ReadableFormatter())

    logger.addHandler(file_handler)

    return logger


logger = setup_logger()
