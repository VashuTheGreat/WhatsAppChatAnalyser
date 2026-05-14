import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from constants import LOGS_DIR


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_FOLDER_SIZE = 2 * 1024 * 1024  # 2MB
MAX_LOG_SIZE = 5 * 1024 * 1024    # This is still needed for RotatingFileHandler if a single run exceeds 5MB

log_file_path = os.path.join(LOGS_DIR, LOG_FILE)


def configure_logger():
    # Ensure logs directory exists
    os.makedirs(LOGS_DIR, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    # File Handler
    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=3)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Avoid duplicate handlers if the logger is re-initialized
    logger.handlers.clear()

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

configure_logger()
logging.info("Logger has started")    


