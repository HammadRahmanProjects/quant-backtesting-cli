import logging
import os
from logging.handlers import RotatingFileHandler

from rich.logging import RichHandler

LOG_DIR  = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

MAX_BYTES    = 5 * 1024 * 1024
BACKUP_COUNT = 3

def setup_logging(level: int = logging.DEBUG) -> None:
    """
    Call this once at application startup in main.py.

    Sets up two handlers:
        1. RichHandler  — writes INFO+ to the terminal with Rich formatting
                          (blends cleanly with the rest of the Rich UI)
        2. RotatingFileHandler — writes DEBUG+ to logs/app.log
                                 rotates at 5MB, keeps last 3 files

    Every module then does:
        import logging
        logger = logging.getLogger(__name__)

    And automatically inherits this configuration.
    """

    os.makedirs(LOG_DIR, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    rich_handler = RichHandler(
        level=logging.INFO,
        show_time=False,       
        show_path=False,      
        rich_tracebacks=True,  
        markup=True,
    )
    rich_handler.setFormatter(logging.Formatter("%(message)s"))

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT))

    for noisy_lib in ("yfinance", "urllib3", "requests", "peewee", "asyncio"):
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)

    root_logger.addHandler(rich_handler)
    root_logger.addHandler(file_handler)

    logging.info("Logging initialized — file: %s", LOG_FILE)