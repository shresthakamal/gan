import sys

from loguru import logger

# Specifying Logger Format
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line})</cyan> - <level>{message}</level>"

# prevent loguru from printing the logs in the console
logger.remove()


# Sink Sepcifications

logger.add(sys.stderr, format=log_format, backtrace=True, diagnose=True)

logger.add(
    "logs/{time}.log",
    format=log_format,
    rotation="5 MB",
    retention="120 Days",
    backtrace=True,
    diagnose=True,
)
