import logging
import logging.config
import os
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: str = "logs/rag_project.log",
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Setup logging configuration for the RAG project.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to the log file
        format_string: Format string for log messages
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': format_string,
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': level,
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': level,
                'formatter': 'standard',
                'filename': log_file,
                'maxBytes': max_bytes,
                'backupCount': backup_count,
                'encoding': 'utf-8'
            }
        },
        'loggers': {
            'rag_project': {
                'level': level,
                'handlers': ['console', 'file'],
                'propagate': False
            }
        },
        'root': {
            'level': 'WARNING',
            'handlers': ['console']
        }
    }

    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Name of the logger (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"rag_project.{name}")


# Default setup if this module is imported
if not logging.getLogger('rag_project').hasHandlers():
    setup_logging()</content>
<parameter name="filePath">d:\Data Science\DataSense\Code\rag-project2\src\utils\logger.py
