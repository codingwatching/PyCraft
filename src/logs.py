import logging
import logging.config
from pathlib import Path

LOG_DIR = Path(__file__).parent / "log"
LOG_DIR.mkdir(exist_ok=True)

def setup_logging(level: int = logging.INFO):
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "[%(levelname)s] %(name)s: %(message)s",
            },
            "debug": {
                "format": "[%(levelname)s] %(name)s: %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": level,
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": str(LOG_DIR / "app.log"),
                "formatter": "debug",
                "level": "DEBUG",
            },
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["console", "file"],
        },
    })


