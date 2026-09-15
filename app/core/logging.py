"""Small JSON logger with explicitly selected, payload-free fields."""

import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime

request_id_context = ContextVar("request_id", default=None)
analysis_id_context = ContextVar("analysis_id", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        fields = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
            "requestId": request_id_context.get(),
            "analysisId": analysis_id_context.get(),
        }
        for key in ("stage", "duration_ms", "error_code", "analysis_id"):
            if hasattr(record, key):
                fields[key] = getattr(record, key)
        return json.dumps(fields, ensure_ascii=False)


def configure_logging():
    logger = logging.getLogger("app")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def log_event(event, **fields):
    logging.getLogger("app").info(event, extra=fields)
