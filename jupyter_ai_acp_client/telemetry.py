"""
Telemetry module for the Jupyter AI ACP Client extension.

Provides structured event schemas and emit helpers for observability
via Jupyter Server's built-in EventLogger system. All telemetry is
purely operational metadata — no customer content or PII is emitted.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jupyter_events import EventLogger

logger = logging.getLogger(__name__)

# Schema IDs
_SERVER_INIT_SCHEMA_ID = (
    "https://jupyter-ai.amazon.com/jupyter_ai_acp_client/events/acp_server_init"
)
_SESSION_INIT_SCHEMA_ID = (
    "https://jupyter-ai.amazon.com/jupyter_ai_acp_client/events/acp_session_init"
)
_CHAT_MESSAGE_SCHEMA_ID = (
    "https://jupyter-ai/jupyter_ai_acp_client/events/acp_chat_message"
)

ACP_SERVER_INIT_SCHEMA: dict = {
    "$id": _SERVER_INIT_SCHEMA_ID,
    "version": "1",
    "title": "ACP Server Initialization Event",
    "description": (
        "Emitted when an ACP server (agent subprocess + client) "
        "initializes or fails to initialize."
    ),
    "type": "object",
    "properties": {
        "event_type": {
            "type": "string",
            "const": "acp_server_init",
        },
        "persona_class": {
            "type": "string",
        },
        "outcome": {
            "type": "string",
            "enum": ["success", "failure"],
        },
        "error_message": {
            "type": ["string", "null"],
        },
    },
    "required": ["event_type", "persona_class", "outcome"],
    "additionalProperties": False,
}


ACP_SESSION_INIT_SCHEMA: dict = {
    "$id": _SESSION_INIT_SCHEMA_ID,
    "version": "1",
    "title": "ACP Session Initialization Event",
    "description": (
        "Emitted when an ACP session is created or loaded, "
        "or fails to do so."
    ),
    "type": "object",
    "properties": {
        "event_type": {
            "type": "string",
            "const": "acp_session_init",
        },
        "persona_class": {
            "type": "string",
        },
        "session_id": {
            "type": ["string", "null"],
        },
        "operation": {
            "type": "string",
            "enum": ["new", "load"],
        },
        "outcome": {
            "type": "string",
            "enum": ["success", "failure"],
        },
        "error_message": {
            "type": ["string", "null"],
        },
    },
    "required": ["event_type", "persona_class", "operation", "outcome"],
    "additionalProperties": False,
}


ACP_CHAT_MESSAGE_SCHEMA: dict = {
    "$id": _CHAT_MESSAGE_SCHEMA_ID,
    "version": "1",
    "title": "ACP Chat Message Event",
    "description": (
        "Emitted when a user sends a message to an ACP-enabled persona. "
        "No message content or PII is recorded."
    ),
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "const": "acp_chat_message",
        },
        "persona_class": {
            "type": "string",
        },
        "session_id": {
            "type": ["string", "null"],
        },
    },
    "required": ["operation", "persona_class", "session_id"],
    "additionalProperties": False,
}


def register_telemetry_schemas(event_logger: EventLogger) -> None:
    """Register all telemetry event schemas with the EventLogger.

    Safe to call — logs errors but does not raise.
    """
    try:
        event_logger.register_event_schema(ACP_SERVER_INIT_SCHEMA)
        event_logger.register_event_schema(ACP_SESSION_INIT_SCHEMA)
        event_logger.register_event_schema(ACP_CHAT_MESSAGE_SCHEMA)
    except Exception:
        logger.error("Failed to register telemetry event schemas.", exc_info=True)


def emit_server_init_event(
    event_logger: EventLogger | None,
    persona_class: str,
    outcome: str,
    error_message: str | None = None,
) -> None:
    """Emit an ACP server initialization event.

    No-op if event_logger is None.
    """
    if event_logger is None:
        return
    try:
        event_logger.emit(
            schema_id=_SERVER_INIT_SCHEMA_ID,
            data={
                "event_type": "acp_server_init",
                "persona_class": persona_class,
                "outcome": outcome,
                "error_message": error_message,
            },
        )
    except Exception:
        logger.warning("Failed to emit server init telemetry event.", exc_info=True)


def emit_session_init_event(
    event_logger: EventLogger | None,
    persona_class: str,
    session_id: str | None,
    operation: str,
    outcome: str,
    error_message: str | None = None,
) -> None:
    """Emit an ACP session initialization event.

    No-op if event_logger is None.
    """
    if event_logger is None:
        return
    try:
        event_logger.emit(
            schema_id=_SESSION_INIT_SCHEMA_ID,
            data={
                "event_type": "acp_session_init",
                "persona_class": persona_class,
                "session_id": session_id,
                "operation": operation,
                "outcome": outcome,
                "error_message": error_message,
            },
        )
    except Exception:
        logger.warning("Failed to emit session init telemetry event.", exc_info=True)


def emit_chat_message_event(
    event_logger: EventLogger | None,
    persona_class: str,
    session_id: str | None,
) -> None:
    """Emit an ACP chat message event.

    No-op if event_logger is None.
    """
    if event_logger is None:
        return
    try:
        event_logger.emit(
            schema_id=_CHAT_MESSAGE_SCHEMA_ID,
            data={
                "operation": "acp_chat_message",
                "persona_class": persona_class,
                "session_id": session_id,
            },
        )
    except Exception:
        logger.warning("Failed to emit chat message telemetry event.", exc_info=True)
