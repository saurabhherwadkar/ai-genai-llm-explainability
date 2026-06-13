# Logging system setup using structlog
"""Configures structured logging based on application settings."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog

from llm_explainability.config.settings import LoggingConfig
from llm_explainability.logging.processors import (
    add_app_context,
    sanitize_sensitive_data,
)

# Mapping from config level strings to Python logging level constants
LOG_LEVEL_MAP: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "error": logging.ERROR,
}


def setup_logging(config: LoggingConfig) -> None:
    """Configure the logging system based on the provided configuration."""
    # Resolve the numeric log level from the string config value
    log_level = LOG_LEVEL_MAP.get(config.level, logging.INFO)
    # Ensure the log output directory exists
    _ensure_log_directory(config.file_output)
    # Configure standard library logging as structlog's backend
    _configure_stdlib_logging(log_level, config)
    # Build the structlog processor chain based on output format
    processors = _build_processor_chain(config.format)
    # Apply the structlog configuration globally
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Create and return a named logger instance for a module."""
    # Bind the module name to all log entries from this logger
    return structlog.get_logger(name)


def _ensure_log_directory(file_output: str) -> None:
    """Create the log file directory if it does not already exist."""
    # Extract the parent directory from the log file path
    log_dir = Path(file_output).parent
    # Create the directory and any missing parents
    log_dir.mkdir(parents=True, exist_ok=True)


def _configure_stdlib_logging(log_level: int, config: LoggingConfig) -> None:
    """Set up Python's standard logging module as structlog's output backend."""
    # Create a stream handler that writes to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    # Set the handler's log level threshold
    console_handler.setLevel(log_level)
    # Configure the root logger with the handler
    logging.basicConfig(
        format="%(message)s",
        handlers=[console_handler],
        level=log_level,
    )
    # Apply per-module log level overrides from configuration
    for module_name, module_level in config.modules.items():
        # Get the numeric level for this module's configured level
        numeric_level = LOG_LEVEL_MAP.get(module_level, logging.INFO)
        # Set the logger for this specific module
        logging.getLogger(module_name).setLevel(numeric_level)


def _build_processor_chain(output_format: str) -> list[structlog.types.Processor]:
    """Construct the chain of structlog processors based on output format."""
    # Start with common processors applied to all formats
    processors: list[structlog.types.Processor] = [
        # Add timestamp to every log entry
        structlog.processors.TimeStamper(fmt="iso"),
        # Add the logger name to each entry
        structlog.processors.add_log_level,
        # Add application context (version, environment)
        add_app_context,
        # Remove sensitive data from log entries
        sanitize_sensitive_data,
        # Format stack traces for exceptions
        structlog.processors.StackInfoRenderer(),
        # Format exceptions with full traceback
        structlog.processors.format_exc_info,
        # Decode unicode characters properly
        structlog.processors.UnicodeDecoder(),
    ]
    # Add format-specific renderer as the final processor
    if output_format == "json":
        # JSON renderer for production (machine-parseable)
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Console renderer for development (human-readable)
        processors.append(structlog.dev.ConsoleRenderer())
    # Return the complete processor chain
    return processors
