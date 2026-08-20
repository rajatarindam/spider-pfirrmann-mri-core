"""
Project execution logger.

This module provides a simple logger for recording script executions.
All executable scripts should use this logger to generate a log file
inside outputs/logs/.
"""

from datetime import datetime
from pathlib import Path
import time

from configs.paths import LOGS_DIR


class ExecutionLogger:
    """
    Simple execution logger for project scripts.
    """

    def __init__(self, script_name: str):
        self.script_name = script_name
        self.start_time = time.time()
        self.timestamp = datetime.now()

        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        self.log_file = (
            LOGS_DIR
            / f"{self.timestamp.strftime('%Y%m%d_%H%M%S')}_{script_name}.log"
        )

        self.lines = []

    def add(self, key: str, value):
        """Add a log entry."""
        self.lines.append(f"{key}: {value}")

    def warning(self, message: str):
        """Record a warning."""
        self.lines.append(f"WARNING: {message}")

    def error(self, message: str):
        """Record an error."""
        self.lines.append(f"ERROR: {message}")

    def save(self, status: str = "SUCCESS"):
        """Write the execution log to disk."""

        duration = time.time() - self.start_time

        with open(self.log_file, "w", encoding="utf-8") as file:
            file.write(f"Script Name: {self.script_name}\n")
            file.write(
                f"Execution Timestamp: "
                f"{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            file.write(f"Execution Duration: {duration:.2f} seconds\n")
            file.write(f"Completion Status: {status}\n\n")

            for line in self.lines:
                file.write(f"{line}\n")