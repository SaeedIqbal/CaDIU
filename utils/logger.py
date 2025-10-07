import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional


class CaDILogger:
    """
    Logger for CaDIU training and unlearning events.
    Supports structured logging to both console and file with timestamped JSON records.
    """
    def __init__(
        self,
        name: str = "CaDIU",
        log_dir: str = "logs",
        level: int = logging.INFO,
        use_file: bool = True,
        use_console: bool = True
    ):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            log_dir: Directory to save log files
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
            use_file: Enable file logging
            use_console: Enable console logging
        """
        self.name = name
        self.log_dir = log_dir
        self.level = level
        
        # Create log directory
        if use_file:
            os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False  # Avoid duplicate logs
        
        # Clear existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # Add console handler
        if use_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_formatter = logging.Formatter(
                '[%(asctime)s] %(name)s - %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Add file handler
        if use_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(log_dir, f"{name.lower()}_{timestamp}.log")
            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                '[%(asctime)s] %(name)s - %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def _log_event(self, level: str, event_type: str, message: str, **kwargs) -> None:
        """Log structured event with metadata."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "metadata": kwargs
        }
        
        # Log as JSON string for file, pretty for console
        if self.logger.isEnabledFor(getattr(logging, level.upper())):
            if level == "info":
                self.logger.info(json.dumps(log_data))
            elif level == "warning":
                self.logger.warning(json.dumps(log_data))
            elif level == "error":
                self.logger.error(json.dumps(log_data))
            elif level == "debug":
                self.logger.debug(json.dumps(log_data))

    def log_training_start(self, task_id: str, instruction: str, dataset_size: int) -> None:
        """Log start of training event."""
        self._log_event(
            "info",
            "TRAINING_START",
            f"Started training on task '{task_id}' with instruction '{instruction}'",
            task_id=task_id,
            instruction=instruction,
            dataset_size=dataset_size
        )

    def log_training_end(self, task_id: str, metrics: Dict[str, float]) -> None:
        """Log end of training event with metrics."""
        self._log_event(
            "info",
            "TRAINING_END",
            f"Completed training on task '{task_id}'",
            task_id=task_id,
            metrics=metrics
        )

    def log_unlearning_start(self, task_id: str) -> None:
        """Log start of unlearning event."""
        self._log_event(
            "info",
            "UNLEARNING_START",
            f"Started unlearning task '{task_id}'",
            task_id=task_id
        )

    def log_unlearning_end(self, task_id: str, success: bool) -> None:
        """Log end of unlearning event."""
        self._log_event(
            "info",
            "UNLEARNING_END",
            f"Completed unlearning task '{task_id}'",
            task_id=task_id,
            success=success
        )

    def log_evaluation(self, task_id: str, metrics: Dict[str, float]) -> None:
        """Log evaluation metrics."""
        self._log_event(
            "info",
            "EVALUATION",
            f"Evaluation metrics for task '{task_id}'",
            task_id=task_id,
            metrics=metrics
        )

    def log_memory_usage(self, memory_mb: float, num_tasks: int) -> None:
        """Log memory usage."""
        self._log_event(
            "info",
            "MEMORY_USAGE",
            f"Current memory usage: {memory_mb:.2f} MB",
            memory_mb=memory_mb,
            num_tasks=num_tasks
        )

    def log_error(self, message: str, **kwargs) -> None:
        """Log error event."""
        self._log_event(
            "error",
            "ERROR",
            message,
            **kwargs
        )

    def log_warning(self, message: str, **kwargs) -> None:
        """Log warning event."""
        self._log_event(
            "warning",
            "WARNING",
            message,
            **kwargs
        )

    def close(self) -> None:
        """Close all handlers."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)