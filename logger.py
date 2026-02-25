"""
Simulation Logging (Infrastructure)
====================================

Concrete implementations of the `Logger` abstract base class.

Thread-safety design:
    CSVLogger uses a producer/consumer queue with a dedicated
    writer thread so that simulation threads never block on
    file I/O. The daemon thread drains the queue and writes
    rows sequentially, guaranteeing ordered output.

Classes:
    CSVLogger: Thread-safe CSV writer with event-type filtering.
    TensorBoardLogger: Writes scalars to TensorBoard.
    CompositeLogger: Fan-out to multiple loggers.
"""

import csv
import queue
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from torch.utils.tensorboard import SummaryWriter
from framework import Logger

class CSVLogger(Logger):
    """
    Thread-safe CSV logger with event-type filtering.

    Uses a queue + daemon thread pattern so the simulation's
    hot path (scheduler.step) never blocks on disk I/O.
    Each CSVLogger instance owns one file and one writer thread.
    """
    def __init__(self, log_file_path: str, fieldnames: list, allowed_event_types: Optional[List[str]] = None):
        self.log_file_path = log_file_path
        self.fieldnames = fieldnames
        self.allowed_event_types = allowed_event_types
        # Keep a set of fieldnames for faster lookups
        self._fieldname_set = set(fieldnames)

        # Queue-based async write: simulation thread enqueues,
        # dedicated writer thread dequeues and flushes to disk.
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._process_queue, daemon=True)
        self._stop_event = threading.Event()

        # Open the file and write the header
        with open(self.log_file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

        self._thread.start()

    def _process_queue(self):
        """
        The target function for the logger thread. It processes the queue of log entries.
        """
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                # Wait for an item to appear in the queue, with a timeout
                log_data = self._queue.get(timeout=0.1)

                with open(self.log_file_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writerow(log_data)

                self._queue.task_done()
            except queue.Empty:
                # Timeout occurred, loop again to check stop_event
                continue

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Adds a new log entry to the queue, filtering for relevant fields and event types.
        """
        # If this logger has a list of allowed event types, filter for them.
        if self.allowed_event_types and event_type not in self.allowed_event_types:
            return

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
        }
        log_entry.update(data)

        # Filter the log entry to only include fields expected in this CSV.
        filtered_entry = {k: v for k, v in log_entry.items() if k in self._fieldname_set}

        # Do not write an entry if it contains no data for the specified fields
        if not filtered_entry:
            return

        # Fill any missing values with None.
        for field in self.fieldnames:
            if field not in filtered_entry:
                filtered_entry[field] = None
        
        self._queue.put(filtered_entry)


    def close(self):
        """
        Graceful shutdown: drain queue, then signal thread to exit.
        Ensures no log entries are lost on simulation end.
        """
        # Wait for the queue to be empty
        self._queue.join()

        # Signal the thread to stop
        self._stop_event.set()

        # Wait for the thread to terminate
        self._thread.join()

class TensorBoardLogger(Logger):
    """
    A logger that writes simulation statistics to a TensorBoard log file.
    """
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Logs a single event to TensorBoard.
        """
        step = data.get('step')
        if step is None:
            return

        if event_type == 'generation':
            agent_id = data.get('agent_id')
            self.writer.add_scalar(f'Agent_{agent_id}/Novelty', data['novelty'], step)
            self.writer.add_scalar(f'Agent_{agent_id}/Interest', data['interest'], step)

        elif event_type == 'share':
            pass

        elif event_type == 'step_end':
            # Log all system-wide, end-of-step metrics here
            if 'domain_size' in data:
                self.writer.add_scalar('Domain/Size', data['domain_size'], step)

            if 'self_threshold' in data:
                self.writer.add_scalar('Thresholds/Self_Share', data['self_threshold'], step)
            if 'domain_threshold' in data:
                self.writer.add_scalar('Thresholds/Domain_Accept', data['domain_threshold'], step)
            if 'boredom_threshold' in data:
                self.writer.add_scalar('Thresholds/Boredom', data['boredom_threshold'], step)
            
            if 'avg_accepted_interest' in data:
                self.writer.add_scalar('Interactions/Avg_Accepted_Interest', data['avg_accepted_interest'], step)
            if 'avg_rejected_interest' in data:
                self.writer.add_scalar('Interactions/Avg_Rejected_Interest', data['avg_rejected_interest'], step)

            if 'avg_knn_size' in data:
                self.writer.add_scalar('System/Avg_kNN_Memory_Size', data['avg_knn_size'], step)


    def close(self):
        """
        Closes the TensorBoard writer.
        """
        self.writer.close()

class CompositeLogger(Logger):
    """
    A logger that delegates to a list of other loggers.
    """
    def __init__(self, loggers: list[Logger]):
        self.loggers = loggers

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Logs an event to all contained loggers.
        """
        for logger in self.loggers:
            logger.log_event(event_type, data)

    def close(self):
        """
        Closes all contained loggers.
        """
        for logger in self.loggers:
            logger.close()