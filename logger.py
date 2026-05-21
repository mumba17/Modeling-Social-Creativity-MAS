import csv
import queue
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from torch.utils.tensorboard import SummaryWriter
from framework import Logger


class CSVLogger(Logger):
    """
    Thread-safe CSV writer decoupling simulation loops from blocking disk I/O operations.
    """
    def __init__(self, log_file_path: str, fieldnames: List[str], allowed_event_types: Optional[List[str]] = None):
        self.log_file_path = log_file_path
        self.fieldnames = fieldnames
        self.allowed_event_types = allowed_event_types
        
        # Cache field names in a hash set to optimize runtime containment checks
        self._fieldname_set = set(fieldnames)

        # Instantiate synchronization primitives and thread parameters for asynchronous I/O isolation
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._process_queue, daemon=True)
        self._stop_event = threading.Event()

        # Open a persistent file descriptor to eliminate repeated open/close syscall overheads
        self._file = open(self.log_file_path, 'w', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()
        self._file.flush()

        self._thread.start()

    def _process_queue(self):
        """
        Background worker loop extracting logged activities from the buffer queue sequentially.
        """
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                # Extract records sequentially from the queue with an explicit timeout bound
                log_data = self._queue.get(timeout=0.1)
                self._writer.writerow(log_data)
                self._queue.task_done()
            except queue.Empty:
                # Re-evaluate shutdown condition status when timeout constraints are hit
                continue
                
            # Force an explicit disk flush when the log queue is depleted to prevent data loss on crashes
            if self._queue.empty():
                self._file.flush()

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Enqueues an individual structural tracking event, pruning unmapped value properties.
        """
        # Validate event categories against configuration filters before processing fields
        if self.allowed_event_types and event_type not in self.allowed_event_types:
            return

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
        }
        log_entry.update(data)

        # Reconstruct tracking payloads containing only predefined structural attributes
        filtered_entry = {k: v for k, v in log_entry.items() if k in self._fieldname_set}
        if not filtered_entry:
            return

        # Explicitly backfill missing record fields with empty null values
        for field in self.fieldnames:
            if field not in filtered_entry:
                filtered_entry[field] = None
        
        self._queue.put(filtered_entry)

    def log_events_batch(self, event_type: str, data_list: List[Dict[str, Any]]):
        """
        Appends a complete array of events to the log queue using a single lock acquisition pass.
        """
        if self.allowed_event_types and event_type not in self.allowed_event_types:
            return

        timestamp = datetime.now().isoformat()
        filtered_entries = []

        # Parse structural data transformations completely outside critical thread locks
        for data in data_list:
            log_entry = {
                'timestamp': timestamp,
                'event_type': event_type,
            }
            log_entry.update(data)
            
            filtered_entry = {k: v for k, v in log_entry.items() if k in self._fieldname_set}
            if not filtered_entry:
                continue

            for field in self.fieldnames:
                if field not in filtered_entry:
                    filtered_entry[field] = None
            
            filtered_entries.append(filtered_entry)

        if not filtered_entries:
            return

        # Atomically push processed entries into storage blocks to limit thread lock contention
        with self._queue.mutex:
            for entry in filtered_entries:
                self._queue._put(entry)
                self._queue.unfinished_tasks += 1
            self._queue.not_empty.notify()

    def close(self):
        """
        Executes a graceful shutdown sequence, flushing out remaining tracking metrics safely.
        """
        # Block until the current queue structures are fully depleted
        self._queue.join()
        self._stop_event.set()
        self._thread.join()

        # Safely shut down active system file streams
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()


class TensorBoardLogger(Logger):
    """
    Real-time performance metric logger redirecting scalar outputs directly to TensorBoard.
    """
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Maps simulation tracking properties onto scalar TensorBoard visualization indices.
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
            # Extract and update global network status properties across step intervals
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
            if 'avg_current_interest' in data:
                self.writer.add_scalar('System/Avg_Current_Interest', data['avg_current_interest'], step)
            if 'avg_average_interest' in data:
                self.writer.add_scalar('System/Avg_Cumulative_Interest', data['avg_average_interest'], step)
            if 'avg_current_novelty' in data:
                self.writer.add_scalar('System/Avg_Current_Novelty', data['avg_current_novelty'], step)

            if 'accepted_count' in data:
                self.writer.add_scalar('Interactions/Accepted_Count', data['accepted_count'], step)
            if 'rejected_count' in data:
                self.writer.add_scalar('Interactions/Rejected_Count', data['rejected_count'], step)

            if 'total_self_evals' in data:
                self.writer.add_scalar('System/Total_Self_Evals', data['total_self_evals'], step)
            if 'total_other_evals' in data:
                self.writer.add_scalar('System/Total_Other_Evals', data['total_other_evals'], step)
            if 'total_shares' in data:
                self.writer.add_scalar('System/Total_Shares', data['total_shares'], step)
            if 'total_domain_adoptions' in data:
                self.writer.add_scalar('System/Total_Domain_Adoptions', data['total_domain_adoptions'], step)

    def log_events_batch(self, event_type: str, data_list: List[Dict[str, Any]]):
        """
        Unpacks incoming batch structures to record metrics linearly across target histories.
        """
        for data in data_list:
            self.log_event(event_type, data)

    def close(self):
        """
        Flushes and destroys the persistent SummaryWriter event tracking instance.
        """
        self.writer.close()


class CompositeLogger(Logger):
    """
    Structural distribution proxy broadcasting tracking events across multiple downstream loggers.
    """
    def __init__(self, loggers: List[Logger]):
        self.loggers = loggers

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Broadcasts an individual simulation transaction down to all registered logger child nodes.
        """
        for logger in self.loggers:
            logger.log_event(event_type, data)

    def log_events_batch(self, event_type: str, data_list: List[Dict[str, Any]]):
        """
        Routes batched collection logs down to specialized optimized writer pathways.
        """
        for logger in self.loggers:
            if hasattr(logger, 'log_events_batch'):
                logger.log_events_batch(event_type, data_list)
            else:
                for data in data_list:
                    logger.log_event(event_type, data)

    def close(self):
        """
        Triggers resource destruction calls sequentially across dependent logger instances.
        """
        for logger in self.loggers:
            logger.close()