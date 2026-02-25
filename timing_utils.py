"""
Timing Utilities
================

This module provides tools for profiling and performance monitoring.
It includes a singleton `TimingStats` class to aggregate execution times
and a `time_it` decorator to easily measure function performance.
"""

import time
import functools
from collections import defaultdict
from typing import Dict
import threading
import numpy as np

_recursion_depths = defaultdict(int)
_start_times = defaultdict(float)

class TimingStats:
    """
    Singleton class to aggregate timing statistics across the application.

    This class collects execution times for decorated functions, calculating
    statistics like mean, median, and standard deviation per simulation step.
    It is thread-safe.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TimingStats, cls).__new__(cls)
                cls._instance.current_step_times = defaultdict(list)
                cls._instance.call_counts = defaultdict(int)
        return cls._instance
    
    def add_timing(self, func_name: str, execution_time: float):
        """
        Records a timing measurement for a specific function.

        Args:
            func_name (str): The name of the function.
            execution_time (float): The duration of the execution in seconds.
        """
        self.current_step_times[func_name].append(execution_time)
        self.call_counts[func_name] += 1
    
    def get_step_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Computes statistics for the current step.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary mapping function names to
                                         their timing statistics (mean, max, etc.).
        """
        stats_dict = {}
        for func_name, timings in self.current_step_times.items():
            if timings:
                stats_dict[func_name] = {
                    'mean': np.mean(timings),
                    'median': np.median(timings),
                    'std': np.std(timings),
                    'min': np.min(timings),
                    'max': np.max(timings),
                    'calls': self.call_counts[func_name],
                    'total_time': sum(timings)
                }
        return stats_dict
    
    def print_step_report(self):
        """Prints a formatted report of current step timing statistics to stdout."""
        stats = self.get_step_stats()
        sorted_funcs = sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        for func_name, func_stats in sorted_funcs:
            print(f"{func_name:<40} {func_stats['total_time']:<12.4f} {func_stats['calls']:<8} "
                  f"{func_stats['mean']:<10.4f} {func_stats['std']:<10.4f}")
    
    def reset_step(self):
        """Resets all timing statistics for the next simulation step."""
        self.current_step_times.clear()
        self.call_counts.clear()

ENABLE_TIMING = True

def time_it(func):
    """
    Decorator to measure and record function execution time.

    It handles recursion correctly by only timing the outermost call.
    """
    if not ENABLE_TIMING:
        return func
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = func.__qualname__
        _recursion_depths[key] += 1
        
        # Only start timing on first entry
        if _recursion_depths[key] == 1:
            _start_times[key] = time.time()
            
        try:
            result = func(*args, **kwargs)
            
            # Only record time on final exit
            if _recursion_depths[key] == 1:
                execution_time = time.time() - _start_times[key]
                TimingStats().add_timing(key, execution_time)
                
            return result
            
        finally:
            _recursion_depths[key] -= 1
            
    return wrapper