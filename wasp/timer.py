import statistics
import time
from contextlib import contextmanager


class Timer:
    def __init__(self):
        self.times = []

    def mean(self):
        return statistics.mean(self.times) if self.times else 0

    def std(self):
        return statistics.stdev(self.times) if len(self.times) > 1 else 0

    @contextmanager
    def timer(self, label="unnamed"):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.times.append(duration)
