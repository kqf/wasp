import statistics
import time
from contextlib import contextmanager


class Timer:
    def __init__(self, label="timer"):
        self.times = []
        self.label = label

    def mean(self):
        return statistics.mean(self.times) if self.times else 0

    def std(self):
        return statistics.stdev(self.times) if len(self.times) > 1 else 0

    @contextmanager
    def __call__(self, label=""):
        self.label = label or self.label
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.times.append(duration)

    def __str__(self):
        return f"{self.label}: {self.mean():.4f}  +/- {self.std():.4f} seconds"

