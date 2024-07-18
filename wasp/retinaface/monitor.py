import pytorch_lightning as pl
from gpumonitor.monitor import GPUStatMonitor


class PyTorchGpuMonitorCallback(pl.Callback):
    def __init__(self, delay=1, display_options=None, log_per_batch=False):
        super().__init__()
        self.delay = delay
        self.display_options = display_options or {}
        self.log_per_batch = log_per_batch
        self.monitor = None

    def _start_monitoring(self):
        if self.monitor is None:
            self.monitor = GPUStatMonitor(self.delay, self.display_options)
            self.monitor.average_stats = []

    def _stop_monitoring(self):
        if self.monitor:
            self.monitor.stop()
            self._log_average_stats()
            self.monitor = None

    def _log_average_stats(self):
        avg_stats = self.monitor.average_stats_per_gpu()
        for gpu_idx, stats in enumerate(avg_stats):
            for key, value in stats.items():
                self._log_metric(f"gpu_{gpu_idx}_{key}", value)

    def _log_metric(self, name, value):
        if logger := self.trainer.logger:
            # Log the metric using the Lightning logger
            logger.log_metrics({name: value}, step=self.trainer.global_step)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.log_per_batch:
            self._start_monitoring()

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        if self.log_per_batch:
            self._stop_monitoring()

    def on_epoch_start(self, trainer, pl_module):
        if not self.log_per_batch:
            self._start_monitoring()

    def on_epoch_end(self, trainer, pl_module):
        if not self.log_per_batch:
            self._stop_monitoring()

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.log_per_batch:
            self._start_monitoring()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        if self.log_per_batch:
            self._stop_monitoring()
