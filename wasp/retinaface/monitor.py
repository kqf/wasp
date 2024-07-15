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

    def _stop_monitoring(self):
        if self.monitor:
            self.monitor.stop()
            print("")
            self.monitor.display_average_stats_per_gpu()
            self.monitor = None

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        if self.log_per_batch:
            self._start_monitoring()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if self.log_per_batch:
            self._stop_monitoring()

    def on_epoch_start(self, trainer, pl_module):
        if not self.log_per_batch:
            self._start_monitoring()

    def on_epoch_end(self, trainer, pl_module):
        if not self.log_per_batch:
            self._stop_monitoring()
