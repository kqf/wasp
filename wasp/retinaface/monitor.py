import pytorch_lightning as pl
from gpumonitor.monitor import GPUStatMonitor

REPORT_FIELDS = {
    "memory.total",
    "memory.used",
}


class PyTorchGpuMonitorCallback(pl.Callback):
    """
    {'index': 0,
    'uuid': 'GPU-de8108cd-fb0b-b55f-d8a5-66d44d674b6a',
    'name': 'Tesla T4',
    'memory.total': 15360,
    'memory.used': 1269,
    'memory_free': 14090,
    'memory_available': 14090,
    'temperature.gpu': 36,
    'fan.speed': None,
    'utilization.gpu': 27,
    'power.draw': 35,
    'enforced.power.limit': 70,
    'processes': [{'username': 'ubuntu',
    'command': 'python3.10',
    'full_command': [/bin/python3.10',
        '/bin/pytest',
        '-s',
        'train.py'],
    'gpu_memory_usage': 869,
    'cpu_percent': 99.4,
    'cpu_memory_usage': 2009829376,
    'pid': 9493}]}
    """

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

    def _stop_monitoring(self, trainer):
        if self.monitor:
            self.monitor.stop()
            self._log_average_stats(trainer)
            self.monitor = None

    def _log_average_stats(self, trainer):
        total_memory_total = 0
        total_memory_used = 0
        num_gpus = len(self.monitor.average_stats)

        for stats in self.monitor.average_stats:
            statistics = stats.jsonify()
            name = f"{statistics['name']}:{statistics['index']}"
            memory_total = statistics.get("memory.total")
            memory_used = statistics.get("memory.used")
            if memory_total and memory_used:
                # for key, value in statistics.items():
                #     if key in REPORT_FIELDS:
                #         # self._log_metric(trainer, f"{name}:{key}", value)
                #         pass

                # Aggregate totals for averages
                total_memory_total += memory_total
                total_memory_used += memory_used

                # Calculate and log memory fraction per GPU
                memory_frac = memory_used / memory_total
                self._log_metric(trainer, f"{name}:memory.frac", memory_frac)

        if num_gpus <= 0:
            return

        # Calculate and log averages across all GPUs
        avg_memory_total = total_memory_total / num_gpus
        avg_memory_used = total_memory_used / num_gpus
        avg_memory_frac = avg_memory_used / avg_memory_total

        self._log_metric(trainer, "average:number.gpus", num_gpus)
        self._log_metric(trainer, "average:memory.total", avg_memory_total)
        self._log_metric(trainer, "average:memory.used", avg_memory_used)
        self._log_metric(trainer, "average:memory.frac", avg_memory_frac)

    def _log_metric(self, trainer, name, value):
        if logger := trainer.logger:
            # Log the metric using the Lightning logger
            logger.log_metrics(
                {name.replace(":", "/"): value},
                step=trainer.global_step,
            )

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
            self._stop_monitoring(trainer)

    def on_epoch_start(self, trainer, pl_module):
        if not self.log_per_batch:
            self._start_monitoring()

    def on_epoch_end(self, trainer, pl_module):
        if not self.log_per_batch:
            self._stop_monitoring(trainer)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.log_per_batch:
            self._start_monitoring()

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        if self.log_per_batch:
            self._stop_monitoring(trainer)
