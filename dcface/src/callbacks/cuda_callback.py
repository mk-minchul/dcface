from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
import torch
from pytorch_lightning.utilities import rank_zero_info
import time


class CUDACallback(Callback):

    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        if hasattr(trainer, 'root_gpu'):
            torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
            torch.cuda.synchronize(trainer.root_gpu)
        else:
            torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
            torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs=None):
        if hasattr(trainer, 'root_gpu'):
            root_gpu = trainer.root_gpu
        else:
            root_gpu  = trainer.strategy.root_device.index
        torch.cuda.synchronize(root_gpu)
        max_memory = torch.cuda.max_memory_allocated(root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass
