import pyrootutils
import dotenv
import os
import shutil
import torch
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
dotenv.load_dotenv(dotenv_path=root.parent.parent / '.env', override=True)
assert os.getenv('DATA_ROOT')
assert os.path.isdir(os.getenv('DATA_ROOT'))
import time

LOG_ROOT = str(root.parent / 'experiments')
os.environ.update({'LOG_ROOT': LOG_ROOT})
os.environ.update({'PROJECT_TASK': root.stem})
os.environ.update({'REPO_ROOT': str(root.parent)})

from typing import List, Optional, Tuple
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.strategies.ddp import DDPStrategy
from src.general_utils import os_utils
from src.utils import option_parsing
from src.utils.hydra_utils import instantiate_callbacks, instantiate_loggers, log_hyperparameters


def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    print(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)()

    print(f"Instantiating model <{cfg.trainer._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.trainer)()

    print("Instantiating callbacks...")
    print(cfg.callbacks)
    callbacks: List[Callback] = instantiate_callbacks(cfg.callbacks)

    print("Instantiating loggers...")
    logger: List[LightningLoggerBase] = instantiate_loggers(cfg.logger)

    print(f"Instantiating trainer <{cfg.lightning._target_}>")
    if cfg.lightning.strategy == 'ddp':
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = None
    trainer: Trainer = hydra.utils.instantiate(cfg.lightning, callbacks=callbacks, logger=logger, strategy=strategy)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        print("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        print("Starting training!")
        if cfg.get("ckpt_path"):
            print('continuing from ', cfg.get("ckpt_path"))
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        print("Starting testing!")
        if cfg.get("ckpt_path") and not cfg.get("train"):
            print("Using predefined ckpt_path", cfg.get('ckpt_path'))
            ckpt_path = cfg.get("ckpt_path")
        elif cfg.get('trainer')['ckpt_path'] and not cfg.get("train"):
            print('Model weight will be loaded during Making the Model')
            ckpt_path = None
        else:
            ckpt_path = trainer.checkpoint_callback.best_model_path

        if ckpt_path == "":
            print("Best ckpt not found! Using current weights for testing...")
            raise ValueError('')
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        print(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path=root / "src/configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    # fix hydra path
    print(f"tmp directory : {cfg.paths.output_dir}")
    if 'experiments' in cfg.paths.output_dir:
        print(f"removing tmp directory : {cfg.paths.output_dir}")
        shutil.rmtree(cfg.paths.output_dir, ignore_errors=True)
    run_name = os_utils.make_runname(cfg.prefix)
    task = os.path.basename(cfg.paths.project_task)
    exp_root = os.path.dirname(cfg.paths.output_dir)
    output_dir = os_utils.make_output_dir(exp_root, task, run_name)
    os.makedirs(output_dir, exist_ok=True)
    cfg.paths.output_dir = output_dir
    cfg.paths.log_dir = output_dir
    if cfg.logger is not None:
        cfg.logger.wandb.name = os.path.basename(cfg.paths.output_dir)
    print(f"Current working directory : {os.getcwd()}")
    print(f"Saving Directory          : {cfg.paths.output_dir}")
    available_gpus = torch.cuda.device_count()
    print("available_gpus------------", available_gpus)
    cfg.datamodule.batch_size = int(cfg.datamodule.total_gpu_batch_size / available_gpus)
    print('Per GPU batchsize:', cfg.datamodule.batch_size)
    time.sleep(1)

    cfg = option_parsing.post_process(cfg)

    # train the model
    metric_dict, _ = train(cfg)
    print(metric_dict)


if __name__ == "__main__":
    main()
