"""Train the model on Zarr dataset."""

import sys
from pathlib import Path

from pytorch_lightning import (
    Callback,
    LightningDataModule,
    Trainer,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader

import wandb

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.zarr_dataset import ZarrNowcastingDataset, create_train_val_split
from dgmr import DGMR

# Add config directory to path and import config utilities
config_path = project_root / "config"
sys.path.insert(0, str(config_path))

try:
    from config_utils import create_config_from_args, print_config
except ImportError:
    print(f"Could not import config_utils from {config_path}")
    print("Please ensure config_utils.py exists in the config directory")
    sys.exit(1)


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Return the wandb logger from the trainer."""
    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning \
            disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """Class for the watch model."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        """Initialize the log frequency and log name."""
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        """Initialize the logger at the start of training."""
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(
            model=trainer.model, log=self.log, log_freq=self.log_freq, log_graph=True
        )


class UploadCheckpointsAsArtifact(Callback):
    """class for logging the checkpoint as artifacts."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        """Initialize the checkpoint directory."""
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        """Run when the user interupts the training by pressing a key on the keyboard."""
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        """Log information about the training of the module at the end of training."""
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

        if self.upload_best_only:
            ckpts.add_file(trainer.checkpoint_callback.best_model_path)
        else:
            for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                ckpts.add_file(str(path))

        experiment.log_artifact(ckpts)


class ZarrDataModule(LightningDataModule):
    """
    LightningDataModule for Zarr dataset.
    """

    def __init__(
        self,
        zarr_path: str,
        input_frames: int = 4,
        target_frames: int = 18,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        """Initialize the data module."""
        super().__init__()

        self.zarr_path = zarr_path
        self.input_frames = input_frames
        self.target_frames = target_frames
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_ratio = train_ratio
        self.seed = seed

        # Will be set in setup()
        self.train_indices = None
        self.val_indices = None

    def setup(self, stage=None):
        """Set up train and validation datasets."""
        # Create train/val split
        self.train_indices, self.val_indices = create_train_val_split(
            self.zarr_path, self.train_ratio, self.seed
        )

        print(f"Train samples: {len(self.train_indices)}")
        print(f"Val samples: {len(self.val_indices)}")

    def train_dataloader(self):
        """Load the training dataset."""
        dataset = ZarrNowcastingDataset(
            zarr_path=self.zarr_path,
            input_frames=self.input_frames,
            target_frames=self.target_frames,
            indices=self.train_indices,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        """Load the validation dataset."""
        dataset = ZarrNowcastingDataset(
            zarr_path=self.zarr_path,
            input_frames=self.input_frames,
            target_frames=self.target_frames,
            indices=self.val_indices,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


def main():
    """Main training function."""
    # Load configuration
    config = create_config_from_args()

    # Print configuration
    print_config(config)

    # Initialize wandb
    wandb.init(project=config.logging.project)
    wandb_logger = WandbLogger(project=config.logging.project)

    # Model checkpoint callback
    model_checkpoint = ModelCheckpoint(
        monitor=config.logging.monitor,
        dirpath=config.logging.checkpoint_dir,
        filename="dgmr-{epoch:02d}-{val_g_loss:.2f}",
        save_top_k=config.logging.save_top_k,
        mode=config.logging.mode,
    )

    # Create trainer
    trainer = Trainer(
        max_epochs=config.training.max_epochs,
        logger=wandb_logger,
        callbacks=[model_checkpoint],
        accelerator=config.hardware.accelerator,
        devices=config.hardware.devices,
        precision=config.hardware.precision,
        log_every_n_steps=config.training.log_every_n_steps,
    )

    # Create model
    model = DGMR(
        forecast_steps=config.model.forecast_steps,
        input_channels=config.model.input_channels,
        output_shape=config.model.output_shape,
        latent_channels=config.model.latent_channels,
        context_channels=config.model.context_channels,
        gen_lr=config.model.gen_lr,
        disc_lr=config.model.disc_lr,
        grid_lambda=config.model.grid_lambda,
    )

    # Create data module
    datamodule = ZarrDataModule(
        zarr_path=config.data.zarr_path,
        input_frames=config.data.input_frames,
        target_frames=config.data.target_frames,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        train_ratio=config.data.train_ratio,
        seed=config.data.seed,
    )

    # Train the model
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
