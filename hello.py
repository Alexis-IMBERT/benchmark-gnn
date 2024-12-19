"""LightningCLI example script."""

from datetime import datetime
from pytorch_lightning.cli import LightningCLI


cli = LightningCLI(
    save_config_kwargs={
        "save_to_log_dir": True,
        "config_filename": f"config-{datetime.now()}.yaml",
    }
)
