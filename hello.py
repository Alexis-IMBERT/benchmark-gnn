"""LightningCLI example script."""

from datetime import datetime
from pytorch_lightning.cli import LightningCLI


if __name__ == "__main__":
    cli = LightningCLI(
        save_config_kwargs={
            "save_to_log_dir": True,
            "config_filename": f"config-{datetime.now()}.yaml",
        },
        parser_kwargs={"parser_mode": "omegaconf"},
    )
