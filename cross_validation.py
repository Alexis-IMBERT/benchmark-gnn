#!/usr/bin/env python3
import argparse
from pathlib import Path
import tempfile
import subprocess
from itertools import product
from joblib import Parallel, delayed

from tqdm.auto import tqdm


def get_latest_checkpoint_path(
    checkpoint_dir=Path("checkpoints"),
):
    checkpoint_files = list(checkpoint_dir.rglob("*.ckpt"))
    if not checkpoint_files:
        return None
    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    return latest_checkpoint


def process_combination(
    config_file_model: Path,
    config_file_data: Path,
    i: int,
    others: list[Path],
    tmp_dir: Path,
):
    temp_dir = tempfile.mkdtemp(dir=tmp_dir)
    tqdm.write(f"Processing {config_file_model} {config_file_data} {i} - {temp_dir}")
    model_content = config_file_model.read_text()
    data_content = config_file_data.read_text()

    others_files_content = "\n".join([other_file.read_text() for other_file in others])

    temp_file_path = Path(temp_dir) / "config.yaml"
    temp_file_path.write_text(
        f"{model_content}\n{data_content}\n{others_files_content}".replace(
            "[[temp_dir]]", str(temp_dir)
        ).replace("[[features]]", config_file_data.stem.split("_")[-1]),
        encoding="utf-8",
    )

    subprocess.run(
        [
            "uv",
            "run",
            "hello.py",
            "fit",
            "-c",
            temp_file_path,
            "--seed_everything",
            str(i),
        ],
        check=False,
    )

    latest_checkpoint_path = get_latest_checkpoint_path(checkpoint_dir=Path(temp_dir))
    if latest_checkpoint_path:
        print(f"Latest checkpoint: {latest_checkpoint_path}")
        subprocess.run(
            [
                "uv",
                "run",
                "hello.py",
                "test",
                "-c",
                temp_file_path,
                "--seed_everything",
                str(i),
                "--ckpt_path",
                str(latest_checkpoint_path),
            ],
            check=False,
        )
    else:
        print("No checkpoints found.")


def main(args):
    others: list[Path] = args.configs_files
    list_model = list(args.model_folder.glob("*.yaml"))
    list_data = list(args.data_folder.glob("*.yaml"))
    list_repeat = list(range(args.n_repeat))

    Parallel(n_jobs=args.n_jobs)(
        delayed(process_combination)(
            config_file_model, config_file_data, i, others, args.tmp_dir
        )
        for config_file_model, config_file_data, i in tqdm(
            product(list_model, list_data, list_repeat),
            total=len(list_model) * len(list_data) * len(list_repeat),
            leave=True,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de base avec des arguments")
    parser.add_argument(
        "--model-folder",
        type=Path,
        required=True,
        help="Model configs folder",
    )
    parser.add_argument(
        "--data-folder",
        type=Path,
        required=True,
        help="Datasets configs folder",
    )
    parser.add_argument(
        "--configs-files",
        type=Path,
        nargs="*",
        required=False,
        help="Configs files",
    )
    parser.add_argument(
        "--n_repeat",
        type=int,
        default=10,
        help="Number of repeat",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default="/tmp/",
        help="Temporary directory",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=2,
        help="Number of parallel jobs",
    )

    args = parser.parse_args()
    main(args)
