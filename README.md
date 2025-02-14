# Benchmark GNN

This project provides a benchmarking framework for Graph Neural Networks (GNNs) using PyTorch Lightning and UV. It allows you to easily configure and run experiments with different models and datasets.

## Requirements

- Python version: 3.12
- Lightning version: 2.4.0

## Installation

### With UV

Install UV by following the instructions [here](https://docs.astral.sh/uv/).

### Without UV

Install the project dependencies with:

```bash
uv run
```

Or without uv :
Install with :

```bash
pip install -e .
```

## Running Experiments

### With UV

To run an experiment with UV, use the following command:

```bash
uv run hello.py -c config/model/mlp.yaml -c configs/data/fake_dataset.yaml -c configs/trainers/fast_dev_run.yaml -c configs/lr_scheduler/reduce_on_plateau.yaml
```

### Without UV

To run an experiment without UV, use the following command:

```bash
python hello.py fit -c config/model/mlp.yaml -c configs/data/fake_dataset.yaml -c configs/trainers/fast_dev_run.yaml -c configs/lr_scheduler/reduce_on_plateau.yaml
```

## Getting Help

To get general help:

```bash
python hello.py -h
```

To get help on training options:

```bash
python hello.py fit -h
```

## Helpful Resources

- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)

## Configurations

- See the `config` folder and choose one file per sub-folder.

### Model Configuration

Lightning uses [jsonargparse](https://jsonargparse.readthedocs.io/en/v4.35.0/#basic-usage) to instantiate classes and facilitate configuration.

Use the class `models.lightning.GraphLitModel` to instantiate a Lightning module.

Specify the module, which can be a `torch_geometric.nn.Sequential`.

If you want to pass a class that will not be instantiated, for example: `models.utils.my_global_mean_pool` as a parameter and not `models.utils.my_global_mean_pool()`, you need to create a function that returns your function.

For more details, see the config file [gcn.yaml](configs/model/gcn.yaml).

## Cross-Validation

Cross-validation is a technique used to evaluate the performance of a model by partitioning the data into subsets, training the model on some subsets, and validating it on the remaining subsets. This helps in assessing how the model generalizes to an independent dataset.

To perform cross-validation, use the `cross_validation.py` script. Below are example usages of the script with and without UV (a hypothetical tool or environment):

With UV :

```bash
uv run cross_validation.py --model-folder config/model --data-folder configs/data --configs-files configs/trainers/fast_dev_run.yaml configs/lr_scheduler/reduce_on_plateau.yaml --n_repeat 10 --tmp-dir /tmp/ --n_jobs 2
```

In this example, the `uv run` command is used to execute the `cross_validation.py` script. The parameters are:

- `--model-folder`: Specifies the directory containing the model configuration.
- `--data-folder`: Specifies the directory containing the data configuration.
- `--configs-files`: Lists the configuration files for the trainer and learning rate scheduler.
- `--n_repeat`: Number of times to repeat the cross-validation process.
- `--tmp-dir`: Temporary directory for storing intermediate files.
- `--n_jobs`: Number of parallel jobs to run.

Without UV :

```bash
python cross_validation.py --model-folder config/model --data-folder configs/data --configs-files configs/trainers/fast_dev_run.yaml configs/lr_scheduler/reduce_on_plateau.yaml --n_repeat 10 --tmp-dir /tmp/ --n_jobs 2
```

This example uses the standard Python interpreter to run the `cross_validation.py` script with the same parameters as above.

By using these commands, you can perform cross-validation to evaluate your model's performance and ensure it generalizes well to new data.
