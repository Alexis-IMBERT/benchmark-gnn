Python version : 3.12

Lightning version : 2.4.0

To run :

With [UV](https://docs.astral.sh/uv/)

```bash
uv run hello.py -c config/model/mlp.yaml -c configs/data/fake_dataset.yaml -c configs/trainers/fast_dev_run.yaml -c configs/lr_scheduler/reduce_on_plateau.yaml
```

Or without uv :
Install with :

```bash
pip install -e .
```

And run with :

```bash
python hello.py fit-c config/model/mlp.yaml -c configs/data/fake_dataset.yaml -c configs/trainers/fast_dev_run.yaml -c configs/lr_scheduler/reduce_on_plateau.yaml
```

To get some help :

```bash
python hello.py -h
```

and training help :

```bash
python hello.py fit -h
```

Helpfull ressources for this repo :
    - [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)

Configurations :

- See config folder and choose one file per sub-folder.

Configuration d'un Modèle :
Lightning utilise [jsonargparse](https://jsonargparse.readthedocs.io/en/v4.35.0/#basic-usage) afin d'instancier les classes et facilité la configuration.

Utiliser la classe `models.lightning.GraphLitModel`pour instancier un module Lightning.

Indiquer le module : qui peut être un `torch_geometric.nn.Sequential`.

Si vous souhaitez passer une classe qui ne sera pas instancié, par exemple :
`models.utils.my_global_mean_pool` en paramètre et pas `models.utils.my_global_mean_pool()`il faut créer une fonction qui retourne votre fonction.

Plus concrètement, regarder le fichier de config [gcn.yaml](configs/model/gcn.yaml)

## Results configuration file name
| Model      | ('all-features', 'valid')               | ('all-features', 'test')                | ('no-3d', 'valid')                                                                                | ('no-3d', 'test')                                                                                 | ('one', 'valid')              | ('one', 'test')               |
|:-----------|:----------------------------------------|:----------------------------------------|:--------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------|:------------------------------|:------------------------------|
| GAT        | ['GAT_B64_L1_H64_A8_all-features']      | ['GAT_B64_L1_H64_A8_all-features']      | ['GAT_B64_L1_H128_A8_no-3d']                                                                      | ['GAT_B64_L1_H128_A8_no-3d']                                                                      | ['GAT_B64_L2_H64_A1_one']     | ['GAT_B64_L2_H64_A1_one']     |
| GATv2      | ['GATv2_B32_L1_H64_A8_all-features']    | ['GATv2_B32_L1_H64_A8_all-features']    | ['GATv2_B64_L1_H32_A8_no-3d']                                                                     | ['GATv2_B64_L1_H32_A8_no-3d']                                                                     | ['GATv2_B64_L2_H64_A8_one']   | ['GATv2_B64_L2_H64_A8_one']   |
| GCN        | ['GCN_B32_L8_H16_all-features']         | ['GCN_B32_L8_H16_all-features']         | ['GCN_B64_L8_H64_no-3d']                                                                          | ['GCN_B64_L8_H64_no-3d']                                                                          | ['GCN_B64_L2_H128_one']       | ['GCN_B64_L2_H128_one']       |
| GIN        | ['GIN_B32_L1_H16_all-features']         | ['GIN_B32_L1_H16_all-features']         | ['GIN_B32_L1_H64_no-3d', 'GIN_B32_L1_H16_no-3d', 'GIN_B32_L1_H128_no-3d', 'GIN_B32_L1_H32_no-3d'] | ['GIN_B32_L1_H64_no-3d', 'GIN_B32_L1_H16_no-3d', 'GIN_B32_L1_H128_no-3d', 'GIN_B32_L1_H32_no-3d'] |                               |                               |
| GrapheSAGE | ['GrapheSAGE_B32_L8_H128_all-features'] | ['GrapheSAGE_B32_L8_H128_all-features'] | ['GrapheSAGE_B32_L2_H128_no-3d']                                                                  | ['GrapheSAGE_B32_L2_H128_no-3d']                                                                  | ['GrapheSAGE_B32_L2_H64_one'] | ['GrapheSAGE_B32_L2_H64_one'] |
| MLP        |                                         |                                         |                                                                                                   |                                                                                                   | ['MLP_B32_L2_H128_one']       | ['MLP_B32_L2_H128_one']       |
