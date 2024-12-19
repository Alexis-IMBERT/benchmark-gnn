Python version : 3.12

To run :

With [UV](https://docs.astral.sh/uv/)

```bash
uv run hello.py -c config.yaml
```

Or without uv :
Install with :

```bash
pip install -e .
```

And run with :

```bash
python hello.py fit -c config.yaml
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
