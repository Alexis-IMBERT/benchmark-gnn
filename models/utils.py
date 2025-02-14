import torch
import torch_geometric


def my_global_mean_pool() -> torch.nn.Module:
    """Return the global mean pooling function from torch_geometric.nn
    Used for the YAML instanciation
    """
    return torch_geometric.nn.global_mean_pool


def my_global_max_pool() -> torch.nn.Module:
    """Return the global max pooling function from torch_geometric.nn
    Used for the YAML instanciation
    """
    return torch_geometric.nn.global_max_pool


def my_global_add_pool() -> torch.nn.Module:
    """Return the global add pooling function from torch_geometric.nn
    Used for the YAML instanciation
    """
    return torch_geometric.nn.global_add_pool
