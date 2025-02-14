from typing import Callable

import torch
import torch.nn.functional as F
import torch_geometric.nn
import torch_geometric.utils
import torch_scatter
import torch_sparse
import utils
from torch.nn import Linear
from torch_geometric.nn import (
    DenseGCNConv,
    GCNConv,
    dense_diff_pool,
    global_add_pool,
)
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.utils import remove_self_loops, to_dense_adj, to_dense_batch
from torch_scatter import scatter_add
from torch_sparse import coalesce, spspmm, transpose
from utils import rank
from torch.nn import Parameter
from torch_geometric.nn.inits import uniform


class MyGCNTopKPool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_channels = 64
        output_channels = 2
        self.conv_1 = torch_geometric.nn.GCNConv(
            in_channels=-1,
            out_channels=hidden_channels,
        )
        self.conv_2 = torch_geometric.nn.GCNConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
        )
        self.conv_3 = torch_geometric.nn.GCNConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
        )
        # self.pool = getattr(torch_geometric.nn, pooling)
        self.pool_1 = torch_geometric.nn.TopKPooling(
            in_channels=hidden_channels, ratio=100
        )
        self.pool_2 = torch_geometric.nn.TopKPooling(
            in_channels=hidden_channels, ratio=25
        )

        self.lin_1 = torch.nn.Linear(
            in_features=3 * hidden_channels, out_features=output_channels
        )

    def forward(self, x: torch.Tensor, edge_index, batch: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x, edge_index)
        x = x.relu()
        x, edge_index, edge_attr, batch, perm, score = self.pool_1(
            x, edge_index, batch=batch
        )
        xs = torch_geometric.nn.global_add_pool(x, batch=batch)

        x = self.conv_2(x, edge_index)
        x = x.relu()
        x, edge_index, edge_attr, batch, perm, score = self.pool_2(
            x, edge_index, batch=batch
        )
        xs = torch.cat([torch_geometric.nn.global_add_pool(x, batch=batch), xs], dim=1)

        x = self.conv_3(x, edge_index)
        x = x.relu()
        x = torch.cat([torch_geometric.nn.global_add_pool(x, batch=batch), xs], dim=1)

        x = self.lin_1(x)
        return x


class GNN(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, normalize=False, lin=True
    ):
        super(GNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(out_channels))

    def forward(self, x, adj, mask=None):
        # batch_size, num_nodes, in_channels = x.size()

        for conv_layer, bns_layer in zip(self.convs, self.bns):
            x_conv = conv_layer(x, adj, mask)
            print(x_conv.shape)
            x = bns_layer(F.relu(x_conv))

        return x


class DiffPoolClf(torch.nn.Module):
    """
    Diffpool avec deux convs. Diffpool est entre les deux
    adapté de https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial16/Tutorial16.ipynb
    """

    def __init__(self, input_channels, hidden_channels, num_nodes=5, num_classes=2):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_nodes = num_nodes
        self.num_classes = num_classes

        self.gnn_pool = DenseGCNConv(input_channels, self.num_nodes)

        self.gnn1_embed = DenseGCNConv(input_channels, self.hidden_channels)
        self.gnn2_embed = DenseGCNConv(self.hidden_channels, self.hidden_channels)
        self.pool = dense_diff_pool
        self.lin = Linear(self.hidden_channels, self.num_classes)
        self.config = {
            "hidden_channels": hidden_channels,
            "nb_conv_layers": 2,
            "pooling": self.pool,
            "nb_nodes": num_nodes,
        }

    def reset_parameters(self):
        self.gnn_pool.reset_parameters()
        self.gnn1_embed.reset_parameters()
        self.gnn2_embed.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index, batch):
        x_dense, mask = to_dense_batch(x, batch)
        nb_graphs, _, _ = x_dense.shape

        adj = to_dense_adj(edge_index, batch)

        s = self.gnn_pool(x_dense, adj, mask)
        x_l1 = self.gnn1_embed(x_dense, adj, mask)

        x_p1, adj_p1, _, _ = self.pool(x_l1, adj, s, mask=mask)

        x_l2 = self.gnn2_embed(x_p1, adj_p1)

        batch_reduced = torch.LongTensor(
            sum([[i] * self.num_nodes for i in range(nb_graphs)], [])
        )

        x_l2 = global_add_pool(x_l2.reshape(-1, self.hidden_channels), batch_reduced)
        x_out = self.lin(x_l2)

        # Appliquer softmax pour obtenir des probabilités de classe
        x_out = torch.softmax(x_out, dim=1)
        return x_out


class MIVSPool(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        Score: Callable,
        topk_complement: bool,
        assign_method: Callable,
        reduce_method: str,
        **kwargs,
    ):
        super(MIVSPool, self).__init__()

        self.in_channels = in_channels
        self.score = Score(in_channels=in_channels, **kwargs)
        self.topk_complement = topk_complement
        self.assignment = assign_method

        self.reduce_method = utils.aggregation(
            reduce_method,
            in_channels=in_channels,
            out_channels=in_channels,
            num_layers=[256],
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.score.reset_parameters()
        self.reduce_method.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        num_nodes = x.size(0)

        # Score function
        score = self.score(
            x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch
        )

        # Select function
        p = self.mivs(
            score, edge_index, num_nodes, batch, topk_complement=self.topk_complement
        )
        next_num_nodes = p.sum()

        # Assignment function
        edge_index_S, edge_weight_S = self.assignment(p, score, edge_index)

        # Reduce function
        x_scored = (edge_weight_S * x[edge_index_S[0]].t()).t()
        x = self.reduce_method(x_scored, edge_index_S[1])

        # Connect function
        sparse_S = torch_sparse.SparseTensor(
            row=edge_index_S[0],
            col=edge_index_S[1],
            value=torch.ones(edge_index_S.size(1), device=score.device),
        )
        edge_value_S = (
            1.0
            / (
                torch_scatter.segment_csr(
                    sparse_S.storage.value(), sparse_S.storage.rowptr()
                )[edge_index_S[0]]
            )
        )
        edge_index, edge_weight = self.StAS(
            edge_index,
            edge_weight,
            edge_index_S,
            edge_value_S,
            num_nodes,
            next_num_nodes,
        )
        batch = batch[p.reshape(-1)]

        return x, edge_index, edge_weight, batch

    def mivs(self, score, edge_index, num_nodes, batch, topk_complement=False):
        rank_score = rank(score)
        edge_index_nsl, _ = remove_self_loops(
            edge_index=edge_index, edge_attr=torch.as_tensor([1] * edge_index.size(1))
        )

        p = torch.zeros(1, num_nodes, device=score.device).bool()
        q = torch.ones(1, num_nodes, device=score.device).bool()
        while not torch.all(~q):
            score_q = torch.mul(rank_score, q.float())
            sparse_score_q = torch_sparse.SparseTensor(
                row=edge_index_nsl[0],
                col=edge_index_nsl[1],
                value=score_q[0, edge_index_nsl[1]],
            )
            v_max = torch_scatter.segment_csr(
                sparse_score_q.storage.value(),
                sparse_score_q.storage.rowptr(),
                reduce="max",
            )
            v_max = torch.nn.functional.pad(
                input=v_max,
                pad=(0, num_nodes - v_max.size(0)),
                mode="constant",
                value=0,
            )
            p = p | ((v_max < rank_score) & q)
            p_neighbors = torch_sparse.spmm(
                index=edge_index_nsl,
                value=torch.ones(edge_index_nsl.size(1), device=score.device),
                m=num_nodes,
                n=num_nodes,
                matrix=p.t(),
            )
            q = (p_neighbors == 0).view(1, -1) & ~p

        if topk_complement:
            num_nodes_per_graph = scatter_add(
                batch.new_ones(batch.size(0)), batch, dim=0
            )
            num_nodes_selected_per_graph = scatter_add(
                p.int().reshape(-1), batch, dim=0
            )
            num_nodes_desired_per_graph = torch.ceil(
                torch.div(
                    num_nodes_per_graph, torch.as_tensor([2], device=score.device)
                )
            ).long()
            score_p = (score * ~p).reshape(-1)
            score_p_splitted = torch.split(score_p, num_nodes_per_graph.tolist())
            additionnal_survivors = torch.cat(
                [
                    torch.topk(score, max(0, k)).indices + position
                    for (k, score, position) in zip(
                        (num_nodes_desired_per_graph - num_nodes_selected_per_graph),
                        score_p_splitted,
                        [0] + num_nodes_per_graph.cumsum(dim=0).tolist()[:-1],
                    )
                ]
            )

            p[:, additionnal_survivors] = True

        return p

    @staticmethod
    def StAS(index_A, value_A, index_S, value_S, num_nodes, next_num_nodes):
        index_A, value_A = coalesce(index_A, value_A, m=num_nodes, n=num_nodes)
        index_S, value_S = coalesce(index_S, value_S, m=num_nodes, n=next_num_nodes)
        index_B, value_B = spspmm(
            index_A, value_A, index_S, value_S, num_nodes, num_nodes, next_num_nodes
        )

        index_St, value_St = transpose(index_S, value_S, num_nodes, next_num_nodes)
        index_B, value_B = coalesce(index_B, value_B, m=num_nodes, n=next_num_nodes)
        index_out, value_out = spspmm(
            index_St,
            value_St,
            index_B,
            value_B,
            next_num_nodes,
            num_nodes,
            next_num_nodes,
        )

        return index_out, value_out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channels},{self.score.__class__.__name__},{self.topk_complement},"
            f"{self.assignment.__name__})"
        )


EPS = 1e-7


class GNNCosMulti(torch.nn.Module):
    def __init__(self, in_channels, GNN: Callable = GCNConv, nb_proj=4, **kwargs):
        super(GNNCosMulti, self).__init__()
        self.in_channels = in_channels
        self.gnn = GNN(in_channels, in_channels, **kwargs)
        self.weight = Parameter(torch.Tensor(in_channels, nb_proj))
        self.nb_proj = nb_proj
        self.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        gnn = self.gnn(x, edge_index, edge_weight=edge_weight)
        score = torch.mm(gnn, self.weight)
        gnn_norm = gnn.norm(p=2, dim=1).unsqueeze(1).repeat(1, self.nb_proj)
        w_norm = self.weight.norm(p=2, dim=0)
        score_norm = score / (w_norm * gnn_norm + EPS)

        return score_norm.abs().max(1)[0] + EPS

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)
        self.gnn.reset_parameters()


def winner_take_it_all(p, score, edge_index):
    rank_score = rank(score)

    edge_index_nsl, _ = remove_self_loops(edge_index=edge_index)
    num_nodes = p.shape[1]

    # 1) We dissociate survivors and non-survivors with p
    arange = torch.arange(num_nodes, device=score.device)
    survivors = arange[p[0]]
    non_survivors = arange[~p[0]]

    # 2) We assign survivors to themselves
    surv_assign = torch.stack([survivors, survivors], dim=0)

    # 3) We assign non-survivors to survivors
    select = ~p[0][edge_index_nsl[0]] * p[0][edge_index_nsl[1]]
    edge_index_select = edge_index_nsl[:, select]

    sparse_src_non_surv_nsl = torch_sparse.SparseTensor(
        row=edge_index_select[0].unique(return_inverse=True)[1],
        col=edge_index_select[1],
        value=rank_score[edge_index_select[1]],
    )

    _, arg = torch_scatter.segment_max_csr(
        sparse_src_non_surv_nsl.storage.value(),
        sparse_src_non_surv_nsl.storage.rowptr(),
    )
    non_surv_assign = torch.stack(
        [non_survivors, sparse_src_non_surv_nsl.storage.col()[arg]], dim=0
    )

    # 4) We create the matrix S
    edge_index_S = torch.cat([surv_assign, non_surv_assign], dim=1)
    _, edge_index_S[1] = edge_index_S[1].unique(return_inverse=True)

    edge_weight_S = score[edge_index_S[0]]

    edge_weight_S_denominator = torch_scatter.scatter_add(
        edge_weight_S, edge_index_S[1]
    )[edge_index_S[1]]

    edge_weight_S = edge_weight_S / edge_weight_S_denominator

    return edge_index_S, edge_weight_S


class Net(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        nb_blocks: int,
        hidden_size: int,
        dropout: float,
        args_pool: dict,
    ):
        super(Net, self).__init__()
        self.nb_blocks = nb_blocks
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.conv1 = GCNConv(-1, self.hidden_size)
        self.pool1 = MIVSPool(self.hidden_size, **args_pool)
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        for _ in range(self.nb_blocks - 1):
            self.convs.append(GCNConv(self.hidden_size, self.hidden_size))
            self.pools.append(MIVSPool(self.hidden_size, **args_pool))

        self.lin1 = Linear(2 * self.nb_blocks * self.hidden_size, self.hidden_size)
        self.lin2 = Linear(self.hidden_size, self.hidden_size // 2)
        self.lin3 = Linear(self.hidden_size // 2, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.pool1.reset_parameters()

        for conv, pool in zip(self.convs, self.pools):
            conv.reset_parameters()
            pool.reset_parameters()

        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight, batch = (
            data.x,
            data.edge_index,
            data.edge_weight,
            data.batch,
        )

        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x, edge_index, edge_weight, batch = self.pool1(
            x, edge_index, edge_weight=edge_weight, batch=batch
        )
        xs = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        for conv, pool in zip(self.convs, self.pools):
            x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
            x, edge_index, edge_weight, batch = pool(
                x, edge_index, edge_weight=edge_weight, batch=batch
            )
            xs2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            xs = torch.cat([xs, xs2], dim=1)

        x = F.relu(self.lin1(xs))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        return F.log_softmax(self.lin3(x), dim=-1)
