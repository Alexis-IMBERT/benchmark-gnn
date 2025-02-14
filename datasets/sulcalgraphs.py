from pathlib import Path
import pandas as pd
import torch
import pickle
from torch_geometric.data import InMemoryDataset
from networkx.classes.graph import Graph
from dataclasses import dataclass, field
from tqdm.auto import tqdm
from torch_geometric.utils import from_networkx


@dataclass
class SulcalGraph:
    path: Path
    graph: Graph = field(init=False)
    id: int = field(init=False)
    side: str = field(init=False)

    def __post_init__(self):
        stem_path = str(self.path.stem).split(".")
        self.id = stem_path[0]
        self.side = stem_path[1]
        self.graph = pickle.loads(self.path.read_bytes())


class SulcalGraphs(InMemoryDataset):
    def __init__(
        self,
        root: Path | str,
        split: str,
        *args,
        csv_path: Path | str = None,
        node_attrs: list[str] = None,
        **kwargs,
    ):
        self.node_attrs = node_attrs if node_attrs else None
        self.csv_path = Path(csv_path) if csv_path else self.raw_dir / "data.csv"
        super().__init__(
            root,
            *args,
            **kwargs,
        )
        if split == "train":
            path = self.processed_paths[0]
        elif split == "val":
            path = self.processed_paths[1]
        elif split == "test":
            path = self.processed_paths[2]
        else:
            raise ValueError(
                f"Split '{split}' found, but expected either 'train', 'val', or 'test'"
            )
        self.data, self.slices = torch.load(
            path,
            weights_only=False,
        )  # IMPORTANT

    @property
    def raw_file_names(self) -> list[Path]:
        return list(Path("data/sulcalgraphs").rglob("*.gpickle"))

    @property
    def processed_file_names(self) -> list[str]:
        return ["train_data.pt", "val_data.pt", "test_data.pt"]

    def download(self): ...

    def process(self):
        # load the csv file
        def from_nx_to_pyg(
            graph,
            y,
            person_id=None,
            side=None,
            node_attrs=self.node_attrs,
        ):
            """
            Convert networkx graph to pytorch graph and add y, person_id and side as attributes
            """
            node_attrs = node_attrs if node_attrs else ["vertex_index"]
            for node in graph.nodes(data=True):
                if "vertex_index" in node[1]:
                    node[1]["vertex_index"] = float(node[1]["vertex_index"])
            pyg_graph = from_networkx(
                graph,
                group_node_attrs=node_attrs,
            )
            pyg_graph.y = y
            pyg_graph.person_id = person_id
            pyg_graph.side = side
            return pyg_graph

        targets = pd.read_csv(self.csv_path)

        assert len(targets) > 0, "No data found in csv file"

        data_list = []
        for file in tqdm(self.raw_file_names):
            sulcalgraph: SulcalGraph = SulcalGraph(file)
            y = torch.tensor(
                0
                if targets[targets["Subject"] == int(sulcalgraph.id)]["Gender"].values[
                    0
                ]
                == "M"
                else 1
            )
            data_list.append(
                from_nx_to_pyg(sulcalgraph.graph, y, sulcalgraph.id, sulcalgraph.side)
            )

        assert len(data_list) > 0, "No data found 1"

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        train_size = int(0.8 * len(data_list))
        val_size = int(0.1 * len(data_list))
        test_size = len(data_list) - train_size - val_size

        train_data, val_data, test_data = torch.utils.data.random_split(
            data_list, [train_size, val_size, test_size]
        )

        train_data, train_slices = self.collate(train_data)
        val_data, val_slices = self.collate(val_data)
        test_data, test_slices = self.collate(test_data)

        torch.save((train_data, train_slices), self.processed_paths[0])
        torch.save((val_data, val_slices), self.processed_paths[1])
        torch.save((test_data, test_slices), self.processed_paths[2])
