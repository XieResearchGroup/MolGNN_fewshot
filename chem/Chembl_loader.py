import os.path as osp
import multiprocessing as mp
import gzip

import torch
from torch_geometric.data import Dataset, Data

# from chemreader.readers import Smiles
from rdkit import Chem
import numpy as np
from tqdm import tqdm

from .util import get_filtered_fingerprint


class ChemBLFP(Dataset):

    # allowable node and edge features in contextPred
    allowable_features = {
        "possible_atomic_num_list": list(range(1, 119)),
        "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
        "possible_chirality_list": [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER,
        ],
        "possible_hybridization_list": [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED,
        ],
        "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "possible_implicit_valence_list": [0, 1, 2, 3, 4, 5, 6],
        "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "possible_bonds": [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ],
        "possible_aromatic_list": [True, False],
        "possible_bond_dirs": [  # only for double bond stereo information
            Chem.rdchem.BondDir.NONE,
            Chem.rdchem.BondDir.ENDUPRIGHT,
            Chem.rdchem.BondDir.ENDDOWNRIGHT,
        ],
    }

    def __init__(
        self,
        root=None,
        transform=None,
        pre_transform=None,
        n_workers=4,
        atom_feat_format="contextpred",
        scale="full",
    ):
        """ Dataset class for ChemBL dataset.

        Args:
            root (str): path to the dataset root directory
            transform (callable): a callable to transform the data on the fly
            pre_transform (callable): a callable to transform the data during processing
            n_workers (int): number of workers for multiprocessing
            atom_feat_format (str): "contextpred" or "sim_atom_type". The
                "sim_atom_type" format has simpler atom types comparing to "contextpred"
                format.
            scale (str): the scale of the dataset. "filtered" or "full". "full" has
                1785415 chemical compounds. "filtered" has 430709 chemical compounds.
        """
        if root is None:
            root = osp.join("data", "ChemBL")
        self.n_workers = n_workers
        assert atom_feat_format in [
            "contextpred",
            "sim_atom_type",
        ], f"{atom_feat_format} should be in ['contextpred', 'sim_atom_type']"
        assert scale in [
            "full",
            "filtered",
        ], f"{scale} should be in ['full', 'filtered']"
        self.atom_feat_format = atom_feat_format
        self.scale = scale
        super().__init__(root, transform, pre_transform)

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        name = "_".join([self.atom_feat_format, self.scale, "processed"])
        return osp.join(self.root, name)

    @property
    def raw_file_names(self):
        if self.scale == "filtered":
            return ["smiles.csv"]
        else:
            return ["chembl_25.csv.gz"]

    @property
    def processed_file_names(self):
        if self.scale == "filtered":
            return ["data_1.pt", "data_2.pt", "data_430000.pt"]
        else:
            return ["data_1.pt", "data_2.pt", "data_1780000.pt"]

    def download(self):
        """ Get raw data and save to raw directory.
        """
        pass

    def save_data(self, q):
        """ Save graphs in q to data.pt files.
        """
        while 1:
            data = q.get()
            if data == "END":
                break
            graph, label, idx = data
            graph.y = label
            graph.id = idx
            torch.save(graph, osp.join(self.processed_dir, f"data_{idx}.pt"))
            print(
                "graph #{} saved to data_{}.pt{}".format(idx, idx, " " * 40), end="\r"
            )

    def create_graph(self, smi, idx, q):
        from chemreader.readers import Smiles

        try:
            graph = Smiles(smi).to_graph(sparse=True, pyg=True)
        except AttributeError:
            return
        fp = get_filtered_fingerprint(smi)
        label = torch.tensor(list(fp), dtype=torch.long)[None, :]
        q.put((graph, label, idx))

    def _create_contextpred_graph(self, smi, idx, q):
        """
        Converts rdkit mol object to graph Data object required by the pytorch
        geometric package. NB: Uses simplified atom and bond features, and represent
        as indices
        :param mol: rdkit mol object
        :return: graph data object with the attributes: x, edge_index, edge_attr
        """
        # atoms
        # num_atom_features = 6  # atom type,  chirality tag
        atom_features_list = []
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return
        for atom in mol.GetAtoms():
            atom_feature = (
                [
                    self.allowable_features["possible_atomic_num_list"].index(
                        atom.GetAtomicNum()
                    )
                ]
                + [
                    self.allowable_features["possible_degree_list"].index(
                        atom.GetDegree()
                    )
                ]
                + [
                    self.allowable_features["possible_formal_charge_list"].index(
                        atom.GetFormalCharge()
                    )
                ]
                + [
                    self.allowable_features["possible_hybridization_list"].index(
                        atom.GetHybridization()
                    )
                ]
                + [
                    self.allowable_features["possible_aromatic_list"].index(
                        atom.GetIsAromatic()
                    )
                ]
                + [
                    self.allowable_features["possible_chirality_list"].index(
                        atom.GetChiralTag()
                    )
                ]
            )
            atom_features_list.append(atom_feature)
        x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

        # bonds
        num_bond_features = 2  # bond type, bond direction
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_feature = [
                    self.allowable_features["possible_bonds"].index(bond.GetBondType())
                ] + [
                    self.allowable_features["possible_bond_dirs"].index(
                        bond.GetBondDir()
                    )
                ]
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format, shape [2, num_edges]
            edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

            # data.edge_attr: Edge feature matrix, shape [num_edges, num_edge_features]
            edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
        else:  # mol has no bonds
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        fp = get_filtered_fingerprint(smi)
        label = torch.tensor(list(fp), dtype=torch.long)[None, :]

        return q.put((data, label, idx))

    def process(self):
        """ The method converting SMILES and labels to graphs.
        """
        # init Queue
        manager = mp.Manager()
        q = manager.Queue(maxsize=self.n_workers * 2)
        # init listener
        writer = mp.Process(target=self.save_data, args=[q])
        writer.start()
        # init pool
        pool = mp.Pool(self.n_workers)
        # init SMILES generator
        data = self._get_data()
        pb = tqdm(data, total=self.len(), desc="Load tasks: ")
        # main loop
        if not self.atom_feat_format:
            worker = self.create_graph
        else:
            worker = self._create_contextpred_graph
        for i, smi in enumerate(pb):
            pool.apply_async(worker, args=[smi, i, q])
        # finish the tasks
        pool.close()
        pool.join()
        q.put("END")
        writer.join()

    def len(self):
        return self.__len__()

    def _get_len(self):
        if self.scale == "filtered":
            return 430710
        else:
            return 1785415

    def __len__(self):
        try:
            return self._data_len
        except AttributeError:
            self._data_len = self._get_len()
            return self._data_len

    def get(self, idx):
        if idx == 604838:  # this molecule
            idx = 604839
        data = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        return data

    def _get_data(self):
        """ Method to get SMILES strings.
        """
        if self.scale == "filtered":
            with open(self.raw_paths[0]) as f:
                for smiles in f.readlines():
                    yield smiles.strip()
        else:
            with gzip.open(self.raw_paths[0]) as f:
                for line in f.readlines():
                    # skip header
                    if line.decode().startswith("smiles"):
                        continue
                    yield line.decode().split(",")[0]


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scale", type=str, default="full")
    parser.add_argument("--atom-feat-format", type=str, default="contextpred")
    args = parser.parse_args()
    chembl = ChemBLFP(
        root=args.root,
        n_workers=args.workers,
        scale=args.scale,
        atom_feat_format=args.atom_feat_format,
    )
    print(chembl[0])
