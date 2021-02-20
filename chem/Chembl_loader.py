import os.path as osp
import multiprocessing as mp

import torch
from torch_geometric.data import Dataset
from chemreader.readers import Smiles
from tqdm import tqdm

from .util import get_filtered_fingerprint


class ChemBLFP(Dataset):
    def __init__(self, root=None, transform=None, pre_transform=None, n_workers=4):
        if root is None:
            root = osp.join("data", "ChemBL")
        self.n_workers = n_workers
        super().__init__(root, transform, pre_transform)

    @property
    def raw_dir(self):
        return self.root

    @property
    def raw_file_names(self):
        return ["smiles.csv"]

    @property
    def processed_file_names(self):
        return ["data_1.pt", "data_2.pt", "data_430000.pt"]

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
            torch.save(graph, osp.join(self.processed_dir, f"data_{idx}.pt"))
            print(
                "graph #{} saved to data_{}.pt{}".format(idx, idx, " " * 40), end="\r"
            )

    def create_graph(self, smi, idx, q):
        try:
            graph = Smiles(smi).to_graph(sparse=True, pyg=True)
        except AttributeError:
            return
        fp = get_filtered_fingerprint(smi)
        label = torch.tensor(list(fp), dtype=torch.long)[None, :]
        q.put((graph, label, idx))

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
        for i, smi in enumerate(pb):
            pool.apply_async(self.create_graph, args=[smi, i, q])
        # finish the tasks
        pool.close()
        pool.join()
        q.put("END")
        writer.join()

    def len(self):
        return self.__len__()

    def _get_len(self):
        n = 0
        with open(self.raw_paths[0]) as f:
            for line in f.readlines():
                n += 1
        return n

    def __len__(self):
        try:
            return self._data_len
        except AttributeError:
            self._data_len = self._get_len()
            return self._data_len

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        return data

    def _get_data(self):
        """ Method to get SMILES strings and generate fingerprint from the raw data.
        """
        with open(self.raw_paths[0]) as f:
            for smiles in f.readlines():
                yield smiles.strip()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    chembl = ChemBLFP(root=args.root)
    print(chembl[0])
