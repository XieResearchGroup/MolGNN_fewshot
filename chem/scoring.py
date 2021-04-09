### screening 
from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn


from tqdm import tqdm
import numpy as np

from model import GNN_MLP, GNN_graphpred, GNN
import pandas as pd

import os

from util import ONEHOT_ENCODING

def main():
    dataset = MoleculeDataset(
        root="/raid/home/public/dataset_ContextPred_0219/" + "repurposing"
    )

    dataset = MoleculeDataset(
        root="/raid/home/public/dataset_ContextPred_0219/" + "repurposing",
        transform=ONEHOT_ENCODING(dataset=dataset),
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
    )

    model = GNN_graphpred(
        num_layer=5,
        node_feat_dim=154,
        edge_feat_dim=2,
        emb_dim=256,
        num_tasks=1,
        JK="last",
        drop_ratio=0.5,
        graph_pooling="mean",
        gnn_type="gine",
        use_embedding=0,
    )



    model.load_state_dict(torch.load("tuned_model/jak3/90.pth"))
    model.eval()
    id=[]
    cid=[]
    score=[]
    fields=['id','cid','score']
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            id.append(batch.id)
            cid.append(batch.cid)
            score.append(pred)

    dict = {'id': id, 'cid': cid, 'score': score}  
    df = pd.DataFrame(dict) 


    df.to_csv('jak3_score_90.csv') 


if __name__ == "__main__":
    main()



