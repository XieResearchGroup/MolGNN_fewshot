import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


from tqdm import tqdm
import numpy as np


from model import GNN, GNN_fingerprint
from sklearn.metrics import roc_auc_score


def train(args, model, device, loader, optimizer, criterion):

    model.train()

    for step, batch in enumerate(tqdm(loader,desc='Iteration')):
        batch = batch.to(device)
        pred = model (batch.x,batch.edge_index)
        y = batch.y.view(pred.shape).to(torch.float64)
       
        loss = criterion(pred, y) 

        # backprop
        optimizer.zero_grad()
        loss.backward()
 
        optimizer.step()


def eval ( args,model,loader,criterion):

    model.eval()
    y_true = []
    y_scores = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index)

        y_true.append(batch.y.view(pred.shape).cpu())
        y_scores.append(pred.cpu())
        loss = criterion(pred, y) 

    labels_all = np.array(y_true)
    preds_all = np.array(y_pred)
    assert labels_all.shape == preds_all.shape
    
    roc_score = roc_auc_score(labels_all, preds_all)




    return roc_score


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of fingerprint')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = '----', help='root directory of dataset. For now, only fiingerprint.')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--output_model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


    #set up dataset, BACEFP for instance
    dataset= DeepChemDatasetPG('dataset/BACEFP')

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    #set up model
    model = GNN_fingerprint (args.num_layer, args.emb_dim, fingerprint_dim = 740, JK = args.JK, drop_ratio = args.dropout_ratio,  gnn_type = args.gnn_type)
    
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file + ".pth")
    
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)  
    print(optimizer)


    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
    
        train(args, model, device, loader, optimizer)

    if not args.output_model_file == "":
        torch.save(model.gnn.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    main()

