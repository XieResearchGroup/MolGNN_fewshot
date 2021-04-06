import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN_MLP, GNN_graphpred, GNN
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
)

from splitters import scaffold_split, random_split, oversample_split, random_scaffold_split
import pandas as pd

import os
import shutil

from tensorboardX import SummaryWriter

from util import ONEHOT_ENCODING


criterion = nn.BCEWithLogitsLoss(reduction="none")


def train(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        # Whether y is non-null or not.
        is_valid = y ** 2 > 0
        # Loss matrix
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        # loss matrix after removing null target
        loss_mat = torch.where(
            is_valid,
            loss_mat,
            torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype),
        )

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    ap_list = []
    roc_list = []
    acc_list = []
    f1_list = []

    #torch.sigmoid(y_scores)
    positive_true = [i for i in y_true if i >0]
    positive_scores = [i for i in y_scores if i >0]


    for i in range(y_true.shape[1]):
        # print(f'y true in the shape[1]:{y_true[:,i]}')
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            # print(f' after if the y_true value:{y_true[:,i]}')
            is_valid = y_true[:, i] ** 2 > 0
            # print(f'is valid {is_valid}')
            roc_list.append(
                roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i])
            )

            
            acc_list.append(accuracy_score(torch.tensor((y_true[is_valid, i] + 1) / 2), torch.sigmoid(torch.tensor(y_scores[is_valid, i])).round()))
            f1_list.append(f1_score(torch.tensor((y_true[is_valid, i] + 1) / 2), torch.sigmoid(torch.tensor(y_scores[is_valid, i])).round(), average='macro'))
            ap_list.append(average_precision_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
            
            
    
    #print(f'roc list{roc_list}, lenth roc : {len(roc_list)}, lenth true:{y_true.shape[1]}')

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(roc_list)) / y_true.shape[1]))

    return (
        sum(roc_list) / len(roc_list),
        sum(acc_list) / len(acc_list),
        sum(f1_list) / len(f1_list),
        sum(ap_list) / len(ap_list),
        len(positive_true), len(positive_scores)) 


# y_true.shape[1]


#


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of pre-training of graph neural networks"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="which gpu to use if any (default: 0)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--lr_scale",
        type=float,
        default=1,
        help="relative learning rate for the feature extraction layer (default: 1)",
    )
    parser.add_argument(
        "--decay", type=float, default=0, help="weight decay (default: 0)"
    )
    parser.add_argument(
        "--num_layer",
        type=int,
        default=5,
        help="number of GNN message passing layers (default: 5).",
    )

    parser.add_argument(
        "--node_feat_dim",
        type=int,
        default=154,
        help="dimension of the node features.",
    )
    parser.add_argument(
        "--edge_feat_dim", type=int, default=2, help="dimension ofo the edge features."
    )

    parser.add_argument(
        "--emb_dim", type=int, default=256, help="embedding dimensions (default: 300)"
    )
    parser.add_argument(
        "--dropout_ratio", type=float, default=0.5, help="dropout ratio (default: 0.5)"
    )
    parser.add_argument(
        "--graph_pooling",
        type=str,
        default="mean",
        help="graph level pooling (sum, mean, max, set2set, attention)",
    )
    parser.add_argument(
        "--JK",
        type=str,
        default="last",
        help="how the node features across layers are combined. last, sum, max or concat",
    )
    parser.add_argument("--gnn_type", type=str, default="gine")
    parser.add_argument(
        "--dataset",
        type=str,
        default="bbbp",
        help="root directory of dataset. For now, only classification.",
    )
    parser.add_argument(
        "--input_model_file",
        type=str,
        default="",
        help="filename to read the model (if there is any)",
    )
    parser.add_argument("--filename", type=str, default="", help="output filename")
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for splitting the dataset."
    )
    parser.add_argument(
        "--runseed",
        type=int,
        default=0,
        help="Seed for minibatch selection, random initialization.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="scaffold",
        help="random or scaffold or random_scaffold",
    )
    parser.add_argument(
        "--eval_train", type=int, default=0, help="evaluating training or not"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of workers for dataset loading",
    )
    parser.add_argument(
        "--use_original", type=int, default=0, help="run benchmark experiment or not"
    )
    #parser.add_argument('--output_model_file', type = str, default = 'finetuned_model/amu', help='filename to output the finetuned model')

    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    elif args.dataset in ["jak1", "jak2", "jak3", "amu", "ellinger", "mpro"]:
        num_tasks = 1
    else:
        raise ValueError("Invalid dataset name.")

    # set up dataset
    #    dataset = MoleculeDataset("contextPred/chem/dataset/" + args.dataset, dataset=args.dataset)
    dataset = MoleculeDataset(
        root="/raid/home/public/dataset_ContextPred_0219/" + args.dataset
    )
    if args.use_original == 0:
        dataset = MoleculeDataset(
            root="/raid/home/public/dataset_ContextPred_0219/" + args.dataset,
            transform=ONEHOT_ENCODING(dataset=dataset),
        )

    print(dataset)

    if args.split == "scaffold":
        smiles_list = pd.read_csv(
            "/raid/home/public/dataset_ContextPred_0219/"
            + args.dataset
            + "/processed/smiles.csv",
            header=None,
        )[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset,
            smiles_list,
            null_value=0,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
        )
        print("scaffold")
    elif args.split == "oversample":
        train_dataset, valid_dataset, test_dataset = oversample_split(
            dataset,
            null_value=0,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=args.seed,
        )
        print("oversample")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset,
            null_value=0,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=args.seed,
        )
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv(
             "/raid/home/public/dataset_ContextPred_0219/"
            + args.dataset
            + "/processed/smiles.csv",
            header=None
           
        )[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset,
            smiles_list,
            null_value=0,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=args.seed,
        )
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # set up model
    model = GNN_graphpred(
        args.num_layer,
        args.node_feat_dim,
        args.edge_feat_dim,
        args.emb_dim,
        num_tasks,
        JK=args.JK,
        drop_ratio=args.dropout_ratio,
        graph_pooling=args.graph_pooling,
        gnn_type=args.gnn_type,
        use_embedding=args.use_original,
    )
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file + ".pth")

    model.to(device)

    # set up optimizer
    # different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append(
            {"params": model.pool.parameters(), "lr": args.lr * args.lr_scale}
        )
    model_param_group.append(
        {"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale}
    )
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    train_roc_list = []
    train_acc_list = []
    train_f1_list = []
    train_ap_list = []
    val_roc_list = []
    val_acc_list = []
    val_f1_list = []
    val_ap_list = []
    test_roc_list = []
    test_acc_list = []
    test_f1_list = []
    test_ap_list = []

    if not args.filename == "":
        fname = (
            "/raid/home/yoyowu/Weihua_b/BASE_TFlogs/"
            + str(args.runseed)
            + "/"
            + args.filename
        )
        # delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train(args, model, device, train_loader, optimizer)
        #if not args.output_model_file == "":
         #   torch.save(model.state_dict(), args.output_model_file + str(epoch)+ ".pth")  

        print("====Evaluation")
        if args.eval_train:
            train_roc, train_acc, train_f1, train_ap, train_num_positive_true, train_num_positive_scores = eval(
                args, model, device, train_loader
            )



        else:
            print("omit the training accuracy computation")
            train_roc = 0
            train_acc = 0
            train_f1 = 0
            train_ap = 0
        val_roc, val_acc, val_f1, val_ap, val_num_positive_true, val_num_positive_scores = eval(args, model, device, val_loader)
        test_roc, test_acc, test_f1, test_ap, test_num_positive_true, test_num_positive_scores  = eval(args, model, device, test_loader)
        #with open('debug_ellinger.txt', "a") as f:
         #   f.write("====epoch " + str(epoch) +" \n training:  positive true count {} , positive scores count {} \n".format(train_num_positive_true,train_num_positive_scores))
          #  f.write("val:  positive true count {} , positive scores count {} \n".format(val_num_positive_true,val_num_positive_scores))
           # f.write("test:  positive true count {} , positive scores count {} \n".format(test_num_positive_true,test_num_positive_scores))
            #f.write("\n")



        print("train: %f val: %f test auc: %f " % (train_roc, val_roc, test_roc))
        val_roc_list.append(val_roc)
        val_f1_list.append(val_f1)
        val_acc_list.append(val_acc)
        val_ap_list.append(val_ap)
        test_acc_list.append(test_acc)
        test_roc_list.append(test_roc)
        test_f1_list.append(test_f1)
        test_ap_list.append(test_ap)
        train_acc_list.append(train_acc)
        train_roc_list.append(train_roc)
        train_f1_list.append(train_f1)
        train_ap_list.append(train_ap)

        if not args.filename == "":
            writer.add_scalar("data/train roc", train_roc, epoch)
            writer.add_scalar("data/train acc", train_acc, epoch)
            writer.add_scalar("data/train f1", train_f1, epoch)
            writer.add_scalar("data/train ap", train_ap, epoch)

            writer.add_scalar("data/val roc", val_roc, epoch)
            writer.add_scalar("data/val acc", val_acc, epoch)
            writer.add_scalar("data/val f1", val_f1, epoch)
            writer.add_scalar("data/val ap", val_ap, epoch)

            writer.add_scalar("data/test roc", test_roc, epoch)
            writer.add_scalar("data/test acc", test_acc, epoch)
            writer.add_scalar("data/test f1", test_f1, epoch)
            writer.add_scalar("data/test ap", test_ap, epoch)

        print("")

    if not args.filename == "":
        writer.close()


if __name__ == "__main__":
    main()
