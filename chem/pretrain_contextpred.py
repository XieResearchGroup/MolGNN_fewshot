import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from model import GNN_MLP, GNN
from loader import MoleculeDataset
from util import ONEHOT_ContextPair, ExtractSubstructureContextPair
from dataloader import DataLoaderSubstructContext


# from .ONEHOT import ONEHOT_ENCODING


def pool_func(x, batch, mode="sum"):
    if mode == "sum":
        return global_add_pool(x, batch)
    elif mode == "mean":
        return global_mean_pool(x, batch)
    elif mode == "max":
        return global_max_pool(x, batch)


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


criterion = nn.BCEWithLogitsLoss()


def train(
    args,
    model_substruct,
    model_context,
    loader,
    optimizer_substruct,
    optimizer_context,
    device,
):
    model_substruct.train()
    model_context.train()

    balanced_loss_accum = 0
    acc_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        # creating substructure representation
        substruct_rep = model_substruct(
            batch.x_substruct, batch.edge_index_substruct, batch.edge_attr_substruct
        )[batch.center_substruct_idx]

        # creating context representations
        overlapped_node_rep = model_context(
            batch.x_context, batch.edge_index_context, batch.edge_attr_context
        )[batch.overlap_context_substruct_idx]

        # Contexts are represented by
        if args.mode == "cbow":
            # positive context representation
            context_rep = pool_func(
                overlapped_node_rep,
                batch.batch_overlapped_context,
                mode=args.context_pooling,
            )
            # negative contexts are obtained by shifting the indicies of context
            # embeddings
            neg_context_rep = torch.cat(
                [
                    context_rep[cycle_index(len(context_rep), i + 1)]
                    for i in range(args.neg_samples)
                ],
                dim=0,
            )

            pred_pos = torch.sum(substruct_rep * context_rep, dim=1)
            pred_neg = torch.sum(
                substruct_rep.repeat((args.neg_samples, 1)) * neg_context_rep, dim=1
            )

        elif args.mode == "skipgram":

            expanded_substruct_rep = torch.cat(
                [
                    substruct_rep[i].repeat((batch.overlapped_context_size[i], 1))
                    for i in range(len(substruct_rep))
                ],
                dim=0,
            )
            pred_pos = torch.sum(expanded_substruct_rep * overlapped_node_rep, dim=1)

            # shift indices of substructures to create negative examples
            shifted_expanded_substruct_rep = []
            for i in range(args.neg_samples):
                shifted_substruct_rep = substruct_rep[
                    cycle_index(len(substruct_rep), i + 1)
                ]
                shifted_expanded_substruct_rep.append(
                    torch.cat(
                        [
                            shifted_substruct_rep[i].repeat(
                                (batch.overlapped_context_size[i], 1)
                            )
                            for i in range(len(shifted_substruct_rep))
                        ],
                        dim=0,
                    )
                )

            shifted_expanded_substruct_rep = torch.cat(
                shifted_expanded_substruct_rep, dim=0
            )
            pred_neg = torch.sum(
                shifted_expanded_substruct_rep
                * overlapped_node_rep.repeat((args.neg_samples, 1)),
                dim=1,
            )

        else:
            raise ValueError("Invalid mode!")

        loss_pos = criterion(
            pred_pos.double(), torch.ones(len(pred_pos)).to(pred_pos.device).double()
        )
        loss_neg = criterion(
            pred_neg.double(), torch.zeros(len(pred_neg)).to(pred_neg.device).double()
        )

        optimizer_substruct.zero_grad()
        optimizer_context.zero_grad()

        loss = loss_pos + args.neg_samples * loss_neg
        loss.backward()
        # To write: optimizer
        optimizer_substruct.step()
        optimizer_context.step()

        balanced_loss_accum += float(
            loss_pos.detach().cpu().item() + loss_neg.detach().cpu().item()
        )
        acc_accum += 0.5 * (
            float(torch.sum(pred_pos > 0).detach().cpu().item()) / len(pred_pos)
            + float(torch.sum(pred_neg < 0).detach().cpu().item()) / len(pred_neg)
        )

    return balanced_loss_accum / step, acc_accum / step


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
        default=256,
        help="input batch size for training (default: 256)",
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
        "--decay", type=float, default=0, help="weight decay (default: 0)"
    )
    parser.add_argument(
        "--num_layer",
        type=int,
        default=5,
        help="number of GNN message passing layers (default: 5).",
    )
    parser.add_argument(
        "--csize", type=int, default=3, help="context size (default: 3)."
    )
    parser.add_argument(
        "--emb_dim", type=int, default=64, help="embedding dimensions (default: 300)"
    )
    parser.add_argument(
        "--node_feat_dim", type=int, default=32, help="dimension of the node features.",
    )
    parser.add_argument(
        "--edge_feat_dim", type=int, default=2, help="dimension ofo the edge features."
    )

    parser.add_argument(
        "--dropout_ratio", type=float, default=0, help="dropout ratio (default: 0)"
    )
    parser.add_argument(
        "--neg_samples",
        type=int,
        default=1,
        help="number of negative contexts per positive context (default: 1)",
    )
    parser.add_argument(
        "--JK",
        type=str,
        default="last",
        help="how the node features are combined across layers."
        "last, sum, max or concat",
    )
    parser.add_argument(
        "--context_pooling",
        type=str,
        default="mean",
        help="how the contexts are pooled (sum, mean, or max)",
    )
    parser.add_argument("--mode", type=str, default="cbow", help="cbow or skipgram")
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/zinc_standard_agent",
        help="root directory of dataset for pretraining",
    )
    parser.add_argument(
        "--output_model_file", type=str, default="trained_model/context_onehot_mlp", help="filename to output the model"
    )
    parser.add_argument("--gnn_type", type=str, default="gine")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for splitting dataset."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of workers for dataset loading",
    )
    args = parser.parse_args()

    print("show all arguments configuration...")
    print(args)

    torch.manual_seed(0)
    np.random.seed(0)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    l1 = args.num_layer - 1
    l2 = l1 + args.csize

    print(args.mode)
    print("num layer: %d l1: %d l2: %d" % (args.num_layer, l1, l2))

    # set up dataset and transform function.
    dataset_og = MoleculeDataset(
        root=args.dataset, dataset=os.path.basename(args.dataset)
    )

    dataset = MoleculeDataset(
        root=args.dataset,
        dataset=os.path.basename(args.dataset),
        transform=ONEHOT_ContextPair(
            dataset=dataset_og, k=args.num_layer, l1=l1, l2=l2
        ),
    )
    loader = DataLoaderSubstructContext(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    # set up models, one for pre-training and one for context embeddings
    model_substruct = GNN_MLP(
        args.num_layer,
        args.node_feat_dim,
        args.edge_feat_dim,
        args.emb_dim,
        JK=args.JK,
        drop_ratio=args.dropout_ratio,
        gnn_type=args.gnn_type,
    ).to(device)
    model_context = GNN_MLP(
        int(l2 - l1),
        args.node_feat_dim,
        args.edge_feat_dim,
        args.emb_dim,
        JK=args.JK,
        drop_ratio=args.dropout_ratio,
        gnn_type=args.gnn_type,
    ).to(device)

    # set up optimizer for the two GNNs
    optimizer_substruct = optim.Adam(
        model_substruct.parameters(), lr=args.lr, weight_decay=args.decay
    )
    optimizer_context = optim.Adam(
        model_context.parameters(), lr=args.lr, weight_decay=args.decay
    )

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))

        train_loss, train_acc = train(
            args,
            model_substruct,
            model_context,
            loader,
            optimizer_substruct,
            optimizer_context,
            device,
        )
        print(train_loss, train_acc)

    if not args.output_model_file == "":
        torch.save(model_substruct.state_dict(), args.output_model_file + ".pth")


if __name__ == "__main__":
    # cycle_index(10,2)
    main()
