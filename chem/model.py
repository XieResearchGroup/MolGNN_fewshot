import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    GlobalAttention,
    Set2Set,
)
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

num_atom_type = 120  # including the extra mask tokens
num_degree = 11
num_formal_charge = 11
num_hybrid = 7
num_aromatic = 2
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim),
        )
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        # self_loop_attr = self_loop_attr.to(edge_attr).to(edge_attr.dtype)
        # edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        # edge_embeddings = self.edge_embedding1(edge_attr) + \
        # self.edge_embedding2(edge_attr)

        return self.propagate(self.aggr, edge_index, x=x)
        # return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINE(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.
    Not the same as contextPred, use MLP instead of embedding layer to convert input
    dimension.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.


    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, edge_feat_dim, emb_dim, aggr="add"):
        super(GINE, self).__init__()
        # input convert
        self.edge_in_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_feat_dim, 2 * emb_dim),
            torch.nn.Relu(),
            torch.nn.Linear(2 * emb_dim, emb_dim),
        )
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim),
        )

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_attr = self.edge_in_mlp(edge_attr)
        return self.propagate(edge_index[0], x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        # assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones(
            (edge_index.size(1),), dtype=dtype, device=edge_index.device
        )
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(
            self.aggr, edge_index, x=x, edge_attr=edge_embeddings, norm=norm
        )

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):

        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool
            for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_degree, emb_dim)
        self.x_embedding3 = torch.nn.Embedding(num_formal_charge, emb_dim)
        self.x_embedding4 = torch.nn.Embedding(num_hybrid, emb_dim)
        self.x_embedding5 = torch.nn.Embedding(num_aromatic, emb_dim)
        self.x_embedding6 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding5.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding6.weight.data)

        # List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 2:
            x, edge_index = argv[0], argv[1]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index = data.x, data.edge_index
        else:
            raise ValueError("unmatched number of arguments.")

        x = x.to(torch.long)
        # to get rid of negative values of formal charge
        for i in x[:, 2]:

            i += 4
        # print(f'before add two{x[:,3]}')
        # x=x+1
        # print (f'after {x[:,3]}')
        # print(f'type of x: {x.type()}')
        # print(f'size of x:{x.size()})
        # print(f'formal charge{x[:,2]}')
        # print(f'self.x_embedding1 (x[:,0]):{self.x_embedding1(x[:,0])}')
        # print(f'self.x_embedding2 (x[:,1]):{self.x_embedding2(x[:,1])}')
        # print(f'self.x_embedding3 (x[:,2])'
        #     f'the formal charge num:{self.x_embedding3(x[:,2])}')
        # print(f'self.x_embedding4 (x[:,3]):{self.x_embedding4(x[:,3])}')
        # print(f'self.x_embedding5 (x[:,4]):{self.x_embedding5(x[:,4])}')
        # print(f'self.x_embedding6 (x[:,5]):{self.x_embedding6(x[:,5])}')
        # x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1]) \
        #     + self.x_embedding4(x[:,3]) + self.x_embedding5(x[:,4]) \
        #     + self.x_embedding6(x[:,5])
        x = (
            self.x_embedding1(x[:, 0])
            + self.x_embedding2(x[:, 1])
            + self.x_embedding3(x[:, 2])
            + self.x_embedding4(x[:, 3])
            + self.x_embedding5(x[:, 4])
            + self.x_embedding6(x[:, 5])
        )
        # print(f'shape of x {x.size()},x[0] :{x[0]}, edge attribute :{edge_attr[0]}')
        h_list = [x]
        for layer in range(self.num_layer):
            # print(f'h list[layer], edge_index, edge_attri:{h_list[layer].size()}'
            # f'{h_list[layer]}###{edge_index.size()}{edge_index}'
            # f'####{edge_attr.size()}{edge_attr}')
            h = self.gnns[layer](h_list[layer], edge_index)
            # print(f'argv infomation{argv[0]}, {argv[1]}, {argv[2]},'
            # f'length of argv{len(argv)}')
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation


class GNN_MLP(torch.nn.Module):
    """


    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool
            for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gine, gcn, graphsage, gat


    Output:
        node representations

    """

    def __init__(
        self,
        num_layer,
        node_feat_dim,
        edge_feat_dim,
        emb_dim,
        JK="last",
        drop_ratio=0,
        gnn_type="gine",
    ):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.node_in_mlp = torch.nn.Sequential(
            torch.nn.Linear(node_feat_dim, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.ReLU(),
        )
        # List of convolutional layers
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            if gnn_type == "gine":
                self.gnns.append(GINE(edge_feat_dim, emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for _ in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.node_in_mlp(x)
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type (str): gin, gine, gcn, graphsage, gat
        use_embedding (bool): whether to use embedding layer to convert imput dim

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(
        self,
        num_layer,
        emb_dim,
        num_tasks,
        JK="last",
        drop_ratio=0,
        graph_pooling="mean",
        gnn_type="gine",
        use_embedding=False,
    ):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if use_embedding:
            self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)
        else:
            self.gnn = GNN_MLP(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(
                    gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1)
                )
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(
                self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks
            )
        else:
            self.graph_pred_linear = torch.nn.Linear(
                self.mult * self.emb_dim, self.num_tasks
            )

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        return self.graph_pred_linear(self.pool(node_representation, batch))


class GNN_fingerprint(torch.nn.Module):
    """
    Extension of GIN to incorporate fingerprint recoginition.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type (str): gin, gine, gcn, graphsage, gat
        use_embedding (bool): whether to use embedding layer to convert input dim

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(
        self,
        num_layer,
        emb_dim,
        fingerprint_dim,
        JK="last",
        graph_pooling="mean",
        drop_ratio=0,
        gnn_type="gine",
        use_embedding=False,
    ):
        super(GNN_fingerprint, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        if use_embedding:
            self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)
        else:
            self.gnn = GNN_MLP(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)
        self.linear_pred_fingerprint = torch.nn.Linear(emb_dim, emb_dim)

        self.fingerprint_decoder = FingerprintDecoder(emb_dim, fingerprint_dim)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(
                    gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1)
                )
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")

    def from_pretrained(self, model_file):
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, batch = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, batch = data.x, data.edge_index, data.batch
        else:
            raise ValueError("unmatched number of arguments.")
        node_representation = self.gnn(x, edge_index)
        score_fingerprint = self.linear_pred_fingerprint(
            self.pool(node_representation, batch)
        )
        fingerprint_rec = self.fingerprint_decoder(score_fingerprint)
        return fingerprint_rec


class FingerprintDecoder(torch.nn.Module):
    def __init__(self, n_in, n_out, dropout=0.1):
        # n_in is the same as the hidden representation of the VGAE.
        super(FingerprintDecoder, self).__init__()
        if n_out > n_in:
            n_hidden = n_out // 2
        else:
            n_hidden = n_in // 2
        self.fc1 = torch.nn.Linear(n_in, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, n_out)
        self.dropout = dropout

    def forward(self, x):
        # input is hidden representation which has the shape of [None, 1, Hidden]
        # --> [None, 1, this hidden] --> [None, 1, output dim]
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        # this is different to Yang's network as we are directly processing based on
        # the N*hidden output of VGAE
        # x = torch.mean(x, -2)
        return x


if __name__ == "__main__":
    pass
