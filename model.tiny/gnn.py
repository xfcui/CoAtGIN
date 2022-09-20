import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform

from conv import GNN_node, GNN_node_Virtualnode
from modify import CoAtGIN

from torch_scatter import scatter_mean

class GNN(torch.nn.Module):

    def __init__(self, num_tasks = 1, num_layers = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = 2, conv_hop = 2, conv_kernel = 2, residual = False,
                    drop_ratio = 0, JK = "last", graph_pooling = "sum"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''
        super(GNN, self).__init__()

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if   virtual_node == 0:
            self.gnn_node = GNN_node(num_layers, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual, gnn_type=gnn_type)
        elif virtual_node == 1:
            self.gnn_node = GNN_node_Virtualnode(num_layers, emb_dim, JK=JK, drop_ratio=drop_ratio, residual=residual, gnn_type=gnn_type)
        elif virtual_node == 2:
            self.gnn_node = CoAtGIN(num_layers, emb_dim, conv_hop, conv_kernel, JK=JK, gnn_type=gnn_type, use_virt=False, use_att=False)
        elif virtual_node == 3:
            self.gnn_node = CoAtGIN(num_layers, emb_dim, conv_hop, conv_kernel, JK=JK, gnn_type=gnn_type, use_virt=True, use_att=False)
        elif virtual_node == 4:
            self.gnn_node = CoAtGIN(num_layers, emb_dim, conv_hop, conv_kernel, JK=JK, gnn_type=gnn_type, use_virt=False, use_att=True)
        elif virtual_node == 5:
            self.gnn_node = CoAtGIN(num_layers, emb_dim, conv_hop, conv_kernel, JK=JK, gnn_type=gnn_type, use_virt=True, use_att=True)
        else:
            raise ValueError("Invalid graph type.")

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        self.norm = torch.nn.GroupNorm(1, self.emb_dim, affine=False)
        if graph_pooling == "set2set":
            self.head = torch.nn.Linear(self.emb_dim*2, self.num_tasks)
        else:
            self.head = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        h = self.gnn_node(batched_data)
        h = self.pool(h, batched_data.batch)
        h = self.norm(h)
        h = self.head(h)
        if not self.training: h.clamp_(0, 20)
        return h


if __name__ == '__main__':
    GNN(num_tasks = 10)
