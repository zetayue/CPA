import torch
import torch.nn.functional as F
from torch.nn import Parameter, Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros, ones
from torch_scatter import scatter_add

from util import const, MessagePassing

class CPAConv(MessagePassing):
    
    def __init__(self, config):
        super(CPAConv, self).__init__('add')

        self.nhid = config.nhid
        self.heads = config.heads
        self.negative_slope = config.alpha
        self.dropout = config.dropout
        self.mod = config.mod
        self.activation = ReLU()

        self.att = Parameter(torch.Tensor(1, self.heads, 2 * self.nhid))
        self.w = Parameter(torch.ones(self.nhid))
        self.l1 = Parameter(torch.FloatTensor(1, self.nhid))
        self.b1 = Parameter(torch.FloatTensor(1, self.nhid))
        self.l2 = Parameter(torch.FloatTensor(self.nhid, self.nhid))
        self.b2 = Parameter(torch.FloatTensor(1, self.nhid))
        
        self.mlp = Sequential(Linear(self.nhid, self.nhid), Dropout(self.dropout), 
                              ReLU(), BatchNorm1d(self.nhid),
                              Linear(self.nhid, self.nhid), Dropout(self.dropout), 
                              ReLU(), BatchNorm1d(self.nhid))
        
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)        
        ones(self.l1)
        zeros(self.b1)
        const(self.l2, 1 / self.nhid)
        zeros(self.b2)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        x = x.view(-1, self.heads, self.nhid)
        output = self.propagate(edge_index, x=x, num_nodes=x.size(0))
        
        output = self.mlp(output)
        
        return output

    def message(self, x_i, x_j, edge_index, num_nodes):
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0], None, num_nodes)
        
        if self.mod == "additive":
            ones = torch.ones_like(alpha)
            h = x_j * ones.view(-1, self.heads, 1)
            h = torch.mul(self.w, h)
            
            return x_j * alpha.view(-1, self.heads, 1) + h
            
        elif self.mod == "scaled":
            ones = alpha.new_ones(edge_index[0].size())
            degree = scatter_add(ones, edge_index[0], dim_size=num_nodes)[edge_index[0]].unsqueeze(-1)
            degree = torch.matmul(degree, self.l1) + self.b1
            degree = self.activation(degree)
            degree = torch.matmul(degree, self.l2) + self.b2
            degree = degree.unsqueeze(-2)
            
            return torch.mul(x_j * alpha.view(-1, self.heads, 1), degree)
            
        elif self.mod == "f-additive":
            alpha = torch.where(alpha > 0, alpha + 1, alpha)
            
        elif self.mod == "f-scaled":
            ones = alpha.new_ones(edge_index[0].size())
            degree = scatter_add(ones, edge_index[0], dim_size=num_nodes)[edge_index[0]].unsqueeze(-1)
            alpha = alpha * degree
            
        else:
            alpha = alpha  # origin
        
        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * self.nhid)
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.nhid, self.heads)
