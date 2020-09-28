import time
import copy
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
from torch_geometric.nn import global_mean_pool, global_add_pool


from layer import CPAConv
from util import Config

class CPA(nn.Module):
    def __init__(self, config: Config):
        super(CPA, self).__init__()
        
        self.nhid = config.nhid
        self.nclass = config.nclass
        self.readout = config.readout
        self.dropout = config.dropout
        
        conv_layer = CPAConv(config)
        self.conv_layers = nn.ModuleList([copy.deepcopy(conv_layer) 
                           for _ in range(config.n_layer)])
            
        self.fc = Linear(config.nfeat, self.nhid, bias=False)
       
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(config.n_layer + 1):
            if layer == 0:
                self.linears_prediction.append(Linear(config.nfeat, self.nhid))
            else:
                self.linears_prediction.append(Linear(self.nhid, self.nhid))
        
        self.bns_fc = torch.nn.ModuleList()
        for layer in range(config.n_layer + 1):
            if layer == 0:
                self.bns_fc.append(BatchNorm1d(config.nfeat))
            else:    
                self.bns_fc.append(BatchNorm1d(self.nhid))
        
        self.linear = Linear(self.nhid, self.nclass)
        
    def forward(self, x, edge_index, batch):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
            
        if self.readout == 'mean':
            output_list = [global_mean_pool(x, batch)]
        else:
            output_list = [global_add_pool(x, batch)] 
        hid_x = self.fc(x)

        for conv in self.conv_layers:
            hid_x = conv(hid_x, edge_index)
            if self.readout == 'mean':
                output_list.append(global_mean_pool(hid_x, batch))               
            else:
                output_list.append(global_add_pool(hid_x, batch))

        score_over_layer = 0
        for layer, h in enumerate(output_list):
            h = self.bns_fc[layer](h)
            score_over_layer += F.relu(self.linears_prediction[layer](h))
            
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        x = self.linear(score_over_layer)
        return F.log_softmax(x, dim=-1)
        
    
    
       
