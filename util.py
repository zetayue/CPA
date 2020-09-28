import os
import os.path as osp
import shutil
import numpy as np
import inspect
import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_scatter import scatter
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

def separate_data(ngraph, k):
    train_graph_list, test_graph_list =[], []
    graph_idx = np.arange(ngraph)
    split_Kfold = KFold(n_splits = k, shuffle=True)
    
    for Kfoldtrain, Kfoldtest in split_Kfold.split(graph_idx):
        Kfoldtrain = shuffle(Kfoldtrain)
        Kfoldtest = shuffle(Kfoldtest)
        train_graph_list.append(torch.LongTensor(Kfoldtrain))
        test_graph_list.append(torch.LongTensor(Kfoldtest))
    
    return train_graph_list, test_graph_list

def const(tensor, c):
    if tensor is not None:
        tensor.data.fill_(c)

class Constant(object):
    '''
    Assign a constant value as the label to each node when a dataset does not come with any node labels.
    Args:
        value (int, optional): The value to add. (default: :obj:`1`)
    '''
    def __init__(self, value=1):
        self.value = value

    def __call__(self, data):
        x = data.x
        c = torch.full((data.num_nodes, 1), self.value, dtype=torch.float)
        if x is None:
            data.x = c
        return data

    def __repr__(self):
        return '{}(value={})'.format(self.__class__.__name__, self.value)

class Config(object):
    def __init__(self, mod, nhid, dropout, alpha, heads, n_layer, nfeat, nclass, readout):
    
        self.mod = mod
        self.nhid = nhid
        self.dropout = dropout
        self.alpha = alpha
        self.heads = heads
        self.n_layer = n_layer
        self.nfeat = nfeat
        self.nclass = nclass
        self.readout = readout
        
class MessagePassing(torch.nn.Module):
    '''
    Base class for creating message passing layers
    This class is based on the one used in pytorch_geometric 1.1.2:
    https://github.com/rusty1s/pytorch_geometric/blob/1.1.2/torch_geometric/nn/conv/message_passing.py
    '''
    def __init__(self, aggr='add', flow='target_to_source'):
        super(MessagePassing, self).__init__()

        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max']

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.message_args = inspect.getargspec(self.message)[0][1:]
        self.update_args = inspect.getargspec(self.update)[0][2:]

    def propagate(self, edge_index, size=None, **kwargs):

        size = [None, None] if size is None else list(size)
        assert len(size) == 2

        kwargs['edge_index'] = edge_index
        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {"_i": i, "_j": j}

        message_args = []
        for arg in self.message_args:
            if arg[-2:] in ij.keys():
                tmp = kwargs[arg[:-2]]
                if tmp is None:
                    message_args.append(tmp)
                else:
                    idx = ij[arg[-2:]]
                    if isinstance(tmp, tuple) or isinstance(tmp, tuple):
                        assert len(tmp) == 2
                        if size[1 - idx] is None:
                            size[1 - idx] = tmp[1 - idx].size(0)
                        tmp = tmp[idx]

                    if size[idx] is None:
                        size[idx] = tmp.size(0)
                    tmp = torch.index_select(tmp, 0, edge_index[idx])
                    message_args.append(tmp)
            else:
                message_args.append(kwargs[arg])

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        kwargs['size'] = size
        update_args = [kwargs[arg] for arg in self.update_args]

        out = self.message(*message_args)
        out = scatter(out, edge_index[i], dim=0, dim_size=size[i], reduce=self.aggr)
        out = self.update(out, *update_args)

        return out

    def message(self, x_j):  # pragma: no cover

        return x_j

    def update(self, aggr_out):  # pragma: no cover

        return aggr_out