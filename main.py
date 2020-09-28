from __future__ import division
from __future__ import print_function

import os.path as osp
import time
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

from model import CPA
from util import separate_data, Constant, Config

def train(model, loader, optimizer, device):
    loss_all = 0
    model.train()
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        clip_grad_value_(model.parameters(), 2.0)
        optimizer.step()
        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)

def test(model, loader, device):
    acc_all = 0
    model.eval()
    
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        acc_all += pred.eq(data.y).sum().item()
    return acc_all / len(loader.dataset)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="PROTEINS", help='name of dataset')
    parser.add_argument('--mod', type=str, default="f-scaled", choices=["origin", "additive", "scaled", "f-additive", "f-scaled"], help='model to be used: origin, additive, scaled, f-additive, f-scaled')
    parser.add_argument('--seed', type=int, default=809, help='random seed')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate')
    parser.add_argument('--wd', type=float, default=1e-3, help='weight decay value')
    parser.add_argument('--n_layer', type=int, default=4, help='number of hidden layers')
    parser.add_argument('--hid', type=int, default=32, help='size of input hidden units')
    parser.add_argument('--heads', type=int, default=1, help='number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha for the leaky_relu')
    parser.add_argument('--kfold', type=int, default=10, help='number of kfold')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--readout', type=str, default="add", choices=["add", "mean"], help='readout function: add, mean')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', args.dataset)
    dataset = TUDataset(path, name=args.dataset, pre_transform=Constant()).shuffle()

    train_graphs, test_graphs = separate_data(len(dataset), args.kfold)

    kfold_num = args.kfold
    print('Dataset:', args.dataset)
    print('# of graphs:', len(dataset))
    print('# of classes:', dataset.num_classes)
          
    test_acc_values = torch.zeros(kfold_num, args.epochs)

    for idx in range(kfold_num):
        print('=============================================================================')
        print(kfold_num, 'fold cross validation:', idx+1)
        
        idx_train = train_graphs[idx]
        idx_test = test_graphs[idx]

        train_dataset = dataset[idx_train]
        test_dataset = dataset[idx_test]
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=args.seed)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        t_start = time.time()
        best_epoch = 0

        config = Config(mod=args.mod, nhid=args.hid, nclass=dataset.num_classes, 
                        nfeat=dataset.num_features, dropout=args.dropout, 
                        heads=args.heads, alpha=args.alpha, 
                        n_layer=args.n_layer, readout=args.readout)
            
        model = CPA(config).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)
        scheduler = MultiStepLR(optimizer, milestones=[50,100,150,200,250,300,350,400,450,500], gamma=0.5)

        for epoch in range(args.epochs):
            train_loss = train(model, train_loader, optimizer, device)
            train_acc = test(model, train_loader, device)
            test_acc = test(model, test_loader, device)
            test_acc_values[idx, epoch] = test_acc
            scheduler.step()
        
            print('Epoch {:03d}'.format(epoch+1),
              'train_loss: {:.4f}'.format(train_loss),
              'train_acc: {:.4f}'.format(train_acc),
              'test_acc: {:.4f}'.format(test_acc))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_start))

    print('=============================================================================')
    mean_test_acc = torch.mean(test_acc_values, dim=0)
    best_epoch = int(torch.argmax(mean_test_acc).data)
    print('Best Epoch:', best_epoch+1)
    print('Best Testing Accs:')
    for i in test_acc_values[:, best_epoch]:
        print('{:0.4f},'.format(i.item()),end='')
    print('\n')
    print('Averaged Best Testing Acc:')
    print('{:0.4f}'.format(mean_test_acc[best_epoch].item()))


if __name__ == '__main__':
    main()
