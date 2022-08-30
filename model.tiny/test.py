import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, ConstantLR, ExponentialLR, SequentialLR

from gnn import GNN

import os
from tqdm import tqdm
import argparse
import time
import numpy as np


### importing OGB-LSC
from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4Mv2Evaluator

def eval(model, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.cuda()

        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]

def test(model, loader):
    model.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.cuda()

        with torch.no_grad():
            pred = model(batch).view(-1,)

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim = 0)

    return y_pred


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on pcqm4m with Pytorch Geometrics')
    parser.add_argument('--gnn', type=str, default='gin-linformer',
                        help='GNN gin, gin-virtual, gin-linformer, or gcn, gcn-virtual, gcn-linformer (default: gin-linformer)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=256,
                        help='dimensionality of hidden units in GNNs (default: 256)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 512)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers (default: 4)')
    parser.add_argument('--checkpoint_dir', type=str, default = '',
                        help='directory to save checkpoint')
    args = parser.parse_args()
    print(args)

    ### automatic dataloading and splitting
    dataset = PygPCQM4Mv2Dataset(root = 'dataset/')

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = PCQM4Mv2Evaluator()

    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    testdev_loader = DataLoader(dataset[split_idx["test-dev"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    testchallenge_loader = DataLoader(dataset[split_idx["test-challenge"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    shared_params = {
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling
    }

    if args.gnn == 'gcn':
        model = GNN(gnn_type='gcn', virtual_node=0, **shared_params).cuda()
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type='gcn', virtual_node=1, **shared_params).cuda()
    elif args.gnn == 'gin':
        model = GNN(gnn_type='gin', virtual_node=0, **shared_params).cuda()
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type='gin', virtual_node=1, **shared_params).cuda()
    elif args.gnn == 'gin-plus100':
        model = GNN(gnn_type='gin', virtual_node=2, conv_hop=1, **shared_params).cuda()
    elif args.gnn == 'gin-plus200':
        model = GNN(gnn_type='gin', virtual_node=2, conv_hop=2, **shared_params).cuda()
    elif args.gnn == 'gin-plus300':
        model = GNN(gnn_type='gin', virtual_node=2, conv_hop=3, **shared_params).cuda()
    elif args.gnn == 'gin-plus310':
        model = GNN(gnn_type='gin', virtual_node=3, conv_hop=3, **shared_params).cuda()
    elif args.gnn == 'gin-plus301':
        model = GNN(gnn_type='gin', virtual_node=4, conv_hop=3, **shared_params).cuda()
    elif args.gnn == 'gin-plus311':
        model = GNN(gnn_type='gin', virtual_node=5, conv_hop=3, **shared_params).cuda()
    else:
        raise ValueError('Invalid GNN type')
    print('#params:', np.sum([np.prod(p.shape) for p in model.parameters()]))
    state = torch.load(os.path.join(args.checkpoint_dir, 'checkpoint.pt'))['model_state_dict']
    model.load_state_dict(state)

    print('Evaluating...')
    valid_mae = eval(model, valid_loader, evaluator)
    print('#mae:', valid_mae)

    testdev_pred = test(model, testdev_loader)
    testdev_pred = testdev_pred.cpu().detach().numpy()

    testchallenge_pred = test(model, testchallenge_loader)
    testchallenge_pred = testchallenge_pred.cpu().detach().numpy()

    print('Saving test submission file...')
    evaluator.save_test_submission({'y_pred': testdev_pred}, args.checkpoint_dir, mode = 'test-dev')
    evaluator.save_test_submission({'y_pred': testchallenge_pred}, args.checkpoint_dir, mode = 'test-challenge')


if __name__ == "__main__":
    main()
