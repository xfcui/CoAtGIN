import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, ConstantLR, ExponentialLR, SequentialLR

from gnn import GNN
from adan import Adan

import os
from tqdm import tqdm
import argparse
import time
import numpy as np


### importing OGB-LSC
from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4Mv2Evaluator

reg_criterion = torch.nn.L1Loss()

learn_rate, weight_decay = 3e-3, 2e-2
def param(model, lr=learn_rate, wd=weight_decay):
    param_groups = [{'params': [], 'lr': lr,   'weight_decay': 0},
                    {'params': [], 'lr': lr,   'weight_decay': wd},
                    {'params': [], 'lr': lr/2, 'weight_decay': wd*2}]

    for n, p in model.named_parameters():
        if n.find('_embedding_') > 0: param_groups[0]['params'].append(p)
        elif n.find('head') > 0:      param_groups[2]['params'].append(p)
        elif n.endswith('scale'):     param_groups[0]['params'].append(p)
        elif n.endswith('bias'):      param_groups[0]['params'].append(p)
        elif n.endswith('weight'):    param_groups[1]['params'].append(p)
        else: raise Exception('Unknown parameter name:', n)

    return param_groups

def train(model, loader, optimizer):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.cuda()

        pred = model(batch).view(-1,)
        optimizer.zero_grad()
        loss = reg_criterion(pred, batch.y)
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)

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
    parser.add_argument('--gnn', type=str, default='coat3211',
                        help='GNN coat3211 (default: coat3211)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='number of GNN message passing layers (default: 4)')
    parser.add_argument('--emb_dim', type=int, default=256,
                        help='dimensionality of hidden units in GNNs (default: 256)')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='input batch size for training (default: 512)')
    parser.add_argument('--warmups', type=int, default=20,
                        help='number of warmups to train (default: 20)')
    parser.add_argument('--epochs', type=int, default=120,
                        help='number of epochs to train (default: 120)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers (default: 4)')
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default = '',
                        help='directory to save checkpoint')
    parser.add_argument('--save_test_dir', type=str, default = '',
                        help='directory to save test submission file')
    args = parser.parse_args()
    print(args)

    ### automatic dataloading and splitting
    dataset = PygPCQM4Mv2Dataset(root = 'data/')
    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = PCQM4Mv2Evaluator()

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    if args.save_test_dir != '':
        testdev_loader = DataLoader(dataset[split_idx["test-dev"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    if args.checkpoint_dir != '':
        os.makedirs(args.checkpoint_dir, exist_ok = True)

    shared_params = {
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling
    }

    if   args.gnn == 'coat1100':
        model = GNN(gnn_type='gin', virtual_node=2, conv_hop=1, conv_kernel=1, **shared_params).cuda()
    elif args.gnn == 'coat2100':
        model = GNN(gnn_type='gin', virtual_node=2, conv_hop=2, conv_kernel=1, **shared_params).cuda()
    elif args.gnn == 'coat3100':
        model = GNN(gnn_type='gin', virtual_node=2, conv_hop=3, conv_kernel=1, **shared_params).cuda()
    elif args.gnn == 'coat3200':
        model = GNN(gnn_type='gin', virtual_node=2, conv_hop=3, conv_kernel=2, **shared_params).cuda()
    elif args.gnn == 'coat3210':
        model = GNN(gnn_type='gin', virtual_node=3, conv_hop=3, conv_kernel=2, **shared_params).cuda()
    elif args.gnn == 'coat3201':
        model = GNN(gnn_type='gin', virtual_node=4, conv_hop=3, conv_kernel=2, **shared_params).cuda()
    elif args.gnn == 'coat3211':
        model = GNN(gnn_type='gin', virtual_node=5, conv_hop=3, conv_kernel=2, **shared_params).cuda()
    else:
        raise ValueError('Invalid GNN type')
    print('#params:', np.sum([np.prod(p.shape) for p in model.parameters()]))

    if args.log_dir != '':
        writer = SummaryWriter(log_dir=args.log_dir)

    optimizer = Adan(param(model), lr=learn_rate, weight_decay=weight_decay)
    sched0 = LinearLR(optimizer, 1e-4, 1.0)
    sched1 = ConstantLR(optimizer, 1.0)
    sched2 = ExponentialLR(optimizer, (1e-5/learn_rate)**(1/args.epochs))
    sched = SequentialLR(optimizer, [sched0, sched1, sched2], [args.warmups//5, args.warmups])

    best_valid_mae = 9999
    for epoch in range(args.warmups+args.epochs+1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_mae = train(model, train_loader, optimizer)

        print('Evaluating...')
        valid_mae = eval(model, valid_loader, evaluator)

        print('#summary: %.4f %.4f %.2e' % (train_mae, valid_mae, sched.get_last_lr()[0]))

        if args.log_dir != '':
            writer.add_scalar('valid/mae', valid_mae, epoch)
            writer.add_scalar('train/mae', train_mae, epoch)

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            if args.checkpoint_dir != '':
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_val_mae': best_valid_mae}
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'checkpoint.pt'))

            if args.save_test_dir != '' and best_valid_mae < 0.091:
                testdev_pred = test(model, testdev_loader)
                testdev_pred = testdev_pred.cpu().detach().numpy()

                print('Saving test submission file...')
                evaluator.save_test_submission({'y_pred': testdev_pred}, args.save_test_dir, mode = 'test-dev')

        sched.step()
            
        print(f'Best validation MAE so far: {best_valid_mae}')

    if args.log_dir != '':
        writer.close()


if __name__ == "__main__":
    main()
