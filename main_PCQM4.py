import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from utils.loss import sim_matrix, NormSoftmaxLoss
from gnn import GNN
from gru import SeqModel
import os
from tqdm import tqdm
import argparse
import numpy as np
import random
from datetime import datetime
now = datetime.now()
timestamp = str(now.year)[-2:] + "_" + str(now.month).zfill(2) + "_" + str(now.day).zfill(2) + "_" + \
            str(now.hour).zfill(2) + str(now.minute).zfill(2) + str(now.second).zfill(2)

### importing OGB-LSC
from ogb.lsc import PCQM4Mv2Evaluator
from pre_process import PygPCQM4Mv2Dataset_SMILES_ADDED
reg_criterion = torch.nn.L1Loss()


def train(model, device, loader, optimizer, contrastive_loss):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch).view(-1, )
        optimizer.zero_grad()
        loss = reg_criterion(pred, batch.y)
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)


def eval(model, device, loader, evaluator, contrastive_loss):
    model.eval()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        with torch.no_grad():
            batch = batch.to(device)
            pred = model(batch).view(-1, )
            loss = reg_criterion(pred, batch.y)
            loss_accum += loss.detach().cpu().item()
    return loss_accum


def test(model, device, loader):
    model.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch).view(-1, )

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim=0)

    return y_pred


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on pcqm4m with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=5,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=600,
                        help='dimensionality of hidden units in GNNs (default: 600)')
    parser.add_argument('--gru_emb', type=int, default=32,
                        help='GRU token embed size (default: 64)')
    parser.add_argument('--gru_hid', type=int, default=64,
                        help='GRU hidden size (default: 256)')
    parser.add_argument('--max_len', type=int, default=500,
                        help='')

    parser.add_argument('--train_subset', type=str, default='T')  # Use only fraction of train set
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--num_tasks', type=int, default=1,
                        help='num_labels, tox21: 12, PCQM4: 1')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default=f'ckpt/{timestamp}', help='directory to save checkpoint')
    parser.add_argument('--save_test_dir', type=str, default=f'test_result/{timestamp}', help='directory to save test submission file')
    args = parser.parse_args()

    print(args)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygPCQM4Mv2Dataset_SMILES_ADDED(root='dataset/')

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = PCQM4Mv2Evaluator()

    if args.train_subset:
        subset_ratio = 0.01
        subset_idx = torch.randperm(len(split_idx["train"]))[:int(subset_ratio * len(split_idx["train"]))]
        train_loader = DataLoader(dataset[split_idx["train"][subset_idx]], batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers)
    else:
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers)

    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    if args.save_test_dir != '':
        testdev_loader = DataLoader(dataset[split_idx["test-dev"]], batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)
        testchallenge_loader = DataLoader(dataset[split_idx["test-challenge"]], batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_workers)

    if args.checkpoint_dir != '':
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    shared_params = {
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
        'graph_pooling': args.graph_pooling
    }

    if args.gnn == 'gin':
        model = GNN(gnn_type='gin', virtual_node=False, **shared_params).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type='gin', virtual_node=True, **shared_params).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type='gcn', virtual_node=False, **shared_params).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type='gcn', virtual_node=True, **shared_params).to(device)
    elif args.gnn == 'gru':
        model = SeqModel(args).to(device)
    else:
        raise ValueError('Invalid GNN type')

    num_params = sum(p.numel() for p in model.parameters())
    print(f'#Params: {num_params}')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if args.log_dir != '':
        writer = SummaryWriter(log_dir=args.log_dir)

    best_valid_mae = 1000
    best_epoch = 0
    if args.train_subset:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.25)
        # args.epochs = 1000
    else:
        scheduler = StepLR(optimizer, step_size=30, gamma=0.25)

    contrastive_loss = NormSoftmaxLoss()

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_mae = train(model, device, train_loader, optimizer, contrastive_loss)

        print('Evaluating...')
        valid_mae = eval(model, device, valid_loader, evaluator, contrastive_loss)

        print({'Train': train_mae, 'Validation': valid_mae})

        if args.log_dir != '':
            writer.add_scalar('valid/mae', valid_mae, epoch)
            writer.add_scalar('train/mae', train_mae, epoch)

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            best_epoch = epoch
            if args.checkpoint_dir != '':
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),
                              'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_valid_mae,
                              'num_params': num_params}
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'checkpoint.pt'))

            if args.save_test_dir != '':
                test_mae = eval(model, device, testchallenge_loader, evaluator, contrastive_loss)
                print(f"Test MAE: {test_mae}")
        scheduler.step()

        print(f'Best validation MAE so far: Epoch {best_epoch}: {best_valid_mae}')
    print(f"Test MAE: {test_mae}")
    if args.log_dir != '':
        writer.close()


if __name__ == "__main__":
    main()
