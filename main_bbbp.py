import os
# os.environ['CUDA_VISIBLE_DEVICES'] = f"0"
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from gnn import GNN
from gru import SeqModel


from tqdm import tqdm
import argparse
import time
import numpy as np
import random
from datetime import datetime
from ogb.graphproppred import Evaluator
now = datetime.now()
timestamp = str(now.year)[-2:] + "_" + str(now.month).zfill(2) + "_" + str(now.day).zfill(2) + "_" + \
            str(now.hour).zfill(2) + str(now.minute).zfill(2) + str(now.second).zfill(2)

### importing OGB-LSC
cls_criterion = torch.nn.BCEWithLogitsLoss()


def train(model, device, loader, optimizer, evaluator):
    model.train()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:

            is_labeled = batch.y == batch.y
            pred = model(batch)
            optimizer.zero_grad()
            loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    return evaluator.eval(input_dict)


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)


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
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=150,
                        help='dimensionality of hidden units in GNNs (default: 600)')
    parser.add_argument('--gru_emb', type=int, default=32,
                        help='GRU token embed size (default: 32)')
    parser.add_argument('--gru_hid', type=int, default=64,
                        help='GRU hidden size (default: 256)')
    parser.add_argument('--max_len', type=int, default=500,
                        help='')
    parser.add_argument('--train_subset', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--num_tasks', type=int, default=1,
                        help='num_labels, tox21: 12, PCQM4: 1')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default=f'ckpt/{timestamp}', help='directory to save checkpoint')
    args = parser.parse_args()

    print(args)



    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    train_dataset = torch.load("dataset/bbbp/processed/train_dataset.pt")
    valid_dataset = torch.load("dataset/bbbp/processed/valid_dataset.pt")
    test_dataset = torch.load("dataset/bbbp/processed/test_dataset.pt")

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator("ogbg-molbbbp")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.checkpoint_dir != '':
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    shared_params = {
        "num_tasks": args.num_tasks,
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

    best_valid_auc = 0
    best_epoch = 0

    scheduler = StepLR(optimizer, step_size=30, gamma=0.25)

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train_auc = train(model, device, train_loader, optimizer, evaluator)
        train_auc = train_auc["rocauc"]
        print('Evaluating...')
        valid_auc = eval(model, device, valid_loader, evaluator)
        valid_auc = valid_auc["rocauc"]
        print({'Train': train_auc, 'Validation': valid_auc})

        if args.log_dir != '':
            writer.add_scalar('valid/auc', valid_auc, epoch)
            writer.add_scalar('train/auc', train_auc, epoch)
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            best_epoch = epoch
            if args.checkpoint_dir != '':
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),
                              'scheduler_state_dict': scheduler.state_dict(), 'best_val_auc': best_valid_auc,
                              'num_params': num_params}
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'checkpoint.pt'))

            test_auc = eval(model, device, test_loader, evaluator)
            print(f"Test MAE: {test_auc}")
        scheduler.step()

        print(f'Best validation MAE so far: Epoch {best_epoch}: {best_valid_auc}')
    print(f"Test AUC: {test_auc}")
    if args.log_dir != '':
        writer.close()


if __name__ == "__main__":
    main()
