import os
import os.path as osp
import shutil
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from pre_process import PygPCQM4Mv2Dataset_SMILES_ADDED, QM9, MoleculeNet, PygGraphPropPredDataset
from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_scaffold(smiles, include_chirality=False):
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality
    )
    return scaffold


def scaffold_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=None):
    print("generating scaffold......")

    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, data in enumerate(dataset):
        smiles = data.smiles
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest, first sort the dictionary's
    # value list according the value from small -> large
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}

    # according to the tuple : (len(x[1]), x[1][0]) to sort the len reflects the
    # molecule size the x[1][0] reflects the smallest index in this molecular set
    all_scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
        )
    ]

    # get train, valid test indices
    train_cutoff = frac_train * len(dataset)
    valid_cutoff = (frac_train + frac_valid) * len(dataset)
    train_idx, valid_idx, test_idx = [], [], []

    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0
    trn_df = dataset[train_idx]
    val_df = dataset[valid_idx]
    test_df = dataset[test_idx]

    return trn_df, val_df, test_df


if __name__ == "__main__":
    pyg_dataset = PygPCQM4Mv2Dataset_SMILES_ADDED(smiles2graph=smiles2graph)
    # qm9_dataset = QM9(root='dataset/qm9')
    # hiv_MoleculeNet = MoleculeNet(root='dataset', name='hiv')
    # tox21_MoleculeNet = MoleculeNet(root='dataset', name='tox21')
    # trn_tox21, val_tox21, test_tox21 = scaffold_split(tox21_MoleculeNet)
    # torch.save(trn_tox21, 'dataset/tox21/processed/train_dataset.pt')
    # torch.save(val_tox21, 'dataset/tox21/processed/valid_dataset.pt')
    # torch.save(test_tox21, 'dataset/tox21/processed/test_dataset.pt')
    # bace_MoleculeNet = MoleculeNet(root='dataset', name='bace')
    # trn_bace, val_bace, test_bace = scaffold_split(bace_MoleculeNet)
    #
    bbbp_MoleculeNet = MoleculeNet(root='dataset', name='bbbp')
    # trn_bbbp, val_bbbp, test_bbbp = scaffold_split(bbbp_MoleculeNet)
    # torch.save(trn_bbbp, 'dataset/bbbp/processed/train_dataset.pt')
    # torch.save(val_bbbp, 'dataset/bbbp/processed/valid_dataset.pt')
    # torch.save(test_bbbp, 'dataset/bbbp/processed/test_dataset.pt')

    # clintox_MoleculeNet = MoleculeNet(root='dataset', name='clintox')
    # sider_MoleculeNet = MoleculeNet(root='dataset', name='sider')
    # hiv_dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='dataset')
    print("Done Done!")
    # a = []
    # for i, mol in enumerate(pyg_dataset):  # 나눠서 할것. 좀 오래걸림
    #     a.extend(mol.x[:, 0].tolist())
    #     a = list(set(a))
    a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 25, 27, 28, 29, 30, 31, 32, 33,
         34, 35]
    print("PCQM4:")  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    print("len(pyg_dataset): ", len(pyg_dataset))
    print("pyg_dataset[123]: ", pyg_dataset[123])
    print()
    #
    # a = []
    # for i, mol in enumerate(qm9_dataset):
    #     a.extend(mol.z.tolist())
    #     a = list(set(a))
    # a = [1, 6, 7, 8, 9]
    # print("QM9:", a)  # --> 5개 1,6,7,8,9
    # print(len(qm9_dataset))
    # print(qm9_dataset[0])
    # a = []
    # for i, mol in enumerate(hiv_MoleculeNet):
    #     a.extend(mol.x[:, 0].tolist())
    #     a = list(set(a))
    # a = [1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    #      35,
    #      40, 42, 44, 45, 46, 47, 50, 51, 52, 53, 55, 64, 65, 67, 74, 75, 77, 78, 79, 80, 81, 82, 83, 89, 92]
    # print("HIV:", a)  #
    # print(len(hiv_MoleculeNet))
    # print(hiv_MoleculeNet[0])

    # a = []
    # for i, mol in enumerate(bace_MoleculeNet):
    #     a.extend(mol.x[:, 0].tolist())
    #     a = list(set(a))
    # a = [6, 7, 8, 9, 16, 17, 35, 53]
    # print("bace:", a)  #
    # print(len(bace_MoleculeNet))
    # print(bace_MoleculeNet[0])
    #
    # # a = []
    # for i, mol in enumerate(bbbp_MoleculeNet):
    #     a.extend(mol.x[:, 0].tolist())
    #     a = list(set(a))
    a = [1, 35, 5, 6, 7, 8, 9, 11, 15, 16, 17, 20, 53]
    print("bbbp:")  #
    print("len(bbbp_MoleculeNet): ", len(bbbp_MoleculeNet))
    print("bbbp_MoleculeNet[0]: ", bbbp_MoleculeNet[0])

    # a = []
    # for i, mol in enumerate(clintox_MoleculeNet):
    #     a.extend(mol.x[:, 0].tolist())
    #     a = list(set(a))
    # a = [0, 1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 20, 22, 24, 25, 26, 29, 30, 33, 34, 35, 43, 53, 78, 79, 80, 81, 83]
    # print("clintox:", a)  #
    # print(len(clintox_MoleculeNet))
    # print(clintox_MoleculeNet[0])

    # a = []
    # for i, mol in enumerate(sider_MoleculeNet):
    #     a.extend(mol.x[:, 0].tolist())
    #     a = list(set(a))
    # a = [1, 3, 5, 6, 7, 8, 9, 11, 12, 15, 16, 17, 19, 20, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 38, 39, 43, 47,
    #      49, 53, 56, 57, 62, 64, 78, 79, 81, 88, 98]
    # print("sider:", a)  #
    # print(len(sider_MoleculeNet))
    # print(sider_MoleculeNet[0])

    # a = []
    # for i, mol in enumerate(tox21_MoleculeNet):
    #     a.extend(mol.x[:, 0].tolist())
    #     a = list(set(a))
    # a = [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34,
    #      35, 38, 40, 42, 46, 47, 48, 49, 50, 51, 53, 56, 60, 64, 66, 70, 78, 79, 80, 81, 82, 83]
    # print("tox21:", a)  #
    # print(len(tox21_MoleculeNet))
    # print(tox21_MoleculeNet[0])
