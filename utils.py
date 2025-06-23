import gzip
import os
import time
import math
import logomaker
import itertools
import argparse
import warnings
import pickle
import random
import datetime
import copy
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from Bio import PDB, Seq, SeqIO
from tqdm import tqdm
from sklearn import metrics
from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
from torch.nn.functional import relu, softmax, unfold
from typing import Any, Tuple, List, Dict, Set, Optional, Union
from typing_extensions import Literal
from scipy.stats import hypergeom, gaussian_kde


def do_nothing(*args):
    pass


class Loader:
    def __init__(self, filename: str):
        self.filename = filename

    def load_npy(self):
        return np.load(self.filename, allow_pickle=True)

    def load_pkl(self) -> Any:
        with open(self.filename, 'rb') as f:
            variable_ = pickle.load(f)
        return variable_

    def load_pth(self) -> Any:
        variable_ = torch.load(self.filename, map_location=torch.device('cpu'), weights_only=False)
        return variable_

    def load_txt_gz(self) -> pd.DataFrame:
        with gzip.open(self.filename, 'rt') as f:
            df = pd.read_csv(f, sep='\t')
        return df


class Saver:
    def __init__(self, variable: Any, filename: str):
        self.variable = variable
        self.filename = filename

    def save_npy(self):
        np.save(self.filename, self.variable)

    def save_pkl(self):
        f: Any  # avoid the Type checking
        with open(self.filename, 'wb') as f:
            pickle.dump(self.variable, f)

    def save_pth(self):
        torch.save(self.variable, f=self.filename)

    def save_str(self):
        with open(self.filename, 'w') as file:
            file.write(self.variable)

    def save_pdb(self):
        with open(self.filename, 'w') as f:
            f.write(self.variable)


def addPos(pos_len: int, channel_len: int):
    # generate Position Profile
    seq_pos = []
    for pos in range(pos_len):
        pos_lst = []
        for channel in range(channel_len):
            if channel % 2 == 0:
                pos_lst.append(math.sin(pos / (10000 ** (channel / channel_len))))
            else:
                pos_lst.append(math.cos(pos / (10000 ** ((channel - 1) / channel_len))))
        seq_pos.append(pos_lst)
    return np.array(seq_pos) # pos, channel


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, pos_threshold: float = 0.5) -> Any:
    pred_label = (preds.detach().cpu() >= pos_threshold).int().numpy()
    pred_score = preds.detach().cpu().numpy()
    true_label = targets.detach().cpu().numpy()
    try:
        tn, fp, fn, tp = metrics.confusion_matrix(true_label, pred_label).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        if (tp + fn == 0) or (tp + fp == 0):
            sensitivity, precision = 0, 0
        else:
            sensitivity, precision = tp / (tp + fn), tp / (tp + fp)
        roc_auc = metrics.roc_auc_score(y_true=true_label, y_score=pred_score)
        f1 = metrics.f1_score(pred_label, true_label)
        precision_lst, recall_lst, _ = metrics.precision_recall_curve(true_label, pred_label)
        pr_auc = metrics.auc(recall_lst, precision_lst)
    except Exception as e:
        acc, sensitivity, roc_auc, f1, precision, pr_auc = 0, 0, 0, 0, 0, 0
        print(e)
    return acc, sensitivity, roc_auc, f1, precision, pr_auc


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def alter_string_at_pos(strings: str, pos: int, new_char: str) -> str:
    # raw_str = list(strings)
    # raw_str[pos] = new_char
    # return ''.join(raw_str)
    return strings[:pos] + new_char + strings[pos + 1:]


def get_device(device_id: Union[Literal['cpu'], int, None]):
    if torch.cuda.is_available() and isinstance(device_id, int):
        return torch.device(f'cuda:{device_id}')
    else:
        return torch.device('cpu')


def get_main_device(device_id: Union[Literal['cpu'], int, List, None]) -> torch.device:
    if not torch.cuda.is_available():
        return get_device('cpu')
    else:
        if isinstance(device_id, list):
            return get_device(device_id[0])
        else:
            return get_device(device_id)


def load_str_batch(batch_size: int, feature: List[str], label,
                   label_mask=None, s=None, drop_last: bool = False):
    # return: feature_str, label, label_anno (None), s (None)
    instance_num = len(feature)
    batches = math.ceil(len(feature) / batch_size)
    padding_0 = torch.zeros(instance_num)
    label = label.float()
    label_mask = padding_0 if label_mask is None else label_mask.float()
    s = padding_0 if s is None else s.transpose(dim0=1, dim1=2).float()
    all_indices = list(range(instance_num))
    random.shuffle(all_indices)
    data_loader = []
    for i in range(batches):
        batch_indice = all_indices[i*batch_size: (i+1)*batch_size]
        batch_fea = [feature[j] for j in batch_indice]
        # [batch_feature, batch_label, batch_label_anno, batch_s]
        batch_data = [batch_fea, label[batch_indice], label_mask[batch_indice], s[batch_indice]]
        data_loader.append(batch_data)
    if drop_last:
        data_loader = data_loader[:-1]
    return data_loader
