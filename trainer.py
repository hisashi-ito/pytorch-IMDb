#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#【trainer】
#
# 概要: IMDB 推論 学習用クラス
#
# 更新履歴:
#          2019.03.31 新規作成
#
import tqdm
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from imdb_dataset import IMDBDateset

class Trainer(object):
    def __init__(self, dir_path, max_len = 100):
        self.train_data = IMDBDateset(dir_path, train = True, max_len = max_len)
        self.test_data  = IMDBDateset(dir_path, train = False, max_len = max_len)
        
