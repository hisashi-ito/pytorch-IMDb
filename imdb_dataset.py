#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#【imdb_dataset】
#
# 概要: IMDb用のデータセットクラス
#
# 更新履歴:
#          2019.3.31 新規作成
#
import tqdm
import pathlib
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from corpus import Corpus

# pytorh のDatassetを継承して独自Datasetを作成
class IMDBDateset(Dataset):
    def __init__(self, dir_path, train = True, max_len = 100):
        self.max_len = max_len
        self.path = pathlib.Path(dir_path)
        self.corpus = Corpus(self.path.joinpath("imdb.vocab"))
        
        
        

        
if __name__ == '__main__':
    dir_path = "./aclImdb"
    IMDBDateset(dir_path)
