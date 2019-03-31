#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#【imdb_dataset】
#
# 概要: IMDb用のデータセットクラス
#
# 更新履歴:
#          2019.3.31 新規作成
#          2019.3.31 corpus インスタンスを引数からわたして利用
#
import pathlib
import glob
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from corpus import Corpus

# pytorh のDatassetを継承して独自Datasetを作成
class IMDBDateset(Dataset):
    def __init__(self, dir_path, corpus, train = True, max_len = 100):
        self.max_len = max_len
        self.path = pathlib.Path(dir_path)
        self.corpus = corpus
        if train:
            target_path = self.path.joinpath("train")
        else:
            target_path = self.path.joinpath("test")
        pos_files = sorted(glob.glob(str(target_path.joinpath("pos/*.txt")))) # positive な評価
        neg_files = sorted(glob.glob(str(target_path.joinpath("neg/*.txt")))) # negative な評価
        # 0:neg, 1:pos のラベルを付与
        self.labeled_files = list(zip([0]*len(neg_files), neg_files)) + list(zip([1]*len(pos_files), pos_files))
        
    def __getitme__(self, idx):
        label, f = self.labeled_files[idx]
        text = open(f).read().lower()
        ids = self.corpus.text2ids(text, self.vocab)
        data, n_tokens = self.corpus.list2tensor(ids, self.max_len)
        return data, label, n_tokens
    
    def __len__(self):
        len(self.labeled_files)


if __name__ == '__main__':
    dir_path = "./aclImdb"
    train_data = IMDBDateset(dir_path, train = True, max_len = 100)
    test_data  = IMDBDateset(dir_path, train = False, max_len = 100)
    
    
