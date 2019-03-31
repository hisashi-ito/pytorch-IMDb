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
import pathlib
import tqdm
import logging
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from imdb_dataset import IMDBDateset
from corpus import Corpus
from network import SequenceTaggingNet

class Trainer(object):
    # GPU,CPU の別名表現
    GPU = 'cuda:0'
    CPU = 'cpu'
    
    def __init__(self, logger, dir_path, epoch_num, batch_size = 32, num_workers = 4, max_len = 100):
        self.logger = logger
        self.path = pathlib.Path(dir_path)
        self.batch_size = int(batch_size)
        self.epoch_num = int(epoch_num)
        self.corpus = Corpus(str(self.path.joinpath("imdb.vocab")))

        # dataset の定義
        train_data = IMDBDateset(dir_path, self.corpus, train = True, max_len = int(max_len))
        test_data  = IMDBDateset(dir_path, self.corpus, train = False, max_len = int(max_len))
        
        # data loader の定義
        self.train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = num_workers)
        self.test_loader  = DataLoader(test_data,  batch_size = batch_size, shuffle = False, num_workers = num_workers)
    
        # モデルを作成
        # num_enbeddings = self.corpus.vocab_size+1 0次元も必要なので+1で初期化
        # LSTM は2-layerをstackする. ネットワークの設定は一旦固定
        self.net = SequenceTaggingNet(self.corpus.vocab_size+1,
                                      embedding_dim = 50,
                                      hidden_size = 50,
                                      num_layers = 2,
                                      dropout = 0.2)
        # network(model)をGPUへ転送
        self.net.to(self.GPU)
        self.opt = optim.Adam(self.net.parameters())
        # https://pytorch.org/docs/stable/nn.html#bceloss
        self.loss_func = nn.BCEWithLogitsLoss()

    # 評価関数(eval)
    # 一応trainerの内部だけで動作するようにするので
    # インスタンス変数を適宜利用して動作するようにする
    # defaultの動作ではCPUで駆動する
    def eval(self, data_loader, device):
        # 評価時の動作を指定
        self.net.eval()
        ys = []
        ypreads = []
        for x, y, l in data_loader:
            x = x.to(device)
            y = y.to(device)
            l = l.to(device)
            with torch.no_grad():
                y_pred = self.net(x, l=l)
                # bin値(0:neg,1:pos)に変換している
                # label もlong (int 64bit longにcast)
                y_pred = (y_pred >= 0).long()
                ys.append(y)
                ypreads.append(y_pred)
                # 配列をflat にする
        ys = torch.cat(ys)
        ypreads = ypreads.cat(ypreads)
        # 推論が一致している場合は1 なのでそれをまとめる
        # 最後の個数で割って平均の正解率(accracy)を算出
        acc = (ys == ypreads).float().sum() / len(ys)    
        # スカラー値を返却
        return acc.item()
        
    # 学習関数(fit)
    def fit(self):
        # epoch loop
        for epoch in range(self.epoch_num):
            losses = []
            # https://pytorch.org/docs/stable/nn.html
            self.net.train()
            
            for x,y,l in tqdm.tqdm(self.train_loader):
                x = x.to(self.GPU)
                y = y.to(self.GPU)
                l = l.to(self.GPU) # パティング前のsequence長
                # foword処理
                y_pred = self.net(x, l=l)
                # 損失値を計算 (yはfloatにcastが必要のようだ)
                loss = self.loss_func(y_pred, y.float())
                # 勾配を一旦初期化
                self.net.zero_grad()
                # 勾配を計算(backwardメソッドはlossのメソッド)
                loss.backward()
                # 重みを更新
                self.opt.step()
                # https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
                # loss.item() gets the a scalar value held in the loss.
                losses.append(loss.item())
                
            # train 評価
            train_acc = self.eval(self.train_loader)
            # validation 評価
            val_acc = self.eval(self.test_loader)
            # 学習状況をloggerで出力
            self.logger.info("ecpoh: {}, loss: {}, train_acc: {}, val_acc: {}".format(str(epoch), str(mean(losses)), str(train_acc), str(val_acc)))
