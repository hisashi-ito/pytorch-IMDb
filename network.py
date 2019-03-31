#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#【network】
#
# 概要: network(model) 定義
#
# 更新履歴:
#           2019.03.31 新規作成
#
import torch
from torch import nn

class SequenceTaggingNet(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim = 50,
                 hidden_size = 50,
                 num_layers = 2,
                 dropout = 0.2):
        super().__init__()
        
        # 1) Embedding layer
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx = 0)
        
        # 2) LSTM layer
        #    batch_first = True のoptionは重要で入力Tensorが
        #    [batch_size, sequence_len, embedding_dim]
        #    指定しない場合は
        #    [sequence_len, batch_size, embedding_dim]
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first = True, dropout = dropout)
        
        # 3) FC layer
        #    2値分類なので出力次元を1とする
        self.linear = nn.Linear(hidden_size, 1)
        
    # 順伝搬関数
    # 明示的に__call__ 関数と同じ意味で自動的に呼び出される
    # chainer も昔は forward 関数だったがまだ残っている...
    # step_size はsequence_len と同じ意味
    def forward(self, x, h0 = None, l = None):
        # 1) Embedding
        #    id を embeddingで多次元ベクトルに変換する
        #    x(batch_size, step_size)
        #    -> x(batch_size, step_size, embedding_dim)
        x = self.emb(x)
        
        # 2) LSTM
        #    h0とともにxをLSTMへ通す
        #    x(batch_size, step_size, embedding_dim)
        #    -> x(batch_size, step_size, hidden_size)
        x, h = self.lstm(x, h0)
        x = x[:,-1,:]
        
        # 3) FC
        # 余分な次元を削除する
        # x(batch_size, 1) -> x(batch_size,)
        # squeezeは上記のように要素数が1だけの軸を削除する
        x = self.linear(x).squeeze()
        return x
