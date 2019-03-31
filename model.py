#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#【model】
#
# 概要: model(network) 定義
#
# 更新履歴:
#           2019.03.31 新規作成
#
import torch
from torch import nn, optim

class SequenceTaggingNet(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim = 300,
                 hidden_size = 128,
                 num_layers = 1,
                 dropout = 0.5):
        super().__init__()
        # 1) Embedding layer
        self.emb  = nn.Embedding(num_embeddings, embedding_dim, paddinx_idx = 0)

        # 2) LSTM layer
        #    batch_first = True のoptionは重要で入力Tensorが
        #    [batch_size, sequence_len, embedding_dim]
        #    指定しない場合は
        #    [sequence_len, batch_size, embedding_dim]
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first = True, dropout = dropout)

        # 3) FC layer
        #    2値分類なので出力次元を1とする
        self.linear = nn.Linear(hidden_size, 1)

    # この関数はsequence毎に呼ばれることに注意
    # [1,2,3] => 1,2,3 と3回 fowordされる
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

        if l is not None:
            # [TIPS] 入力の長さがある場合はそれを利用(重要)
            #        0-paddingされている箇所を利用しない
            x = x[list(range(len(x))), l-1,:]
        else:
            # 最後のステップを保存
            x = x[:,-1,:]
        # 3) FC
        x = self.linear(x)
        # 余分な次元を削除する
        # x(batch_size, 1) -> x(batch_size,)
        # squeezeは上記のように要素数が1だけの軸を削除する
        x = x.squeeze()
        return x
