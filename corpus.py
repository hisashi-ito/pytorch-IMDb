#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#【corpus】
#
# 概要: IMDb 関連のデータ処理を実施
#
# 更新履歴:
#           2019.03.30 新規作成
#
import pathlib
import torch
from tokenizer import Tokenizer

class Corpus(object):
    def __init__(self, file):
        self.vocab = self._load_vocab(file)
        self.tokenizer = Tokenizer()
        
    @property
    def vocab_size(self):
        return len(self.vocab)
    
    # IMDb が用意してくている語彙辞書を読み込み
    def _load_vocab(self, file):
        voc = {}
        f = open(file, "r")
        for line in f:
            try:
                v = line.rstrip()
                # index は 0 を明けておいて
                # 1から開始する
                voc[v] = len(voc) + 1
            except ValueError:
                continue
        f.close()
        return voc

    # text 情報をidsリストへ変換する
    # tokenize はnltk を利用
    def text2ids(self, text):
        tokens = self.tokenizer.tokenize(text)
        # [TIPS] self.voccab に存在しない場合はindexを0にする
        return [self.vocab.get(token, 0) for token in tokens]

    # ids化されたtextをTensor化(0-paddingも実施)
    def list2tensor(self, token_indexs, max_len = 100):
        # 最初のtoken長を保存
        n_tokens = len(token_indexs)
        if len(token_indexs) > max_len:
            token_indexs = token_indexs[0:max_len]
        else:
            # 0-padding 
            token_indexs = token_indexs + [0] * (max_len - len(token_indexs))
        # pytorch ではindxのidは'int64'にする必要がある
        return torch.tensor(token_indexs, dtype=torch.int64), n_tokens


if __name__ == '__main__':
    voc_dic = "./aclImdb/imdb.vocab"
    text = "I saw 'Liz and the blue bird' at Shinjuku Piccadilly."
    c = Corpus(voc_dic)
    token_indexs = c.text2ids(text)
    print(c.list2tensor(token_indexs))
