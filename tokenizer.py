#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#【tokenizer】
#
# 概要: NLTK(https://www.nltk.org/) を利用したtokenizer を実装します
#       NLTKの parser を利用時は事前に以下のコマンドを実行しておく
#
#       $ python3
#       >>> import nltk
#       >>> nltk.download('punkt')
#
# 更新履歴:
#          2019.03.30 新規作成
#
import nltk

class Tokenizer(object):
    def __init__(self):
        pass
    
    def tokenize(self, text):
        try:
            tokens = nltk.word_tokenize(text)
        except:
            raise Exception("NLTK 形態素解析で失敗しました: {0}".format(text))
        forms = []
        for token in tokens:
            if len(token) == 0:
                continue
            forms.append(token)
        return forms

    
if __name__ == '__main__':
     text = "I saw 'Liz and the blue bird' at Shinjuku Piccadilly."
     m = Tokenizer()
     ret = m.tokenize(text)
     # ['I', 'saw', "'Liz", 'and', 'the', 'blue', 'bird', "'", 'at', 'Shinjuku', 'Piccadilly', '.']
     print(ret)
