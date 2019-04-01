#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#【svm】
#
# 概要: SVM(BoW)を利用した推論
#
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression

train_x, train_y = load_svmlight_file("../aclImdb/train/unsupBow.feat")
#test_x, test_y = load_svmlight_file("../aclImdb/test/unsupBow.feat", n_features=train_x.shape[1])

model = LogisticRegression(C=0.1, max_iter=10)
model.fit(train_x, train_y)

# 推論(評価)
# model.score(train_x, train_y), model.score(test_x, test_y)
