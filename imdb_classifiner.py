#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#【imdb_classifiner】
#
# 概要: IMDb データのPOS/NEG判定器メイン関数
#       本コマンド本ディレクトリ配下にあるnetwork.py のネットワーク設定にもとづき
#       IMDb データのpos,neg判定を実施すするためのコード郡を提供する
#
# usage: imdb_classifiner --mode <train>            (モード指定,trainのみ動作)
#                         -i <dir_path>             (IMDbデータパス)
#                         --epoch_num <epoch_num>   (エポック数)
#                         --batch_size <batch_size> (バッチサイズ)
#                         --worker <worker_num>     (前処理で利用するCPU数)
#  更新履歴:
#           2019.03.31 新規作成
# 
import sys
import argparse
import logging
from trainer import Trainer

def main():
    # logger の設定
    logging.basicConfig(level=logging.INFO,format='[%(asctime)s] %(levelname)s -- : %(message)s')
    logger = logging.getLogger()
    parser = argparse.ArgumentParser(description='This code is IMDb data set classifiner.')
    parser.add_argument("--mode", required=True, help='mode')
    parser.add_argument("-i", required=True, help='input dir')
    parser.add_argument("--epoch_num", required=True, help='epoch num')
    parser.add_argument("--batch_size", required=True, help='batch size')
    parser.add_argument("--worker", required=True, help='worker num')
    args = parser.parse_args()
    logger.info("*** start imdb_classifiner ***")
    if args.mode == "train":
        # 学習
        logger.info("train mode")
        t = Trainer(logger, args.i, int(args.epoch_num), int(args.batch_size), int(args.worker))
        t.fit()
    elif args.mode == "evaluate":
        # evaluate 未実装
        pass
    else:
        # predict 未実装
        pass

    
if __name__ == '__main__':
    main()

