# pytorch-IMDb classifier
IMDb classifier using pytorch

### インストール方法
```bash
$ git clone https://github.com/hisashi-ito/pytorch-IMDb.git
```
### 学習方法  
```bash
$ cd ./pytorch-IMDb
$ ./imdb_classifiner.sh
```
上記のシェルをキックするだけで学習が始まります。内部的に`imdb_classifiner.py`をキックします。データ類は既に設置済み。また動作環境はGPU環境を想定しています。

* コマンド引数
```bash
usage: imdb_classifiner --mode <train>             (mode only train)
                         -i <dir_path>             (IMDb data pash)
                         --epoch_num <epoch_num>   (number of epoch)
                         --batch_size <batch_size> (number of bachsize)
                         --worker <worker_num>     (number of workers for makaing dataset)
```
現在は動作モードが `train` のみ実装済みです。

### 学習結果  
エポック数30回で以下のような学習状況となります。lossは速やかに減少しています。また`val_acc` が `75[%]` 程度が限界のようです。
<p align="center">
<img src="https://user-images.githubusercontent.com/8604827/55297697-ced3e700-5463-11e9-9c2f-cc7dd942277f.png" width="550px">
</p>

### データセット   
Large Movie Review Dataset  
http://ai.stanford.edu/~amaas/data/sentiment/
