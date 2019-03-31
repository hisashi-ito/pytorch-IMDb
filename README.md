# pytorch-IMDb classifier
IMDb classifier using pytorch

## install
```bash
$ git clone https://github.com/hisashi-ito/pytorch-IMDb.git
```
## train
```bash
$ cd ./pytorch-IMDb
$ ./imdb_classifiner.sh
```
* usage
```bash
usage: imdb_classifiner --mode <train>             (mode only train)
                         -i <dir_path>             (IMDb data pash)
                         --epoch_num <epoch_num>   (number of epoch)
                         --batch_size <batch_size> (number of bachsize)
                         --worker <worker_num>     (number of workers for makaing dataset)
```

## dataset  
Large Movie Review Dataset  
http://ai.stanford.edu/~amaas/data/sentiment/
