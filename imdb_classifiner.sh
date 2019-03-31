#! /bin/bash
cmd="./imdb_classifiner.py"
mode="train"
input="./aclImdb"
epoch_num=30
batch_size=32
worker=4
main_cmd="${cmd} -i ${input} --mode ${mode} --epoch_num ${epoch_num} --batch_size ${batch_size} --worker ${worker}"
echo ${main_cmd}
eval ${main_cmd}
