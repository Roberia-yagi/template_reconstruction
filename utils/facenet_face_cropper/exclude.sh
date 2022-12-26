#!/bin/bash

dataset_dir=~/nas/dataset/CelebA_MTCNN64

while read line
do
    echo $line
    sed -i -e "/$line/d" $dataset_dir/identity_CelebA.txt
    sed -i -e "/$line/d" $dataset_dir/list_eval_partition.txt
done < not_cropped.txt