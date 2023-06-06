#!/bin/bash

start=$1
step=$2
end=$3

for i in `seq $start $step $end`; do
	python step1_train_transferer_PCA.py --gpu_idx 0 --num_of_images $i
done