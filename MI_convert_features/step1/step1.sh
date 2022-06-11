#!/bin/bash

start=$1
step=$2
end=$3

for i in `seq $start $step $end`; do
	python step1_train_transferer_naive.py --gpu_idx 1 --num_of_identities $i --num_per_identity 10
done