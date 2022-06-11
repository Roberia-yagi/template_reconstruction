#!/bin/bash

start=$1
step=$2
end=$3

for i in `seq $start $step $end`; do
	python find_best_GAN_param.py --gpu_idx 1 --learning_rate $i --identifier learning_rate_$i
done