#!/bin/bash

# Train GAN for attack

n=$1
n_epochs=$2
n_critic=$3
dataset_dir="${HOME}/share/dataset/CelebA"
result_dir="${HOME}/nas/results/step2"

for i in `seq 1 $n`; do
	python3 step2.py --save=True --n_epochs=$n_epochs --n_critic=$n_critic --dataset_dir=$dataset_dir --result_dir=$result_dir
done