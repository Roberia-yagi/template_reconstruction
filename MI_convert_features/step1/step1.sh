#!/bin/bash

k=5
for i in `seq 1 6`; do
	python step1_train_transferer_naive.py --gpu_idx 0 --target_dataset_dir /home/akasaka/nas/dataset/For_training_converter/CASIAWebFace_MTCNN160_FaceNet --attack_dataset_dir /home/akasaka/nas/dataset/For_training_converter/CASIAWebFace_MTCNN112_Arcface --target_model FaceNet --attack_model Magface --num_of_identities $k --num_per_identity 20
	k=$(($k*4))
done