dataset_dir=~/nas/dataset/CelebA_MTCNN160
n_epochs=$1
gpu_idx=$2
python3 step1.py --save=True --identifier=resnet152_mtcnn --target_model=ResNet152 --n_epochs=$n_epochs --gpu_idx=$gpu_idx --dataset_dir=$dataset_dir