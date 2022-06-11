dataset_dir=~/nas/dataset/CelebA_MTCNN160
python3 step1.py \
    --save=True \
    --n_epochs=$1 \
    --gpu_idx=$2 \
    --identifier=$3 \
    --target_model=FaceNet \
    --dataset_dir="~/nas/dataset/CelebA_MTCNN160" \
    --classifier_mode="multi_class"