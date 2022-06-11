dataset_dir=~/nas/dataset/CelebA_MTCNN160
python3 step1.py \
    --save=True \
    --n_epochs=0 \
    --gpu_idx=$1 \
    --identifier=$2 \
    --target_model=FaceNet \
    --dataset_dir="~/nas/dataset/CelebA_MTCNN160" \
    --classifier_mode="features"