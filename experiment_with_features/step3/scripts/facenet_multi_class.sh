python3 step3.py \
    --save=True \
    --target_model=FaceNet \
    --img_size=160 \
	--identifier="${1}_${3}epoch_${4}batch_target${5}_lambda${6}" \
    --step2_dir=$1 \
    --gpu_idx=$2 \
    --epochs=$3 \
    --batch_size=$4 \
    --target_index=$5 \
    --lambda_i=$6

    

# ./scripts/facenet.sh facenet_proposed_5e-2_3_100 ~/nas/results/step2/facenet_proposed_5e-2 3 100 2