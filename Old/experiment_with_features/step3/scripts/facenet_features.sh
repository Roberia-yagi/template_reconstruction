single_mode=$7 #single or multi

if [ "$single_mode" = "single" ]; then
	identifier="${1}_${3}epoch_single_${4}batch_target${5}_lambda${6}" \
    single_mode="True"
else
	identifier="${1}_${3}epoch_multiple_${4}batch_target${5}_lambda${6}" \
    single_mode="False"
fi

python3 step3.py \
    --save=True \
    --target_model=FaceNet \
    --img_size=160 \
    --identifier=$identifier\
    --step2_dir=$1 \
    --gpu_idx=$2 \
    --epochs=$3 \
    --batch_size=$4 \
    --target_dir_name=$5 \
    --lambda_i=$6 \
    --single_mode=$single_mode
    

# ./scripts/facenet.sh facenet_proposed_5e-2_3_100 ~/nas/results/step2/facenet_proposed_5e-2 3 100 2