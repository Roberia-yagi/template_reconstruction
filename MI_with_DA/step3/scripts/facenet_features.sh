single_mode=$6 #single or multi

if [ "$single_mode" = "single" ]; then
	identifier="${1}_${2}epoch_single_target${3}_lambda${4}_target_model_${5}" \
    single_mode="True"
else
	identifier="${1}_${2}epoch_multiple_target${3}_lambda${4}_target_model_${5}" \
    single_mode="False"
fi

python3 step3.py \
    --save=True \
    --img_size=160 \
    --identifier=$identifier\
    --step2_dir=$1 \
    --epochs=$2 \
    --target_dir_name=$3 \
    --lambda_i=$4 \
    --single_mode=$single_mode

# ./scripts/facenet.sh facenet_proposed_5e-2_3_100 ~/nas/results/step2/facenet_proposed_5e-2 3 100 2