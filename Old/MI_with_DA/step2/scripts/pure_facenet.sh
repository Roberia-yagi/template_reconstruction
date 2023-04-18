identifier=$1
n_epochs=$2
step1_dir=$3
gpu_idx=$4
python3 step2.py \
	--save=True \
	--method=pure \
	--identifier=$identifier \
	--target_model=FaceNet \
	--img_size=160 \
	--n_epochs=$n_epochs \
	--step1_dir=$step1_dir \
	--gpu_idx=$gpu_idx
