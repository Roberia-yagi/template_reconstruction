n_epochs=100
dataset_dir="${HOME}/share/dataset/CelebA"
result_dir="${HOME}/nas/results/step2"
step1_dir="${HOME}/nas/results/step1/2021_06_30_05_00"
iterations=10

for i in $(seq $iterations); do
    echo "Iteration: $i"

    # Pure method
    # python3 step2.py --method=pure --save=True --n_epochs=$n_epochs --dataset_dir=$dataset_dir --result_dir=$result_dir --step1_dir=$step1_dir

    # Proposed method
    # python3 step2.py --method=proposed --beta=0.0001 --save=True --n_epochs=$n_epochs --dataset_dir=$dataset_dir --result_dir=$result_dir --target_model_path=$target_model_path --label_map_path=$label_map_path
    # python3 step2.py --method=proposed --beta=0.0005 --save=True --n_epochs=$n_epochs --dataset_dir=$dataset_dir --result_dir=$result_dir --target_model_path=$target_model_path --label_map_path=$label_map_path
    # python3 step2.py --method=proposed --beta=0.001 --save=True --n_epochs=$n_epochs --dataset_dir=$dataset_dir --result_dir=$result_dir --target_model_path=$target_model_path --label_map_path=$label_map_path
    # python3 step2.py --method=proposed --beta=0.005 --save=True --n_epochs=$n_epochs --dataset_dir=$dataset_dir --result_dir=$result_dir --target_model_path=$target_model_path --label_map_path=$label_map_path
    # python3 step2.py --method=proposed --beta=0.01 --save=True --n_epochs=$n_epochs --dataset_dir=$dataset_dir --result_dir=$result_dir --step1_dir=$step1_dir
    # python3 step2.py --method=proposed --beta=0.05 --save=True --n_epochs=$n_epochs --dataset_dir=$dataset_dir --result_dir=$result_dir --step1_dir=$step1_dir
    # python3 step2.py --method=proposed --beta=0.1 --save=True --n_epochs=$n_epochs --dataset_dir=$dataset_dir --result_dir=$result_dir --step1_dir=$step1_dir
    # python3 step2.py --method=proposed --identifier=proposed_beta5e-1 --beta=0.5 --save=True --n_epochs=$n_epochs --dataset_dir=$dataset_dir --result_dir=$result_dir --step1_dir=$step1_dir

    # Existing method
    # python3 step2.py --method=existing --identifier=existing_1e-4 --alpha=0.0001 --save=True --n_epochs=$n_epochs --dataset_dir=$dataset_dir --result_dir=$result_dir --step1_dir=$step1_dir
    # python3 step2.py --method=existing --identifier=existing_alpha5e-4 --alpha=0.0005 --save=True --n_epochs=$n_epochs --dataset_dir=$dataset_dir --result_dir=$result_dir --step1_dir=$step1_dir
    # python3 step2.py --method=existing --alpha=0.001 --save=True --n_epochs=$n_epochs --dataset_dir=$dataset_dir --result_dir=$result_dir --step1_dir=$step1_dir
    # python3 step2.py --method=existing --alpha=0.005 --save=True --n_epochs=$n_epochs --dataset_dir=$dataset_dir --result_dir=$result_dir --step1_dir=$step1_dir
    # python3 step2.py --method=existing --alpha=0.01 --save=True --n_epochs=$n_epochs --dataset_dir=$dataset_dir --result_dir=$result_dir --target_model_path=$target_model_path --label_map_path=$label_map_path
    # python3 step2.py --method=existing --alpha=0.05 --save=True --n_epochs=$n_epochs --dataset_dir=$dataset_dir --result_dir=$result_dir --target_model_path=$target_model_path --label_map_path=$label_map_path
    # python3 step2.py --method=existing --alpha=0.10 --save=True --n_epochs=$n_epochs --dataset_dir=$dataset_dir --result_dir=$result_dir --target_model_path=$target_model_path --label_map_path=$label_map_path
    # python3 step2.py --method=existing --alpha=0.50 --save=True --n_epochs=$n_epochs --dataset_dir=$dataset_dir --result_dir=$result_dir --target_model_path=$target_model_path --label_map_path=$label_map_path
done
