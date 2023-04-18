import sys
import subprocess

def main():
	args = sys.argv

	if len(args) != 2:
		print(f"Usage  : {args[0]} {{GPU index}}")
		print(f"Example: {args[0]} 0")
		sys.exit(1)

	gpu_idx = args[1]

	identifiers = [
		# FaceNet, existing
		'facenet_existing_5e-3',
		# 'facenet_existing_5e-4',
		# FaceNet, proposed
		'facenet_proposed_1e-0',
		# 'facenet_proposed_7.5e-1',
		# 'facenet_proposed_5e-1',
		# 'facenet_proposed_2.5e-1',
	]

	lambda_i_list = [10]

	# lr_list = [0.02, 0.002]
	# lr_list = [0.2, 0.02, 0.002]
	lr_list = [0.02]

	times = [1]

	# epochs = [1, 10, 100]
	epochs = [1]

	for identifier in identifiers:
		for lambda_i in lambda_i_list:
			for lr in lr_list:
				for time in times:
					for epoch in epochs:
						subprocess.run([
							"bash",
							"./scripts/facenet.sh",
							f"test_{identifier}_{time}times_{epoch}epochs_lambda{lambda_i}_lr{lr}",
							f"~/nas/results/step2/{identifier}",
							str(lambda_i),
							str(lr),
							str(time),
							str(epoch),
							str(gpu_idx),
						])
	

main()