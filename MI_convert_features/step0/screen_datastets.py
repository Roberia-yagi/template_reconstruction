import sys
import shutil
import os
sys.path.append('../')
sys.path.append("../../")
from glob import glob
from utils.util import remove_path_prefix, resolve_path
import random

# LFWA
# dataset_facenet_path = '/home/akasaka/nas/dataset/LFWA/lfw-deepfunneled-MTCNN160'
# dataset_arcface_path = '/home/akasaka/nas/dataset/LFWA_MTCNN112_Arcface'
# result_dir = '/home/akasaka/nas/dataset/For_experiment/LFWA'

# CASIA-Webface
# dataset_facenet_path = '/home/akasaka/nas/dataset/CASIAWebFace_MTCNN160_Facenet'
# dataset_arcface_path = '/home/akasaka/nas/dataset/CASIAWebFace_MTCNN112_Arcface'
# result_dir = '/home/akasaka/nas/dataset/For_experiment/CASIA'

# ColorFeret
dataset_facenet_path = '/home/akasaka/nas/dataset/colorferet_image_MTCNN160_FaceNet'
dataset_arcface_path = '/home/akasaka/nas/dataset/colorferet_image_MTCNN112_Arcface'
result_dir = '/home/akasaka/nas/dataset/For_experiment/colorferet'


identity_list = []
image_path_list_facenet = []
image_path_list_arcface = []
random.seed(42)

def main():
    # Create folders for storing results
    os.makedirs(result_dir, exist_ok=True)
    result_dir_facenet = resolve_path(result_dir, 'FaceNet')
    result_dir_arcface = resolve_path(result_dir, 'Arcface')
    os.makedirs(result_dir_facenet, exist_ok=True)
    os.makedirs(result_dir_arcface, exist_ok=True)

    # list identity
    identity_count = 0
    identity_facenet_paths = glob(resolve_path(dataset_facenet_path, '*'))
    random.shuffle(identity_facenet_paths)

    for identity_facenet_path in identity_facenet_paths:
        # find the identity which satisfy the image number straints
        if identity_count == 300:
            break
        identity_name = remove_path_prefix(identity_facenet_path)
        image_facenet_paths = glob(resolve_path(identity_facenet_path, '*'))
        image_arcface_paths = glob(resolve_path(dataset_arcface_path, identity_name, '*'))
        # print(f'Facenet: {len(image_facenet_paths)}, Arcface: {len(image_arcface_paths)}')
        if len(image_facenet_paths) < 2 or len(image_facenet_paths) != len(image_arcface_paths):
            continue
        identity_list.append(identity_name)

        # list image
        for i, image_path_facenet in enumerate(image_facenet_paths):
            if i == 2:
                break
            image_name = remove_path_prefix(image_path_facenet)
            # create a folder for image to be saved and copy them
            # FaceNet
            identity_dir_facenet = resolve_path(result_dir_facenet, identity_name)
            os.makedirs(identity_dir_facenet, exist_ok=True)
            shutil.copy(image_path_facenet, identity_dir_facenet)
            image_path_list_facenet.append(image_path_facenet)
            # Arcface
            identity_dir_arcface = resolve_path(result_dir_arcface, identity_name)
            os.makedirs(identity_dir_arcface, exist_ok=True)
            image_path_arcface = resolve_path(dataset_arcface_path, identity_name, image_name)
            shutil.copy(image_path_arcface, identity_dir_arcface)
            image_path_list_arcface.append(image_path_arcface)

        identity_count += 1

    with open(resolve_path(result_dir, 'identity.txt'),'w') as f:
        f.writelines([d+"\n" for d in sorted(identity_list)])
    with open(resolve_path(result_dir_facenet, 'image_path.txt'),'w') as f:
        f.writelines([d+"\n" for d in sorted(image_path_list_facenet)])
    with open(resolve_path(result_dir_arcface, 'image_path.txt'),'w') as f:
        f.writelines([d+"\n" for d in sorted(image_path_list_arcface)])
    
if __name__ == '__main__':
    main()
