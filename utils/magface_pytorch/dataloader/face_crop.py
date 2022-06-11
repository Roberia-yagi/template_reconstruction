import sys
sys.path.append("../")
sys.path.append("../../")

from PIL import Image
from facenet_pytorch import MTCNN
import glob

from torchvision.utils import save_image
import torchvision.transforms as transforms

from util import resolve_path

dataset_in_dir = resolve_path("/home/akasaka/nas/dataset/faces_emore/outputs")
dataset_out_dir = resolve_path("/home/akasaka/nas/dataset/faces_emore/MTCNN")

image_crop_size = 112
image_size = 160

# Create MTCNN
mtcnn = MTCNN(image_size=image_size)

# Create transform for fallback
transform = transforms.Compose([
    transforms.CenterCrop(image_crop_size),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

folder_list = glob.glob(dataset_in_dir + '/*')

for i, folder in enumerate(folder_list):
    file_list = glob.glob(folder + '/*')
    for filename in file_list:
        out_path = resolve_path(dataset_out_dir, filename[7], filename[8])

        img = Image.open(filename)
        img_cropped = mtcnn(img, save_path=out_path)

        if img_cropped == None:
            print(filename, file=sys.stderr)
            # save_image(transform(img), out_path)

    if i % 100 == 0:
        print(f"Done {i}")