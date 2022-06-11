import sys
sys.path.append("../")

from PIL import Image
from facenet_pytorch import MTCNN

from torchvision.utils import save_image
import torchvision.transforms as transforms

from util import resolve_path

dataset_in_dir = resolve_path("~/nas/dataset/target_images/Sato")
dataset_out_dir = resolve_path("~/nas/dataset/target_images/Sato_MTCNN160")

image_crop_size = 100
image_size = 160

# Create MTCNN
mtcnn = MTCNN(image_size=image_size)

# Create transform for fallback
transform = transforms.Compose([
    transforms.CenterCrop(image_crop_size),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

# change the range to variable
for i in range(1, 9 + 1):
    filename = f"{i:03d}.jpg"
    in_path = resolve_path(dataset_in_dir, filename)
    out_path = resolve_path(dataset_out_dir, filename)

    img = Image.open(in_path)
    img_cropped = mtcnn(img, save_path=out_path)

    if img_cropped == None:
        print(filename, file=sys.stderr)
        # save_image(transform(img), out_path)

    if i % 100 == 0:
        print(f"Done {i}/{18}")