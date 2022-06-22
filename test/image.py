from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

folder_path = '/home/akasaka/projects/akasaka/dataset/test'
img_path = '/home/akasaka/projects/akasaka/dataset/test/001/001.jpg'
# print(f'cv: {cv2.imread(path)}')
# print(f'PIL: {np.array(Image.open(path))}')
transform = transforms.ToTensor()
dataset = ImageFolder(folder_path)
print(f'cv: {transform(cv2.imread(img_path))}')
print(f'PIL: {transform(Image.open(img_path))}')
img = transform(Image.open(img_path))
img[0], img[2] = img.clone()[2], img.clone()[0]
print(f'PIL: {img}')
for x in dataset:
    print(f'ImageFolder:{transform(x[0])}')

