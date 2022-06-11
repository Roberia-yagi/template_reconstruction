import pickle
import torchvision.transforms as transforms
from PIL import Image

with open('/home/akasaka/nas/results/common/step3/2022_06_06_20_30/20/best_image.pkl', 'rb') as f:
    tensor1 = pickle.load(f)
with open('/home/akasaka/nas/results/common/step3/2022_06_06_20_30/30/best_image.pkl', 'rb') as f:
    tensor2 = pickle.load(f)

print(tensor1)
print(tensor2)

image1 = Image.open('/home/akasaka/nas/results/common/step3/2022_06_06_20_30/20/best_image_cossim_0.8893260955810547.png')
image2 = Image.open('/home/akasaka/nas/results/common/step3/2022_06_06_20_30/30/best_image_cossim_0.917019248008728.png')

transform = transforms.Compose([
    transforms.ToTensor()
])
  
# transform = transforms.PILToTensor()
# Convert the PIL image to Torch tensor
img_tensor1 = transform(image1)
img_tensor2 = transform(image2)
  
# print the converted Torch tensor
print(img_tensor1)
print(img_tensor2)