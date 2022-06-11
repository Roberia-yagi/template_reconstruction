import sys
sys.path.append('../')
from utils.pytorch_GAN_zoo.hubconf import DCGAN

model = DCGAN(pretrained=True)
print(model.getNetD())