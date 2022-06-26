import sys
sys.path.append('../')
from utils.pytorch_GAN_zoo.hubconf import PGAN

model = PGAN(pretrained=True, model_name='celebAHQ-256')
print(model.getNetD())