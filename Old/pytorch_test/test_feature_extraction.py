import torch
import torchvision
from arcface_pytorch.models import * 
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

# Example for arcface_pytorch
# Feature extraction with resnet
# model = resnet50()
model = torchvision.models.resnet152()
train_name, eval_name = get_graph_node_names(model)
# print('\n'.join('{}: {}'.format(*k) for k in enumerate(train_name)))

# create dict[layer_name] = layer_name for feature_extractor
layer_dict = dict()
for layer_name in train_name:
    layer_dict[layer_name] = layer_name

model = create_feature_extractor(
    model, layer_dict)
out = model(torch.rand(1, 3, 112, 112))
print('\n'.join('{}: {}'.format(k, v.shape) for k, v in out.items() if type(v) is not int))

# Example for torchvision
# Feature extraction with resnet
# model = torchvision.models.resnet50()
# train_name, eval_name = get_graph_node_names(torchvision.models.resnet50())
# # print('\n'.join('{}: {}'.format(*k) for k in enumerate(train_name)))

# # create dict[layer_name] = layer_name for feature_extractor
# layer_dict = dict()
# for layer_name in train_name:
#     layer_dict[layer_name] = layer_name

# model = create_feature_extractor(
#     model, layer_dict)
# out = model(torch.rand(1, 3, 112, 112))
# print('\n'.join('{}: {}'.format(k, v.shape) for k, v in out.items())