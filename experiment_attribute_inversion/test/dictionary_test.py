import sys
sys.path.append('../')
import json
import itertools
from typing import Any

def load_json(path: str) -> Any:
    with open(path) as json_obj:
        return json.load(json_obj)

attribute_group_list = load_json('../attributes.json')
# for attribute_group in attribute_group_list:
    # print(attribute_group)
    # print(attribute_group_list[attribute_group])

attribute_list = []

for attribute_group in attribute_group_list:
    attribute_list.append(attribute_group_list[attribute_group])

# print(attribute_list)
attribute1 = attribute_list[0]
attribute2 = attribute_list[1]
attribute3 = attribute_list[2]
print(attribute1)
print(list(itertools.product(*attribute_list)))