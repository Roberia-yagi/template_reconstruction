import pandas as pd
import os
import sys
import shutil
from tqdm import tqdm
from util import resolve_path


base_dir = '/home/akasaka/nas/dataset/IJB-C_MTCNN160/'
df = pd.read_csv(resolve_path(base_dir, 'datalist.csv'))
print(df)

for _, data in df.iterrows():
    id = str(data['personal_id'])
    if not os.path.isdir(resolve_path(base_dir, 'organized_images/img', id)):
        os.mkdir(resolve_path(base_dir, 'organized_images/img', id))
    list = (data['image_filenames'])
    list = list.strip('][').split(', ')
    for filename in list:
        filename = filename.strip('\'')
        source_file = resolve_path(base_dir, 'images', filename)
        if filename[:filename.rfind('/')] == 'frames':
            continue
        if os.path.isfile(source_file):
            shutil.copy2(source_file, resolve_path(base_dir, 'organized_images/img', id, filename[filename.rfind('/')+1:]))