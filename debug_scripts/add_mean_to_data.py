import os
import json

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def write_json(path, data, pretty=False):
    with open(path, 'w') as f:
        if not pretty:
            json.dump(data, f)
        else:
            json.dump(data, f, indent=4)

anno_dir = 'labelling/id_annotations_filtered'

data_dir = 'datasets'
data_types = ['train', 'test', 'val', 'mini']

for dirname in data_types:
    subdir = os.path.join(data_dir, dirname)

    for filename in os.listdir(subdir):
        data_path = os.path.join(subdir, filename)
        anno_path = os.path.join(anno_dir, filename)

        data = read_json(data_path)
        anno = read_json(anno_path)
        
        if 'mean_vals' in anno:
            data['mean_vals'] = anno['mean_vals']

            write_json(data_path, data)

            print('corrected ' + filename)
