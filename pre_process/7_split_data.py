import os
import numpy as np
import shutil

def get_identifier(file_key):
    identifier = '_'.join(file_key.split('_')[0:2])

    return identifier

def split_data(annotations_dir, train_pct, val_pct, train_dir, val_dir, test_dir):
    file_dict = {}
    for filename in os.listdir(annotations_dir):
        if not filename.endswith('.json'):
            continue

        file_key = get_identifier(filename)

        if not file_key in file_dict:
            file_dict[file_key] = []

        file_dict[file_key].append(os.path.join(annotations_dir, filename))

    file_keys = list(file_dict.keys())
    num_keys = len(file_keys)

    num_train = int(num_keys*train_pct)
    num_val = int(num_keys*val_pct)
    
    shuffle_inds = np.random.choice(num_keys, size=num_keys, replace=False)

    for ind in shuffle_inds:
        if ind < num_train:
            output_dir = train_dir
        elif ind < num_train + num_val:
            output_dir = val_dir
        else:
            output_dir = test_dir

        file_key = file_keys[shuffle_inds[ind]]
        
        for src in file_dict[file_key]:
            dest = os.path.join(output_dir, os.path.basename(src))

            shutil.copyfile(src, dest)

train_pct = 0.60
val_pct = 0.21
annotations_dir = 'labelling/id_annotations_filtered'
train_dir = 'datasets/train'
val_dir = 'datasets/val'
test_dir = 'datasets/test'

split_data(annotations_dir, train_pct, val_pct, train_dir, val_dir, test_dir)
