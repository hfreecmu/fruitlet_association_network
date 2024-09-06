import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation 

from util.util import read_json

def cloud_pad(num_pad, feature_vectors, pos_vectors, fruit_ids):
    # to not affect batch norm will select a random fruitlet
    rand_fruit_inds = np.random.randint(low=0, high=feature_vectors.shape[0], size=num_pad)

    features_to_cat = np.copy(feature_vectors[rand_fruit_inds])
    feature_vectors = np.concatenate([feature_vectors, features_to_cat])

    pos_to_cat = np.copy(pos_vectors[rand_fruit_inds])
    pos_vectors = np.concatenate([pos_vectors, pos_to_cat])

    # making this -2 for consistency
    ids_to_cat = np.zeros((num_pad), dtype=fruit_ids.dtype) - 2
    fruit_ids = np.concatenate([fruit_ids, ids_to_cat])

    return feature_vectors, pos_vectors, fruit_ids

def string_to_date(date_str):
    month, day = int(date_str[0:2]), int(date_str[2:])

    if month in [1, 3, 5, 7, 8, 10, 12]:
        month_mult = 31
    elif month in [4, 6, 9, 11]:
        month_mult = 30
    elif month in [2]:
        month_mult = 28
    else:
        raise RuntimeError('Invalid month')
    
    date = month*month_mult + day
    return date

class PhenoDataset(Dataset):
    def __init__(self,
                 anno_root,
                 pheno_type,
                 anno_subdir,
                 is_test,
                 augment,
                 max_comps,
                 day_lim,
                 **kwargs
                 ):
        super().__init__()
        
        annotations_dir = os.path.join(anno_root, pheno_type, anno_subdir)
        if not os.path.exists(annotations_dir):
            raise RuntimeError('Invalid annotations dir: ' + annotations_dir)

        self.annotations_dir = annotations_dir
        self.max_comps = max_comps
        self.day_lim = day_lim
        self.augment = augment
        self.is_test = is_test

        self.file_data = self._get_files()

    def __len__(self):
        return len(self.file_data)
    
    def load_data(self, annotations_path, should_flip, can_drop):
        data = read_json(annotations_path)

        # TODO all augment stuff include random shifting and slight mod pca?
        if self.augment and (np.random.uniform() < self.augment.rotate_pct):
            rotation = Rotation.random().as_matrix()
        else:
            rotation = np.eye(3)

        if self.augment:
            width_scale = np.random.uniform(0.75, 1.25)
            height_scale = np.random.uniform(0.75, 1.25)
        else:
            width_scale = 1.0
            height_scale = 1.0

        if should_flip:
            flip_axis = np.random.randint(0, 3)
            one_array = np.array([1.0, 1.0, 1.0])
            one_array[flip_axis] = -1
            rotation = np.diag(one_array) @ rotation
        else:
            pass

        label_list = list(data.keys())
        if self.augment and can_drop and len(label_list) > 10:
            rand_drop_ind = np.random.randint(0, len(label_list))
            label_to_del = label_list[rand_drop_ind]
            del data[label_to_del]

        feature_vectors = []
        pos_vectors = []
        labels = []
        for label in data.keys():
            length = data[label]['length'] * height_scale
            width = data[label]['width'] * width_scale
            princ_0 = data[label]['princ_0'] @ rotation.T
            princ_1 = data[label]['prince_1'] @ rotation.T
            loc = data[label]['loc'] @ rotation.T

            feature_vec = np.concatenate(([length, width], princ_0, princ_1))
            feature_vectors.append(feature_vec)
            pos_vectors.append(loc)
            labels.append(int(label))
        
        feature_vectors = np.array(feature_vectors)
        pos_vectors = np.array(pos_vectors)
        labels = np.array(labels)

        pos_vectors = pos_vectors - pos_vectors.mean(axis=0)        

        return feature_vectors, pos_vectors, labels

    def _get_data(self, entry, should_flip, can_drop):
        annotations_path = entry
        feature_vectors, pos_vectors, fruit_ids = self.load_data(annotations_path, should_flip, can_drop)
        
        is_pad = np.zeros((self.max_comps), dtype=bool)
        num_pad = self.max_comps - feature_vectors.shape[0]

        if num_pad < 0:
            raise RuntimeError('Too many components')
        
        if num_pad > 0:
            feature_vectors, pos_vectors, fruit_ids = cloud_pad(num_pad, feature_vectors, pos_vectors, fruit_ids)
            is_pad[-num_pad:] = True

        feature_vectors = feature_vectors.astype(np.float32)
        pos_vectors = pos_vectors.astype(np.float32)

        file_key = os.path.basename(annotations_path).replace('.json', '')

        return file_key, feature_vectors, pos_vectors, is_pad, fruit_ids, annotations_path

    def __getitem__(self, index):
        entry_0, entry_1 = self.file_data[index]

        if self.augment:
            should_flip = np.random.uniform() < 0.5
        else:
            should_flip = False

        # TODO might want both dropping if enough?
        can_drop_0 = False
        can_drop_1 = False
        if self.augment:
            if np.random.uniform() < 0.5:
                entry_0, entry_1 = entry_1, entry_0
            
            if np.random.uniform() < self.augment.drop_pct:
                can_drop_0 = True

        file_key_0, feature_vectors_0, pos_vectors_0, is_pad_0, fruit_ids_0, anno_path_0 = self._get_data(entry_0, should_flip, can_drop_0)
        file_key_1, feature_vectors_1, pos_vectors_1, is_pad_1, fruit_ids_1, anno_path_1 = self._get_data(entry_1, should_flip, can_drop_1)

        matches_gt = np.zeros((self.max_comps, self.max_comps)).astype(np.float32)
        masks_gt = np.ones((self.max_comps, self.max_comps)).astype(np.float32)

        masks_gt[is_pad_0, :] = 0.0
        masks_gt[:, is_pad_1] = 0.0

        for ind_0 in range(self.max_comps):
            fruit_id_0 = fruit_ids_0[ind_0]
            if fruit_id_0 < 0:
                continue

            if not fruit_id_0 in fruit_ids_1:
                continue

            ind_1 = np.where(fruit_ids_1 == fruit_id_0)[0][0]
            matches_gt[ind_0, ind_1] = 1.0

        return file_key_0, feature_vectors_0, pos_vectors_0, is_pad_0, fruit_ids_0, \
               file_key_1, feature_vectors_1, pos_vectors_1, is_pad_1, fruit_ids_1, \
               matches_gt, masks_gt

    def _get_files(self):
        cluster_dict = {}
        for dirname in os.listdir(self.annotations_dir):
            subdir = os.path.join(self.annotations_dir, dirname)
            if not os.path.isdir(subdir):
                continue

            cluster_dict[dirname] = []

            for filename in os.listdir(subdir):
                if not filename.endswith('_info.json'):
                    continue

                anno_path = os.path.join(subdir, filename)
                cluster_dict[dirname].append(anno_path)

        file_data = []
        for key in cluster_dict:
            entries = cluster_dict[key]
            num_entries = len(entries)

            for ind_0 in range(num_entries):
                entry_0 = entries[ind_0]
                date_0 = os.path.basename(entry_0).split('_')[1]
                date_0 = string_to_date(date_0)

                for ind_1 in range(ind_0 + 1, num_entries):
                    entry_1 = entries[ind_1]
                    date_1 = os.path.basename(entry_1).split('_')[1]
                    date_1 = string_to_date(date_1)

                    if np.abs(date_1 - date_0) > self.day_lim:
                        continue

                    file_data.append([entry_0, entry_1])

        return file_data

                