import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation

# TODO do I want this?
# for point cloud
CLOUD_MEANS = [0.0, 0.0, 0.0]
CLOUD_STDS = [1.0, 1.0, 1.0]

def cloud_pad(num_pad, cloud_boxes, fruit_ids):
    # to not affect batch norm will select a random fruitlet
    rand_fruit_inds = np.random.randint(low=0, high=cloud_boxes.shape[0], size=num_pad)
    clouds_to_cat = np.copy(cloud_boxes[rand_fruit_inds])
    cloud_boxes = np.concatenate([cloud_boxes, clouds_to_cat])

    # making this -2 for consistency
    ids_to_cat = np.zeros((num_pad), dtype=fruit_ids.dtype) - 2
    fruit_ids = np.concatenate([fruit_ids, ids_to_cat])

    return cloud_boxes, fruit_ids

class PhenoDataset(Dataset):
    def __init__(self,
                 anno_root,
                 pheno_type,
                 anno_subdir,
                 is_test,
                 augment,
                 max_comps,
                 **kwargs
                 ):
        super().__init__()
        
        annotations_dir = os.path.join(anno_root, pheno_type, anno_subdir)
        if not os.path.exists(annotations_dir):
            raise RuntimeError('Invalid annotations dir: ' + annotations_dir)

        self.annotations_dir = annotations_dir
        self.max_comps = max_comps
        self.augment = augment
        self.is_test = is_test

        self.file_data = self._get_files()

    def __len__(self):
        return len(self.file_data)
    
    def load_data(self, annotations_path, should_flip, can_drop):
        data = np.load(annotations_path)

        full_points = data[:, 0:3]
        labels = data[:, 3].astype(int)

        if self.augment and (np.random.uniform() < self.augment.rotate_pct):
            rotation = Rotation.random().as_matrix()
        else:
            rotation = np.eye(3)

        if should_flip:
            flip_axis = np.random.randint(0, 3)
            one_array = np.array([1.0, 1.0, 1.0])
            one_array[flip_axis] = -1
            rotation = np.diag(one_array) @ rotation
        else:
            pass

        unique_labels = np.unique(labels)
        if self.augment and can_drop and unique_labels.shape[0] > 3:
            rand_drop_ind = np.random.randint(0, unique_labels.shape[0])
            unique_labels = np.concatenate([unique_labels[0:rand_drop_ind], unique_labels[rand_drop_ind + 1:]])

        full_points = (rotation @ full_points.T).T
        cloud_boxes = []
        full_cloud_points = []
        for label in unique_labels:
            cloud_points = full_points[labels == label]

            # rand drop 10% of cloud points
            # will not have effect unless min / max is dropped
            if self.augment and (np.random.uniform() < self.augment.drop_cloud_pct):
                clouds_to_keep = int(np.round(cloud_points.shape[0]*0.9))
                rand_inds = np.random.permutation(cloud_points.shape[0])
                cloud_points = cloud_points[rand_inds[0:clouds_to_keep]] 

            # get mins and maxes
            mins = cloud_points.min(axis=0)
            maxs = cloud_points.max(axis=0)
            box_3d = np.array([mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]])
            
            cloud_boxes.append(box_3d)
            full_cloud_points.append(cloud_points)

        cloud_boxes = np.array(cloud_boxes)
        full_cloud_points = np.concatenate(full_cloud_points)

        mean_3d = full_cloud_points.mean(axis=0)
        cloud_boxes[:, 0:2] -= mean_3d[0]
        cloud_boxes[:, 2:4] -= mean_3d[1]
        cloud_boxes[:, 4:6] -= mean_3d[2]

        return cloud_boxes, unique_labels



    def _get_data(self, entry, should_flip, can_drop):
        annotations_path = entry
        cloud_boxes, fruit_ids = self.load_data(annotations_path, should_flip, can_drop)
        
        is_pad = np.zeros((self.max_comps), dtype=bool)
        num_pad = self.max_comps - cloud_boxes.shape[0]

        if num_pad < 0:
            raise RuntimeError('Too many components')
        
        if num_pad > 0:
            cloud_boxes, fruit_ids = cloud_pad(num_pad, cloud_boxes, fruit_ids)
            is_pad[-num_pad:] = True

        if self.augment:
            randperm = torch.randperm(self.max_comps)
            cloud_boxes = cloud_boxes[randperm]
            fruit_ids = fruit_ids[randperm]
            is_pad = is_pad[randperm]

        cloud_boxes = cloud_boxes.astype(np.float32)
        file_key = os.path.basename(annotations_path).replace('.npy', '')

        return file_key, cloud_boxes, is_pad, fruit_ids, annotations_path

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

        file_key_0, cloud_boxes_0, is_pad_0, fruit_ids_0, anno_path_0 = self._get_data(entry_0, should_flip, can_drop_0)
        file_key_1, cloud_boxes_1, is_pad_1, fruit_ids_1, anno_path_1 = self._get_data(entry_1, should_flip, can_drop_1)

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

        return file_key_0, cloud_boxes_0, is_pad_0, fruit_ids_0, \
               file_key_1, cloud_boxes_1, is_pad_1, fruit_ids_1, \
               matches_gt, masks_gt

    def _get_files(self):
        cluster_dict = {}
        for dirname in os.listdir(self.annotations_dir):
            subdir = os.path.join(self.annotations_dir, dirname)
            if not os.path.isdir(subdir):
                continue

            cluster_dict[dirname] = []

            for filename in os.listdir(subdir):
                if not filename.endswith('.npy'):
                    continue

                npy_path = os.path.join(subdir, filename)
                cluster_dict[dirname].append(npy_path)

        file_data = []
        for key in cluster_dict:
            entries = cluster_dict[key]
            num_entries = len(entries)

            for ind_0 in range(num_entries):
                entry_0 = entries[ind_0]

                for ind_1 in range(ind_0 + 1, num_entries):
                    entry_1 = entries[ind_1]

                    file_data.append([entry_0, entry_1])

        return file_data

                