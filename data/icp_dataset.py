import os
import numpy as np

from util.util import read_json
from data.interface import DatasetInterface

class ICPAssociationDataset(DatasetInterface):
    def __init__(self,
                 anno_root,
                 anno_subdir,
                 images_dir,
                 cross_day,
                 is_test,
                 **kwargs
                 ):
        
        super().__init__(anno_root, anno_subdir, images_dir,
                         cross_day, is_test)

    def __len__(self):
        return len(self.file_data)
    
    def load_data(self, annotations_path):
        full_anno = read_json(annotations_path)
        annotations = full_anno['annotations']
        if 'mean_vals' in full_anno:
            mean_vals = np.array(full_anno['mean_vals'])
        else:
            mean_vals = np.array([0, 0, 0])

        clouds = []
        cloud_inds = []
        fruitlet_ids = []

        for det in annotations:
            if det['fruitlet_id'] < 0:
                continue

            if len(det['cloud_points']) == 0:
                cloud_points = np.zeros((0, 3))
            else:
                cloud_points = np.array(det['cloud_points'])
                
            clouds.append(cloud_points)
            cloud_inds.append(np.zeros((cloud_points.shape[0])) + det['fruitlet_id'])
            fruitlet_ids.append(det['fruitlet_id'])

        clouds = np.concatenate(clouds)
        cloud_inds = np.concatenate(cloud_inds)
        fruitlet_ids = np.array(fruitlet_ids)

        return clouds, cloud_inds, fruitlet_ids, mean_vals
    
    def _get_data(self, entry):
        file_key, annotations_path, _ = entry
        clouds, cloud_inds, fruitlet_ids, mean_vals = self.load_data(annotations_path)

        return file_key, clouds, cloud_inds, fruitlet_ids, mean_vals
    
    def __getitem__(self, index):
        entry_0, entry_1 = self.file_data[index]

        file_key_0, clouds_0, cloud_inds_0, fruitlet_ids_0, mean_vals_0 = self._get_data(entry_0)
        file_key_1, clouds_1, cloud_inds_1, fruitlet_ids_1, mean_vals_1 = self._get_data(entry_1)

        num_0 = fruitlet_ids_0.shape[0]
        num_1 = fruitlet_ids_1.shape[0]

        matches_gt = np.zeros((num_0, num_1)).astype(np.float32)

        for ind_0 in range(num_0):
            fruitlet_id_0 = fruitlet_ids_0[ind_0]
            if fruitlet_id_0 < 0:
                continue

            if not fruitlet_id_0 in fruitlet_ids_1:
                continue

            ind_1 = np.where(fruitlet_ids_1 == fruitlet_id_0)[0][0]
            matches_gt[ind_0, ind_1] = 1.0

        return file_key_0, clouds_0, cloud_inds_0, \
               fruitlet_ids_0, \
               file_key_1, clouds_1, cloud_inds_1, \
               fruitlet_ids_1, \
               matches_gt, mean_vals_0, mean_vals_1
    