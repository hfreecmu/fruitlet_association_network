import os
import numpy as np
from torch.utils.data import Dataset

from util.util import get_identifier, read_json, write_json
from data.dataset import get_assoc_fruitlets

class ICPAssociationDataset(Dataset):
    def __init__(self,
                 anno_root,
                 anno_subdir,
                 images_dir,
                 cache,
                 min_fruitlets_per_im=3,
                 min_fruitlet_matches=3,
                 **kwargs
                 ):
        
        super().__init__()

        annotations_dir = os.path.join(anno_root, anno_subdir)
        if not os.path.exists(annotations_dir):
            raise RuntimeError('Invalid annotations dir: ' + annotations_dir)
        
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir

        self.file_data = self._get_files(min_fruitlets_per_im, min_fruitlet_matches, cache)

    def __len__(self):
        return len(self.file_data)
    
    def load_data(self, annotations_path):
        annotations = read_json(annotations_path)['annotations']

        clouds = []
        cloud_inds = []
        fruitlet_ids = []

        for det in annotations:
            if det['fruitlet_id'] < 0:
                continue

            cloud_points = np.array(det['cloud_points'])
            clouds.append(cloud_points)
            cloud_inds.append(np.zeros((cloud_points.shape[0])) + det['fruitlet_id'])
            fruitlet_ids.append(det['fruitlet_id'])

        clouds = np.concatenate(clouds)
        cloud_inds = np.concatenate(cloud_inds)
        fruitlet_ids = np.array(fruitlet_ids)

        return clouds, cloud_inds, fruitlet_ids
    
    def _get_data(self, entry):
        file_key, annotations_path, _ = entry
        clouds, cloud_inds, fruitlet_ids = self.load_data(annotations_path)

        return file_key, clouds, cloud_inds, fruitlet_ids
    
    def __getitem__(self, index):
        entry_0, entry_1 = self.file_data[index]

        file_key_0, clouds_0, cloud_inds_0, fruitlet_ids_0 = self._get_data(entry_0)
        file_key_1, clouds_1, cloud_inds_1, fruitlet_ids_1 = self._get_data(entry_1)


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
               matches_gt

    # TODO duplicate logic remove
    def _get_files(self, 
                   min_fruitlets_per_im, 
                   min_fruitlet_matches,
                   cache,
                   ):
        if cache:
            cache_path = os.path.join(self.annotations_dir, 'cache.txt')
            if os.path.exists(cache_path):
                print('Using Dataset Cache')
                file_data = read_json(cache_path)
                return file_data
            
        cluster_dict = {}
        for filename in os.listdir(self.annotations_dir):
            if not filename.endswith('.json'):
                continue

            if 'right' in filename:
                continue

            file_key = filename.split('.json')[0]

            annotations_path = os.path.join(self.annotations_dir, filename)
            image_path = os.path.join(self.images_dir, file_key + '.png')

            if not os.path.exists(image_path):
                raise RuntimeError('Image path dne: ' + image_path)
            
            identifier = get_identifier(file_key)

            if not identifier in cluster_dict:
                cluster_dict[identifier] = []

            entry = (file_key, annotations_path, image_path)
            cluster_dict[identifier].append(entry)

        file_data = []
        for identifier in cluster_dict:
            entries = cluster_dict[identifier]
            num_entries = len(entries)

            for ind_0 in range(num_entries):
                entry_0 = entries[ind_0]

                # read annotations in entry_0
                # filter if less than three fruiltets
                anno_0 = read_json(entry_0[1])['annotations']
                fruitlet_ids_0 = get_assoc_fruitlets(anno_0)
                if len(fruitlet_ids_0) < min_fruitlets_per_im:
                    continue

                for ind_1 in range(ind_0 + 1, num_entries):
                    entry_1 = entries[ind_1]

                    # read annotations in entry_1
                    # filter if less than three fruitlets
                    anno_1 = read_json(entry_1[1])['annotations']
                    fruitlet_ids_1 = get_assoc_fruitlets(anno_1)
                    if len(fruitlet_ids_1) < min_fruitlets_per_im:
                        continue

                    # if entries don't have at least 2 matching fruitlets continue
                    num_match = 0
                    for fruitlet_id in fruitlet_ids_0:
                        if fruitlet_id in fruitlet_ids_1:
                            num_match += 1
                    if num_match < min_fruitlet_matches:
                        continue

                    file_data.append([entry_0, entry_1])
        
        if cache:
            write_json(cache_path, file_data)

        return file_data