import os
from torch.utils.data import Dataset

from util.util import get_identifier, read_json, write_json

class DatasetInterface(Dataset):
    def __init__(self,
                 anno_root,
                 anno_subdir,
                 images_dir,
                 cache,
                 min_fruitlets_per_im,
                 min_fruitlet_matches,
                 ):
        super().__init__()

        assert min_fruitlet_matches <= min_fruitlets_per_im

        annotations_dir = os.path.join(anno_root, anno_subdir)
        if not os.path.exists(annotations_dir):
            raise RuntimeError('Invalid annotations dir: ' + annotations_dir)
        
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir

        self.file_data = self._get_files(min_fruitlets_per_im, min_fruitlet_matches, cache)

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
    
def get_assoc_fruitlets(anno):
    fruitlet_ids = set()
    for det in anno:
        if det['fruitlet_id'] < 0:
            continue

        fruitlet_ids.add(det['fruitlet_id'])

    return fruitlet_ids