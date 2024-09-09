import os
from torch.utils.data import Dataset

from util.util import get_identifier, CROSS_DAY_DICT, read_json

class DatasetInterface(Dataset):
    def __init__(self,
                 anno_root,
                 anno_subdir,
                 images_dir,
                 cross_day,
                 is_test,
                 ):
        super().__init__()

        annotations_dir = os.path.join(anno_root, anno_subdir)
        if not os.path.exists(annotations_dir):
            raise RuntimeError('Invalid annotations dir: ' + annotations_dir)
        
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir
        self.cross_day = cross_day
        self.is_test = is_test

        self.file_data = self._get_files()

    def _get_files(self, 
                   ):

        if self.cross_day:
            #TODO pass this in on cfg
            backwards_mapping_path = os.path.join(os.path.dirname(self.images_dir), 'backward_mappings.json')
            backwards_map = read_json(backwards_mapping_path)


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

                if self.cross_day:
                    map_key = os.path.basename(entry_0[2])
                    orig_path = backwards_map[map_key]
                    date_str_0 = '-'.join(os.path.basename(os.path.dirname(os.path.dirname(orig_path))).split('_')[1].split('-')[0:3])
                    _, mm_0, dd_0 = date_str_0.split('-')
                    mm_0, dd_0 = int(mm_0), int(dd_0)

                    # if (not date_str_0 in CROSS_DAY_DICT):
                    #     continue 

                for ind_1 in range(ind_0 + 1, num_entries):
                    entry_1 = entries[ind_1]

                    if self.cross_day:
                        map_key = os.path.basename(entry_1[2])
                        orig_path = backwards_map[map_key]
                        date_str_1 = '-'.join(os.path.basename(os.path.dirname(os.path.dirname(orig_path))).split('_')[1].split('-')[0:3])
                        # if (not date_str_1 in CROSS_DAY_DICT):
                        #     continue

                        if date_str_0 == date_str_1:
                            continue

                        _, mm_1, dd_1 = date_str_1.split('-')
                        mm_1, dd_1 = int(mm_1), int(dd_1)
                        
                        swap_day = False
                        if mm_1 > mm_0:
                            pass
                        elif mm_1 < mm_0:
                            swap_day = True
                        elif dd_1 > dd_0:
                            pass
                        elif dd_1 < dd_0:
                            swap_day = True
                        else:
                            raise RuntimeError('should not happen')

                    if self.cross_day and swap_day:
                        file_data.append([entry_1, entry_0])
                    else:
                        file_data.append([entry_0, entry_1])

        return file_data
