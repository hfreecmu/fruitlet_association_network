import os
from torch.utils.data import Dataset

from util.util import get_identifier

class DatasetInterface(Dataset):
    def __init__(self,
                 anno_root,
                 anno_subdir,
                 images_dir,
                 is_test,
                 ):
        super().__init__()

        annotations_dir = os.path.join(anno_root, anno_subdir)
        if not os.path.exists(annotations_dir):
            raise RuntimeError('Invalid annotations dir: ' + annotations_dir)
        
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir
        self.is_test = is_test

        self.file_data = self._get_files()

    def _get_files(self, 
                   ):

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

                for ind_1 in range(ind_0 + 1, num_entries):
                    entry_1 = entries[ind_1]

                    file_data.append([entry_0, entry_1])

        return file_data
