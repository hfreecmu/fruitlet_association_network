import os
import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
import torchvision.transforms.functional

from util.util import get_identifier, read_json, write_json

# torch uses these for resnet and vit
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# timm vit uses this
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

# for point cloud
#CLOUD_MINS = np.array([-0.04106932, -0.05654648, -0.07063983])
#CLOUD_MAXS = np.array([0.05363945, 0.04220505, 0.27370004])
#CLOUD_MEANS = [0.43363797, 0.57261368, 0.20514566]
#CLOUD_STDS = [0.14553247, 0.08723398, 0.03884694]
#I am modifying to include the largest abs values i
CLOUD_MINS = np.array([-0.2737000429336701, -0.2737000429336701, -0.2737000429336701])
CLOUD_MAXS = np.array([0.2737000429336701, 0.2737000429336701, 0.2737000429336701])
CLOUD_MEANS = [0.5, 0.5, 0.5]
CLOUD_STDS = [0.02517948, 0.01573716, 0.02443657]

def get_image_transform(mean, std):
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ConvertImageDtype(torch.float32),
        torchvision.transforms.Normalize(mean=mean, std=std),
    ])

    return image_transform

def get_fruitlet_transform(fruitlet_image_size):
    fruitlet_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((fruitlet_image_size, fruitlet_image_size)),
    ])

    return fruitlet_transform

def get_cloud_transform(mean, std):
    cloud_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=mean, std=std),
    ])

    return cloud_transform

def fruitlet_pad(num_pad, fruitlet_images, cloud_images, fruitlet_ids):
    img_height, img_width = fruitlet_images.shape[2:]
    
    images_to_cat = torch.zeros(*[num_pad, 3, img_height, img_width], dtype=fruitlet_images.dtype)
    fruitlet_images = torch.vstack([fruitlet_images, images_to_cat])

    clouds_to_cat = torch.zeros(*[num_pad, 4, img_height, img_width], dtype=cloud_images.dtype)
    cloud_images = torch.vstack([cloud_images, clouds_to_cat])

    # making this -2 to avoid confusion
    ids_to_cat = np.zeros((num_pad), dtype=fruitlet_ids.dtype) - 2
    fruitlet_ids = np.concatenate([fruitlet_ids, ids_to_cat])

    return fruitlet_images, cloud_images, fruitlet_ids

class AssociationDataset(Dataset):
    def __init__(self,
                 anno_root,
                 anno_subdir, 
                 images_dir,
                 image_size,
                 encoder_type,
                 augment,
                 cache,
                 max_fruitlets=6,
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
        self.max_fruitlets = max_fruitlets
        self.augment = augment

        if encoder_type == 'vit':
            mean = OPENAI_DATASET_MEAN
            std = OPENAI_DATASET_STD
        elif 'resnet' in encoder_type:
            mean = IMAGENET_MEAN
            std = IMAGENET_STD
        else:
            raise RuntimeError('Invalid encoder type: ' + encoder_type)

        self.image_transform = get_image_transform(mean, std)
        self.fruitlet_transform = get_fruitlet_transform(image_size)
        self.cloud_transform = get_cloud_transform(CLOUD_MEANS, CLOUD_STDS)

        self.random_brightness = torchvision.transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1)

        self.file_data = self._get_files(min_fruitlets_per_im, min_fruitlet_matches, cache)

    def __len__(self):
        return len(self.file_data)
    
    def load_data(self, annotations_path, image_path, should_flip):
        # read image and annotations
        image = torchvision.io.read_image(image_path)
        annotations = read_json(annotations_path)['annotations']

        # transform image by scaling to 0->1 and normalizing
        image = self.image_transform(image)

        # get fruitlet data
        fruitlet_ims = []
        cloud_ims = []
        fruitlet_ids = []

        # if augmnet always rotate?
        if self.augment:
            rotation = Rotation.random().as_matrix()
        else:
            rotation = np.eye(3)

        for det in annotations:
            if det['fruitlet_id'] < 0:
                continue

            # get seg_inds and cloud_points
            # size should be equal
            seg_inds = np.array(det['seg_inds'])
            cloud_points = np.array(det['cloud_points'])

            # if should flip then cloud points are flipped along the x axis
            if should_flip:
                cloud_points[:, 0] = -cloud_points[:, 0]

            if seg_inds.shape[0] != cloud_points.shape[0]:
                raise RuntimeError('seg inds and cloud points should have same size' \
                                   + annotations_path)
            
            #rotate cloud points after flipping
            cloud_points = (rotation @ cloud_points.T).T

            uniformed_cloud_points = (cloud_points - CLOUD_MINS) / (CLOUD_MAXS - CLOUD_MINS)
            uniformed_cloud_points = torch.as_tensor(uniformed_cloud_points, dtype=image.dtype)

            # larger is further away
            # so nothing there should be inifite - max value
            cloud_image_sub = torch.zeros_like(image) + np.max(CLOUD_MAXS)
            cloud_image_sub[:, seg_inds[:, 0], seg_inds[:, 1]] = uniformed_cloud_points.T
            cloud_image_sub = self.cloud_transform(cloud_image_sub)

            cloud_image = torch.zeros(4, image.shape[1], image.shape[2], dtype=image.dtype)
            cloud_image[0:3] = cloud_image_sub
            cloud_image[3, seg_inds[:, 0], seg_inds[:, 1]] = 1.0

            # TODO add seg on fruitlet image?
            x0 = det["x0"]
            y0 = det["y0"]
            x1 = det["x1"]
            y1 = det["y1"]
            fruitlet_id = det["fruitlet_id"]

            round_x0 = int(np.round(x0))
            round_y0 = int(np.round(y0))
            round_x1 = int(np.round(x1))
            round_y1 = int(np.round(y1))

            fruitlet_im = image[:, round_y0:round_y1, round_x0:round_x1]
            cloud_im = cloud_image[:, round_y0:round_y1, round_x0:round_x1]

            # random crop
            if self.augment:
                _, fruitlet_height, fruitlet_width = fruitlet_im.shape
                new_height = int(np.random.uniform(low=0.9, high=1.0)*fruitlet_height)
                new_width = int(np.random.uniform(low=0.9, high=1.0)*fruitlet_width)
                i, j, h, w = torchvision.transforms.RandomCrop.get_params(fruitlet_im, 
                                                                       output_size=(new_height, new_width))
                
                fruitlet_im = torchvision.transforms.functional.crop(fruitlet_im,
                                                                     i, j, h, w)
                
                cloud_im = torchvision.transforms.functional.crop(cloud_im,
                                                                     i, j, h, w)
                
            _, fruitlet_height, fruitlet_width = fruitlet_im.shape

            # pad fruitlet_im
            # (left, right, top, bottom)
            if fruitlet_height > fruitlet_width:
                num_horiz_pad = fruitlet_height - fruitlet_width
                left_pad = num_horiz_pad // 2
                right_pad = num_horiz_pad - left_pad
                padding = (left_pad, right_pad, 0, 0)
            elif fruitlet_width > fruitlet_height:
                num_vert_pad = fruitlet_width - fruitlet_height
                top_pad = num_vert_pad // 2
                bottom_pad = num_vert_pad - top_pad
                padding = (0, 0, top_pad, bottom_pad)
            else:
                padding = (0, 0, 0, 0)

            padded_fruitlet_im = F.pad(fruitlet_im, padding, mode='constant', value=0)
            fruitlet_im = self.fruitlet_transform(padded_fruitlet_im)

            padded_cloud_im = F.pad(cloud_im, padding, mode='constant', value=0)
            cloud_im = self.fruitlet_transform(padded_cloud_im)

            fruitlet_ims.append(fruitlet_im)
            cloud_ims.append(cloud_im)
            fruitlet_ids.append(fruitlet_id)

        fruitlet_ims = torch.stack(fruitlet_ims)
        cloud_ims = torch.stack(cloud_ims)
        fruitlet_ids = np.array(fruitlet_ids)

        # if should flip then flip left and right cloud images
        if should_flip:
            fruitlet_ims = torch.flip(fruitlet_ims, dims=(-1,))
            cloud_ims = torch.flip(cloud_ims, dims=(-1,))

        return fruitlet_ims, cloud_ims, fruitlet_ids

    def _get_data(self, entry, should_flip):
        file_key, annotations_path, image_path = entry
        fruitlet_ims, cloud_ims, fruitlet_ids = self.load_data(annotations_path, image_path, should_flip)

        is_pad = np.zeros((self.max_fruitlets), dtype=bool)
        num_pad = self.max_fruitlets - fruitlet_ims.shape[0]

        if num_pad <= 0: # I always want one in case of matches logic below
            raise RuntimeError('Too many fruitlets')
        
        if num_pad > 0:
            fruitlet_ims, cloud_ims, fruitlet_ids = fruitlet_pad(num_pad, fruitlet_ims, cloud_ims, fruitlet_ids)
            is_pad[-num_pad:] = True

        if self.augment:
            randperm = torch.randperm(self.max_fruitlets)
            fruitlet_ims = fruitlet_ims[randperm]
            cloud_ims = cloud_ims[randperm]
            is_pad = is_pad[randperm]
            fruitlet_ids = fruitlet_ids[randperm]

        return file_key, fruitlet_ims, cloud_ims, is_pad, fruitlet_ids

    def __getitem__(self, index):
        entry_0, entry_1 = self.file_data[index]

        if self.augment:
            should_flip = np.random.uniform() < 0.5
        else:
            should_flip = False

        file_key_0, fruitlet_ims_0, cloud_ims_0, is_pad_0, fruitlet_ids_0 = self._get_data(entry_0, should_flip)
        file_key_1, fruitlet_ims_1, cloud_ims_1, is_pad_1, fruitlet_ids_1 = self._get_data(entry_1, should_flip)

        matches_gt = np.zeros((self.max_fruitlets, self.max_fruitlets)).astype(np.float32)
        masks_gt = np.ones((self.max_fruitlets, self.max_fruitlets)).astype(np.float32)

        masks_gt[is_pad_0, :] = 0.0
        masks_gt[:, is_pad_1] = 0.0

        for ind_0 in range(self.max_fruitlets):
            fruitlet_id_0 = fruitlet_ids_0[ind_0]
            if fruitlet_id_0 < 0:
                continue

            if not fruitlet_id_0 in fruitlet_ids_1:
                continue

            ind_1 = np.where(fruitlet_ids_1 == fruitlet_id_0)[0][0]
            matches_gt[ind_0, ind_1] = 1.0

        return file_key_0, fruitlet_ims_0, cloud_ims_0, \
               is_pad_0, fruitlet_ids_0, \
               file_key_1, fruitlet_ims_1, cloud_ims_1, \
               is_pad_1, fruitlet_ids_1, \
               matches_gt, masks_gt

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
    
def get_assoc_fruitlets(anno):
    fruitlet_ids = set()
    for det in anno:
        if det['fruitlet_id'] < 0:
            continue

        fruitlet_ids.add(det['fruitlet_id'])

    return fruitlet_ids