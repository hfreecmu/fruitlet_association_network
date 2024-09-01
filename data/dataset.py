import os
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional

from data.interface import DatasetInterface
from util.util import read_json

# torch uses these for resnet and vit
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# timm vit uses this
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

# for point cloud
CLOUD_MEANS = [0.0, 0.0, 0.0]
CLOUD_STDS = [1.0, 1.0, 1.0]

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

def fruitlet_pad(num_pad, fruitlet_images, fruitlet_clouds, fruitlet_ids, pos_2ds):    
    # to not affect batch norm will select a random fruitlet
    rand_fruitlet_inds = np.random.randint(low=0, high=fruitlet_images.shape[0], size=num_pad)
    images_to_cat = torch.clone(fruitlet_images[rand_fruitlet_inds])
    fruitlet_images = torch.vstack([fruitlet_images, images_to_cat])

    clouds_to_cat = np.copy(fruitlet_clouds[rand_fruitlet_inds])
    fruitlet_clouds = np.concatenate([fruitlet_clouds, clouds_to_cat])

    # making this -2 to avoid confusion
    ids_to_cat = np.zeros((num_pad), dtype=fruitlet_ids.dtype) - 2
    fruitlet_ids = np.concatenate([fruitlet_ids, ids_to_cat])

    pos_2ds_to_cat = np.copy(pos_2ds[rand_fruitlet_inds])
    pos_2ds = np.concatenate([pos_2ds, pos_2ds_to_cat])

    return fruitlet_images, fruitlet_clouds, fruitlet_ids, pos_2ds

class AssociationDataset(DatasetInterface):
    def __init__(self,
                 anno_root,
                 anno_subdir, 
                 images_dir,
                 image_size,
                 encoder_type,
                 augment,
                 max_fruitlets,
                 is_test,
                 **kwargs
                 ):
        
        super().__init__(anno_root, anno_subdir, images_dir,
                         is_test)

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

        self.random_brightness = torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                                                    saturation=0.1, hue=0.05)

    def __len__(self):
        return len(self.file_data)
    
    def load_data(self, annotations_path, image_path, should_flip, can_drop):
        # read image and annotations
        image = torchvision.io.read_image(image_path)
        annotations = read_json(annotations_path)['annotations']

        # transform image by scaling to 0->1 and normalizing
        image = self.image_transform(image)
        if self.augment and (np.random.uniform() < self.augment.bright_pct):
            image = self.random_brightness(image)

        # get fruitlet data
        fruitlet_ims = []
        cloud_boxes = []
        fruitlet_ids = []
        pos_2ds = []

        full_cloud_points = {}

        if self.augment and (np.random.uniform() < self.augment.rotate_pct):
            rotate_rand = np.random.uniform()
            rotate_theta = np.deg2rad(np.random.uniform(-10, 10))
            if rotate_rand < 0.25:
                rotation = np.array([[np.cos(rotate_theta), -np.sin(rotate_theta), 0],
                                     [np.sin(rotate_theta), np.cos(rotate_theta), 0],
                                     [0, 0, 1.0]
                                     ])
            elif rotate_rand < 0.5:
                rotation = np.array([[np.cos(rotate_theta), 0, np.sin(rotate_theta)],
                                     [0, 1.0, 0],
                                     [-np.sin(rotate_theta), 0, np.cos(rotate_theta)]])
            elif rotate_rand < 0.75:
                rotation = np.array([[1.0, 0, 0],
                                     [0, np.cos(rotate_theta), -np.sin(rotate_theta)],
                                     [0, np.sin(rotate_theta), np.cos(rotate_theta)]
                                     ])
            else:
                rotation = np.eye(3)
        else:
            rotation = np.eye(3)

        used_det_inds = []
        for det_ind, det in enumerate(annotations):
            if det['fruitlet_id'] < 0:
                continue

            # get cloud_points
            if len(det['cloud_points']) > 0:
                cloud_points = np.array(det['cloud_points'])

                # normalize
                cloud_points = (cloud_points - CLOUD_MEANS) / CLOUD_STDS

                # if should flip then cloud points are flipped along the x axis
                if should_flip:
                    cloud_points[:, 0] = -cloud_points[:, 0]

                # rotate cloud points after flipping
                cloud_points = (rotation @ cloud_points.T).T

                # rand drop 10% of cloud points
                # will not have effect unless min / max is dropped
                if self.augment and (np.random.uniform() < self.augment.drop_cloud_pct):
                    clouds_to_keep = int(np.round(cloud_points.shape[0]*0.9))
                    rand_inds = np.random.permutation(cloud_points.shape[0])
                    cloud_points = cloud_points[rand_inds[0:clouds_to_keep]] 

                # get mins and maxes
                mins = cloud_points.min(axis=0)
                maxs = cloud_points.max(axis=0)
                medians = np.median(cloud_points, axis=0)

                full_cloud_points[det_ind] = cloud_points
                box_3d = np.array([mins[0], maxs[0], mins[1], maxs[1], medians[2], 1.0])
            else:
                box_3d = [-1.0, -1.0, -1.0, -1.0, -1.0, 0.0]
            
            x0 = det["x0"]
            y0 = det["y0"]
            x1 = det["x1"]
            y1 = det["y1"]
            seg_inds = np.array(det["seg_inds"])
            fruitlet_id = det["fruitlet_id"]

            # rand drop 10% of seginds
            if self.augment and (np.random.uniform() < self.augment.drop_seg_pct):
                segs_to_keep = int(np.round(seg_inds.shape[0]*0.9))
                rand_inds = np.random.permutation(seg_inds.shape[0])
                seg_inds = seg_inds[rand_inds[0:segs_to_keep]]

            round_x0 = int(np.round(x0))
            round_y0 = int(np.round(y0))
            round_x1 = int(np.round(x1))
            round_y1 = int(np.round(y1))

            fruitlet_im = image[:, round_y0:round_y1, round_x0:round_x1]
            seg_im = torch.zeros_like(image[0:1, :, :])
            seg_im[:, seg_inds[:, 0], seg_inds[:, 1]] = 1.0
            seg_im = seg_im[:, round_y0:round_y1, round_x0:round_x1]

            # combine fruitlet im and seg_im
            fruitlet_im = torch.concatenate([fruitlet_im, seg_im])

            # for 2d pos encoding
            pos_x0 = round_x0
            pos_y0 = round_y0
            pos_x1 = round_x1
            pos_y1 = round_y1

            # random crop for image
            if self.augment and (np.random.uniform() < self.augment.crop_pct):
                _, fruitlet_height, fruitlet_width = fruitlet_im.shape
                new_height = int(np.random.uniform(low=0.9, high=1.0)*fruitlet_height)
                new_width = int(np.random.uniform(low=0.9, high=1.0)*fruitlet_width)
                i, j, h, w = torchvision.transforms.RandomCrop.get_params(fruitlet_im, 
                                                                       output_size=(new_height, new_width))
                
                fruitlet_im = torchvision.transforms.functional.crop(fruitlet_im,
                                                                     i, j, h, w)
                
                pos_x0 = pos_x0 + j
                pos_y0 = pos_y0 + i
                pos_x1 = pos_x0 + w
                pos_y1 = pos_y0 + h

            _, fruitlet_height, fruitlet_width = fruitlet_im.shape
            pos_2d = np.array([pos_x0, pos_y0, pos_x1, pos_y1], dtype=float)

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

            fruitlet_ims.append(fruitlet_im)
            cloud_boxes.append(box_3d)
            fruitlet_ids.append(fruitlet_id)
            used_det_inds.append(det_ind)
            pos_2ds.append(pos_2d)

        fruitlet_ims = torch.stack(fruitlet_ims)
        cloud_boxes = np.array(cloud_boxes)
        fruitlet_ids = np.array(fruitlet_ids)
        used_det_inds = np.array(used_det_inds)
        pos_2ds = np.array(pos_2ds)

        # rand drop a single fruitlet from assoc
        # only if more than 3
        if self.augment and can_drop \
            and fruitlet_ims.shape[0] > 3:

            rand_drop_ind = np.random.randint(0, fruitlet_ims.shape[0])
            fruitlet_ims = torch.concatenate([fruitlet_ims[0:rand_drop_ind], fruitlet_ims[rand_drop_ind+1:]])
            cloud_boxes = np.concatenate([cloud_boxes[0:rand_drop_ind], cloud_boxes[rand_drop_ind+1:]])
            fruitlet_ids = np.concatenate([fruitlet_ids[0:rand_drop_ind], fruitlet_ids[rand_drop_ind+1:]])
            pos_2ds = np.concatenate([pos_2ds[0:rand_drop_ind], pos_2ds[rand_drop_ind+1:]])
            
            if used_det_inds[rand_drop_ind] in full_cloud_points:
                del full_cloud_points[used_det_inds[rand_drop_ind]]

            used_det_inds = np.concatenate([used_det_inds[0:rand_drop_ind], used_det_inds[rand_drop_ind+1:]])


        if len(full_cloud_points) > 0:
            full_cloud_points = [full_cloud_points[key] for key in full_cloud_points]
            full_cloud_points = np.concatenate(full_cloud_points)

            mean_3d = np.mean(full_cloud_points, axis=0)

            cloud_box_adjust_inds = (cloud_boxes[:, -1] == 1.0)
            cloud_boxes[cloud_box_adjust_inds, 0:2] -= mean_3d[0]
            cloud_boxes[cloud_box_adjust_inds, 2:4] -= mean_3d[1]
            cloud_boxes[cloud_box_adjust_inds, 4] -= mean_3d[2]

        # if should flip then flip left and right cloud images
        if should_flip:
            fruitlet_ims = torch.flip(fruitlet_ims, dims=(-1,))
            pos_2ds[:, 0] = image.shape[2] - pos_2ds[:, 0]
            pos_2ds[:, 2] = image.shape[2] - pos_2ds[:, 2]
        
        # normalize pos_2ds
        cxs = (pos_2ds[:, 0] + pos_2ds[:, 2]) / 2
        cys = (pos_2ds[:, 1] + pos_2ds[:, 3]) / 2
        cx_mean = cxs.mean()
        cy_mean = cys.mean()
        pos_2ds[:, 0] -= cx_mean
        pos_2ds[:, 2] -= cx_mean
        pos_2ds[:, 1] -= cy_mean
        pos_2ds[:, 3] -= cy_mean
        #TODO make them between 0 and 1?
        # same applies to 3d?

        return fruitlet_ims, cloud_boxes, fruitlet_ids, used_det_inds, pos_2ds

    def _get_data(self, entry, should_flip, can_drop):
        file_key, annotations_path, image_path = entry
        fruitlet_ims, cloud_boxes, fruitlet_ids, used_det_inds, pos_2ds = self.load_data(annotations_path, image_path, should_flip, can_drop)

        is_pad = np.zeros((self.max_fruitlets), dtype=bool)
        num_pad = self.max_fruitlets - fruitlet_ims.shape[0]

        # TODO is this still valid?
        # I do not think so. let's get rid of it later
        if num_pad <= 0: # I always want one in case of matches logic below
            raise RuntimeError('Too many fruitlets')
        
        if num_pad > 0:
            fruitlet_ims, cloud_boxes, fruitlet_ids, pos_2ds = fruitlet_pad(num_pad, fruitlet_ims, cloud_boxes, fruitlet_ids, pos_2ds)
            is_pad[-num_pad:] = True

            used_det_inds_pad = np.zeros((self.max_fruitlets), dtype=int)
            used_det_inds_pad[0:used_det_inds.shape[0]] = used_det_inds
            used_det_inds = used_det_inds_pad

        if self.augment:
            randperm = torch.randperm(self.max_fruitlets)
            fruitlet_ims = fruitlet_ims[randperm]
            cloud_boxes = cloud_boxes[randperm]
            is_pad = is_pad[randperm]
            fruitlet_ids = fruitlet_ids[randperm]
            pos_2ds = pos_2ds[randperm]
            used_det_inds = used_det_inds[randperm]

        cloud_boxes = cloud_boxes.astype(np.float32)
        pos_2ds = pos_2ds.astype(np.float32)

        return file_key, fruitlet_ims, cloud_boxes, is_pad, fruitlet_ids, image_path, annotations_path, used_det_inds, pos_2ds

    def __getitem__(self, index):
        entry_0, entry_1 = self.file_data[index]

        if self.augment:
            should_flip = np.random.uniform() < 0.5
        else:
            should_flip = False

        can_drop_0 = False
        if self.augment:
            if np.random.uniform() < 0.5:
                entry_0, entry_1 = entry_1, entry_0
            
            if np.random.uniform() < self.augment.drop_fruitlet_pct:
                can_drop_0 = True

        file_key_0, fruitlet_ims_0, cloud_boxes_0, is_pad_0, fruitlet_ids_0, im_path_0, anno_path_0, det_inds_0, pos_2ds_0 = self._get_data(entry_0, should_flip, can_drop_0)
        file_key_1, fruitlet_ims_1, cloud_boxes_1, is_pad_1, fruitlet_ids_1, im_path_1, anno_path_1, det_inds_1, pos_2ds_1 = self._get_data(entry_1, should_flip, False)

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

        if not self.is_test:
            return file_key_0, fruitlet_ims_0, cloud_boxes_0, \
                is_pad_0, fruitlet_ids_0, pos_2ds_0, \
                file_key_1, fruitlet_ims_1, cloud_boxes_1, \
                is_pad_1, fruitlet_ids_1, pos_2ds_1, \
                matches_gt, masks_gt
        else:
            return file_key_0, fruitlet_ims_0, cloud_boxes_0, \
                is_pad_0, fruitlet_ids_0, pos_2ds_0, \
                file_key_1, fruitlet_ims_1, cloud_boxes_1, \
                is_pad_1, fruitlet_ids_1, pos_2ds_1, \
                matches_gt, masks_gt, \
                im_path_0, anno_path_0, det_inds_0, \
                im_path_1, anno_path_1, det_inds_1
    