import os
import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import cv2
from sklearn.decomposition import PCA

from util.util import get_identifier, read_json, read_pickle

FRUITLET_MEAN = [0.30344757, 0.3133871, 0.32248256]
FRUITLET_STD = [0.051711865, 0.0505018, 0.056481156]

ELLIPSE_MEAN = [[0.01515367144828234, -0.014630158413319364, 0.4377650249983249], 0.004248618083974694, 0.0021318646462382008]
ELLIPSE_STD = [[0.03693850598871499, 0.028846982679794086, 0.07828516204898538], 0.0021931332935247344, 0.000561814052844269]


def get_image_transform():
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ConvertImageDtype(torch.float32),
        torchvision.transforms.Normalize(mean=FRUITLET_MEAN, std=FRUITLET_STD),
    ])

    return image_transform

def get_fruitlet_transform(fruitlet_image_size):
    fruitlet_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((fruitlet_image_size, fruitlet_image_size)),
    ])

    return fruitlet_transform

def vis_padded_resized_im(padded_im, identifier, det_ind, output_dir):
    padded_im = padded_im.numpy().transpose(1, 2, 0)
    padded_im = padded_im * FRUITLET_STD + FRUITLET_MEAN

    padded_im = (padded_im * 255).astype(np.uint8)

    padded_im = cv2.cvtColor(padded_im, cv2.COLOR_RGB2BGR)

    if identifier is not None:
        output_name = identifier + '_' + str(det_ind) + '.png'
    else:
        output_name = str(det_ind) + '.png'

    output_path = os.path.join(output_dir, output_name)

    cv2.imwrite(output_path, padded_im)

#TODO could add augment here to add unecessary fruitlet
def load_data(annotations_path, image_path, cloud_path, 
              image_transform, fruitlet_transform):
    
    image = torchvision.io.read_image(image_path)
    annotations = read_json(annotations_path)['annotations']
    seg_clouds = read_pickle(cloud_path)

    image = image_transform(image)

    fruitlet_model_ims = []
    fruitlet_ellipses = []
    fruitlet_ids = []
    is_gt_fruitlets = []
    for ind in range(len(annotations)):
        anno = annotations[ind]
        cloud_points = seg_clouds[ind]

        ######## Start with annotations ########
        x0 = anno["x0"]
        y0 = anno["y0"]
        x1 = anno["x1"]
        y1 = anno["y1"]
        fruitlet_id = anno["fruitlet_id"]

        round_x0 = int(np.round(x0))
        round_y0 = int(np.round(y0))
        round_x1 = int(np.round(x1))
        round_y1 = int(np.round(y1))

        fruitlet_im = image[:, round_y0:round_y1, round_x0:round_x1]
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

        
        padded_im = F.pad(fruitlet_im, padding, mode='constant', value=0)
        fruitlet_model_im = fruitlet_transform(padded_im)
        fruitlet_model_ims.append(fruitlet_model_im)

        # import uuid
        # identifier = uuid.uuid4().hex[0:8]
        # vis_padding_dir = '/home/frc-ag-3/Downloads/debug_fruitlet/vis_paddings'
        # vis_resize_dir = '/home/frc-ag-3/Downloads/debug_fruitlet/vis_resizes'
        # vis_padded_resized_im(padded_im, identifier, ind, vis_padding_dir)
        # vis_padded_resized_im(fruitlet_model_im, identifier, ind, vis_resize_dir)
        ######## End annotations ########

        ######## Start ellipse ########
        nan_inds = np.isnan(cloud_points).any(axis=1)
        cloud_points = cloud_points[~nan_inds]
        if cloud_points.shape[0] == 0:
            fruitlet_ellipses.append([0, 0, 0, 0, 0, 0, 0])
        else:
            med_vals = np.median(cloud_points, axis=0)

            cloud_points = cloud_points[:, 0:2]
            pca = PCA(n_components=2)
            _ = pca.fit_transform(cloud_points)
            eig_vals, eig_vecs = pca.explained_variance_, pca.components_
            scale_0, scale_1 = np.sqrt(eig_vals)
            rot_mat = eig_vecs
            theta = np.arccos(rot_mat[0, 0])
            while theta > np.pi/2:
                theta = theta - np.pi
            while theta <= -np.pi / 2:
                theta = theta + np.pi

            # theta = np.linspace(0, 2*np.pi, 1000)
            # x_points = np.cos(theta)
            # y_points = np.sin(theta)
            # ellipse = (rot_mat @ np.diag([scale_0, scale_1]) @ np.stack([x_points, y_points])).T
            # theta_qua = np.arccos(rot_mat[0, 0])
            # rot_mat[0, 0] = np.cos(theta_qua - np.pi)
            # rot_mat[0, 1] = np.sin(np.pi - theta_qua)
            # rot_mat[1, 0] = np.sin(theta_qua - np.pi)
            # rot_mat[1, 1] = np.cos(theta_qua - np.pi)
            # ellipse_2 = (rot_mat @ np.diag([scale_0, scale_1]) @ np.stack([x_points, y_points])).T
            # import matplotlib.pyplot as plt
            # plt.plot(ellipse[:,0], ellipse[:,1])
            # plt.plot(ellipse_2[:,0], ellipse_2[:,1], '--')
            # plt.show()

            # normalize values
            
            # between -pi / 2 and pi / 2
            # make between -1 and 1
            theta = 2 * theta / np.pi
            scale_0 = (scale_0 - ELLIPSE_MEAN[1]) / ELLIPSE_STD[1]
            scale_1 = (scale_1 - ELLIPSE_MEAN[2]) / ELLIPSE_STD[2]
            med_vals = (med_vals - ELLIPSE_MEAN[0]) / ELLIPSE_STD[0]

            fruitlet_ellipses.append(med_vals.tolist() + [scale_0, scale_1, theta, 1])
        ######## End ellipse ########

        ######## Start is_gt_fruitlet #######
        is_gt_fruitlet = (fruitlet_id >= 0) # I didn't use 0 but just in case

        fruitlet_ids.append(fruitlet_id)
        is_gt_fruitlets.append(is_gt_fruitlet)
        ######## End is_gt_fruitlet ########

    fruitlet_model_ims = torch.stack(fruitlet_model_ims)
    fruitlet_ellipses = np.stack(fruitlet_ellipses).astype(np.float32)
    fruitlet_ids = np.array(fruitlet_ids)
    is_gt_fruitlets = np.array(is_gt_fruitlets)
    
    return fruitlet_model_ims, fruitlet_ellipses, fruitlet_ids, is_gt_fruitlets

def fruitlet_pad(num_pad, fruitlet_images, fruitlet_ellipses, fruitlet_ids):
    img_chann, img_height, img_width = fruitlet_images.shape[1:]
    
    images_to_cat = torch.zeros(*[num_pad, img_chann, img_height, img_width], dtype=fruitlet_images.dtype)
    fruitlet_images = torch.vstack([fruitlet_images, images_to_cat])

    ellipses_to_cat = np.zeros((num_pad, 7), dtype=fruitlet_ellipses.dtype)
    fruitlet_ellipses = np.vstack([fruitlet_ellipses, ellipses_to_cat])

    # making this -2 to avoid confusion
    ids_to_cat = np.zeros((num_pad), dtype=fruitlet_ids.dtype) - 2
    fruitlet_ids = np.concatenate([fruitlet_ids, ids_to_cat])

    return fruitlet_images, fruitlet_ellipses, fruitlet_ids

class AssociationDataset(Dataset):
    def __init__(self,
                 annotations_dir,
                 images_dir,
                 pointcloud_dir,
                 image_size,
                 augment,
                 max_gt_fruitlets=8,
                 ignore_right=True,
                 ):
        
        super().__init__()
        
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir
        self.pointcloud_dir = pointcloud_dir
        self.max_gt_fruitlets = max_gt_fruitlets
        self.augment = augment

        self.image_transform = get_image_transform()
        self.fruitlet_transform = get_fruitlet_transform(image_size)

        self.file_data = self._get_files(ignore_right)

    def __len__(self):
        return len(self.file_data)
    
    def __getitem__(self, index):
        entry_0, entry_1 = self.file_data[index]
        file_key_0, annotations_path_0, image_path_0, cloud_path_0 = entry_0
        file_key_1, annotations_path_1, image_path_1, cloud_path_1 = entry_1

        data_0 = load_data(annotations_path_0, image_path_0, cloud_path_0, 
                           self.image_transform, self.fruitlet_transform)
        data_1 = load_data(annotations_path_1, image_path_1, cloud_path_1, 
                           self.image_transform, self.fruitlet_transform)
        
        fruitlet_images_0, fruitlet_ellipses_0, fruitlet_ids_0, is_gt_fruitlets_0 = data_0
        fruitlet_images_1, fruitlet_ellipses_1, fruitlet_ids_1, is_gt_fruitlets_1 = data_1

        # get only gt fruitlet data
        # augment possibility
        fruitlet_images_0 = fruitlet_images_0[is_gt_fruitlets_0]
        fruitlet_ellipses_0 = fruitlet_ellipses_0[is_gt_fruitlets_0]
        fruitlet_ids_0 = fruitlet_ids_0[is_gt_fruitlets_0]

        fruitlet_images_1 = fruitlet_images_1[is_gt_fruitlets_1]
        fruitlet_ellipses_1 = fruitlet_ellipses_1[is_gt_fruitlets_1]
        fruitlet_ids_1 = fruitlet_ids_1[is_gt_fruitlets_1]

        # final step, done after aug
        # but should maybe shuffle
        is_pad_0 = np.zeros((self.max_gt_fruitlets), dtype=bool)
        is_pad_1 = np.zeros((self.max_gt_fruitlets), dtype=bool)

        num_pad_0 = self.max_gt_fruitlets - fruitlet_images_0.shape[0]
        if num_pad_0 < 0:
            raise RuntimeError('Too many fruitlets 0')
        
        num_pad_1 = self.max_gt_fruitlets - fruitlet_images_1.shape[0]
        if num_pad_1 < 0:
            raise RuntimeError('Too many fruitlets 1')
        
        if num_pad_0 > 0:
            fruitlet_images_0, fruitlet_ellipses_0, fruitlet_ids_0 = fruitlet_pad(num_pad_0, fruitlet_images_0, fruitlet_ellipses_0, fruitlet_ids_0)
            is_pad_0[-num_pad_0:] = True

        if num_pad_1 > 0:
            fruitlet_images_1, fruitlet_ellipses_1, fruitlet_ids_1 = fruitlet_pad(num_pad_1, fruitlet_images_1, fruitlet_ellipses_1, fruitlet_ids_1)
            is_pad_1[-num_pad_1:] = True

        matches_0 = np.zeros_like(fruitlet_ids_0) - 1
        matches_1 = np.zeros_like(fruitlet_ids_1) - 1
        match_ind = 0
        for ind_0 in range(self.max_gt_fruitlets):
            fruitlet_id_0 = fruitlet_ids_0[ind_0]
            if fruitlet_id_0 < 0:
                continue

            if not fruitlet_id_0 in fruitlet_ids_1:
                continue

            ind_1 = np.where(fruitlet_ids_1 == fruitlet_id_0)[0][0]

            matches_0[match_ind] = ind_0
            matches_1[match_ind] = ind_1
            match_ind += 1

        return file_key_0, fruitlet_images_0, fruitlet_ellipses_0, \
               fruitlet_ids_0, is_pad_0, matches_0, \
               file_key_1, fruitlet_images_1, fruitlet_ellipses_1, \
               fruitlet_ids_1, is_pad_1, matches_1

    def _get_files(self, ignore_right):
        cluster_dict = {}
        for filename in os.listdir(self.annotations_dir):
            if not filename.endswith('.json'):
                continue

            if (ignore_right) and ('right' in filename):
                continue

            file_key = filename.split('.json')[0]

            annotations_path = os.path.join(self.annotations_dir, filename)
            image_path = os.path.join(self.images_dir, file_key + '.png')
            cloud_path = os.path.join(self.pointcloud_dir, file_key + '.pkl')

            if not os.path.exists(image_path):
                raise RuntimeError('Image path dne: ' + image_path)
            
            if not os.path.exists(cloud_path):
                raise RuntimeError('Cloud path dne: ' + cloud_path)
            
            identifier = get_identifier(file_key)

            if not identifier in cluster_dict:
                cluster_dict[identifier] = []

            entry = (file_key, annotations_path, image_path, cloud_path)
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

def get_data_loader(annotations_dir,
                    images_dir,
                    pointcloud_dir,
                    image_size,
                    augment,
                    batch_size,
                    shuffle):
    
    dataset = AssociationDataset(annotations_dir, images_dir, pointcloud_dir,
                                 image_size, augment)

    dloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dloader