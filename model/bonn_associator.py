import os
import lightning as L
import numpy as np
import open3d
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch

from util.util import unravel_clouds, read_point_cloud

def get_ind_pairs(centroids):
    source_inds = []
    target_inds = []
    for ind_0 in range(centroids.shape[0]):
        for ind_1 in range(centroids.shape[0]):
            if ind_0 == ind_1:
                continue

            source_inds.append(ind_0)
            target_inds.append(ind_1)

    return source_inds, target_inds

def get_angles_and_z(centroids):
    source_inds, target_inds = get_ind_pairs(centroids)

    angle_array = [None]*centroids.shape[0]
    z_front_array = [None]*centroids.shape[0]

    for source_ind, target_ind in zip(source_inds, target_inds):
        xi, yi, zi = centroids[source_ind]
        xj, yj, zj = centroids[target_ind]

        thetaj = np.rad2deg(np.arctan2(xj - xi, yj - yi))
        zj = zj - zi

        if angle_array[source_ind] is None:
            angle_array[source_ind] = []
            z_front_array[source_ind] = []

        angle_array[source_ind].append(thetaj)
        z_front_array[source_ind].append(zj < 0)

    angle_array = np.array(angle_array)
    z_front_array = np.array(z_front_array)
    
    return angle_array, z_front_array

def get_descriptor(centroids):
    angle_array, z_front_array = get_angles_and_z(centroids)

    # NEEDS TO BE DIVISIBLE BY 180!!!
    bin_resolution = 30
    bin_centers = np.arange(-180, 180 + bin_resolution, bin_resolution)
    
    front_bins = np.zeros((centroids.shape[0], bin_centers.shape[0]))
    back_bins = np.zeros_like(front_bins)

    for ind, entry in enumerate(zip(angle_array, z_front_array)):
        angles, z_in_fronts = entry

        bin_indices = np.argmin(np.abs(angles[:, np.newaxis] - bin_centers), axis=1)

        for bi, in_front in zip(bin_indices, z_in_fronts):
            if in_front:
                front_bins[ind, bi] += 1
            else:
                back_bins[ind, bi] += 1

    #fix 180 and -180 thing
    front_bins[:, 0] += front_bins[:, -1]
    back_bins[:, 0] += back_bins[:, -1]

    front_bins = front_bins[:, 0:-1]
    back_bins = back_bins[:, 0:-1]
    
    histograms = np.concatenate([front_bins, back_bins], axis=-1)
    histograms = histograms / np.linalg.norm(histograms, axis=1)[:, None]

    return histograms

class BonnAssociator(L.LightningModule):
    def __init__(self,
                 full_cloud_dir,
                 **kwargs):
        super().__init__()

        self.full_cloud_dir = full_cloud_dir

    def forward(self, data_0, data_1):
        _, centroids_0, has_points_0 = unravel_clouds(*data_0[2:])
        _, centroids_1, has_points_1 = unravel_clouds(*data_1[2:])

        full_cloud_0 = read_point_cloud(os.path.join(self.full_cloud_dir, data_0[0] + '.pcd'))
        full_cloud_1 = read_point_cloud(os.path.join(self.full_cloud_dir, data_1[0] + '.pcd'))

        fruitlet_ids_0 = data_0[-1].numpy()
        fruitlet_ids_1 = data_1[-1].numpy()

        mean_vals_0 = data_0[1].numpy()
        mean_vals_1 = data_1[1].numpy()

        full_cloud_0 -= mean_vals_0
        full_cloud_1 -= mean_vals_1

        rad_0 = np.max(np.linalg.norm(centroids_0[has_points_0], axis=1))
        rad_1 = np.max(np.linalg.norm(centroids_1[has_points_1], axis=1))

        pad_rad_0 = rad_0 + 0.05
        pad_rad_1 = rad_1 + 0.05

        full_cloud_0 = full_cloud_0[np.linalg.norm(full_cloud_0, axis=1) < pad_rad_0]
        full_cloud_1 = full_cloud_1[np.linalg.norm(full_cloud_1, axis=1) < pad_rad_1]

        source = open3d.geometry.PointCloud()
        target = open3d.geometry.PointCloud()

        source.points = open3d.utility.Vector3dVector(full_cloud_0)
        target.points = open3d.utility.Vector3dVector(full_cloud_1)

        source = source.voxel_down_sample(voxel_size=0.001)
        target = target.voxel_down_sample(voxel_size=0.001)

        #coarse icp
        icp_coarse = open3d.pipelines.registration.registration_icp(
            source, target, 0.05, np.identity(4),
            estimation_method=open3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )

        icp_fine = open3d.pipelines.registration.registration_icp(
            source, target, 0.003, icp_coarse.transformation,
            estimation_method=open3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )

        M = icp_fine.transformation

        centroids_1_homo = np.ones((centroids_1.shape[0], 4))
        centroids_1_homo[:, 0:3] = centroids_1

        centroids_1_homo = (M @ centroids_1_homo.T).T
        centroids_1_new = centroids_1_homo[:, 0:3]
        centroids_1 = centroids_1_new

        val_inds_0 = np.arange(has_points_0.shape[0])[has_points_0]
        val_inds_1 = np.arange(has_points_1.shape[0])[has_points_1]
        centroids_0 = centroids_0[has_points_0]
        centroids_1 = centroids_1[has_points_1]

        # smaller one should be rows
        if centroids_0.shape[0] <= centroids_1.shape[0]:
            is_flip = False
        else:
            centroids_1, centroids_0 = centroids_0, centroids_1
            is_flip = True

        descriptors_0 = get_descriptor(centroids_0)
        descriptors_1 = get_descriptor(centroids_1)

        ALPHA = 0.75
        BETA = 2.0
        GAMMA = 2.5
        # ALPHA = 0.62
        # BETA = 0.15
        # GAMMA = 2.7
        p_dist = cdist(centroids_0, centroids_1)
        d_dist = cdist(descriptors_0, descriptors_1)

        cost = ALPHA*p_dist + BETA*d_dist
        unmatched_cost = np.zeros((centroids_0.shape[0], centroids_0.shape[0])) + GAMMA
        total_cost = np.concatenate([cost, unmatched_cost], axis=1)

        row_inds, col_inds = linear_sum_assignment(total_cost)
        row_inds = row_inds[col_inds < centroids_1.shape[0]]
        col_inds = col_inds[col_inds < centroids_1.shape[0]]

        if is_flip:
            row_inds, col_inds = col_inds, row_inds

        row_inds = val_inds_0[row_inds]
        col_inds = val_inds_1[col_inds]

        match_matrix = np.zeros((fruitlet_ids_0.shape[0], fruitlet_ids_1.shape[0]),
                                dtype=np.float32)
        
        for ind in range(len(row_inds)):
            row = row_inds[ind]
            col = col_inds[ind]

            match_matrix[row, col] = 1.0
        
        return torch.from_numpy(match_matrix)

    def test_step(self, batch, batch_idx):
        file_key_0, clouds_0, cloud_inds_0, \
               fruitlet_ids_0, \
               file_key_1, clouds_1, cloud_inds_1, \
               fruitlet_ids_1, \
               matches_gt, mean_vals_0, mean_vals_1 = batch
        
        if not clouds_0.shape[0] == 1:
            raise RuntimeError('only batch size 1 supported icp_assoc')

        data_0 = (file_key_0[0], mean_vals_0[0], clouds_0[0], cloud_inds_0[0], fruitlet_ids_0[0])
        data_1 = (file_key_1[0], mean_vals_1[0], clouds_1[0], cloud_inds_1[0], fruitlet_ids_1[0])

        matches_gt = matches_gt[0]
        matches_pred = self(data_0, data_1)

        true_pos = (matches_gt*matches_pred).sum()
        false_pos = ((1-matches_gt)*matches_pred).sum()
        false_neg = matches_gt.sum() - true_pos

        if true_pos > 0:
            precision = true_pos / (true_pos + false_pos)
            recall = true_pos / (true_pos + false_neg)
            f1 = 2*precision*recall/(precision + recall)
        else:
            precision = 0
            recall = 0
            f1 = 0

        self.log('precision', precision)
        self.log('recall', recall)
        self.log('f1', f1)

        return 0