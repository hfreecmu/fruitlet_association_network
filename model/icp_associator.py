import open3d
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import lightning as L

from util.util import unravel_clouds

class ICPAssociator(L.LightningModule):
    def __init__(self,
                 **kwargs):
        super().__init__()

        self.errors = []

    def forward(self, data_0, data_1):
        _, centroids_0, has_points_0 = unravel_clouds(*data_0)
        _, centroids_1, has_points_1 = unravel_clouds(*data_1)

        full_cloud_0 = data_0[0]
        full_cloud_1 = data_1[0]

        fruitlet_ids_0 = data_0[2].numpy()
        fruitlet_ids_1 = data_1[2].numpy()

        source = open3d.geometry.PointCloud()
        target = open3d.geometry.PointCloud()

        source.points = open3d.utility.Vector3dVector(full_cloud_1.numpy())
        target.points = open3d.utility.Vector3dVector(full_cloud_0.numpy())

        source = source.voxel_down_sample(voxel_size=0.001)
        target = target.voxel_down_sample(voxel_size=0.001)

        icp_coarse = open3d.pipelines.registration.registration_icp(
            source, target, 0.03, np.identity(4),
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

        dist = cdist(centroids_0, centroids_1_new)

        val_inds_0 = np.arange(has_points_0.shape[0])[has_points_0]
        val_inds_1 = np.arange(has_points_1.shape[0])[has_points_1]

        row_inds, col_inds = linear_sum_assignment(dist[has_points_0][:, has_points_1])
        row_inds = val_inds_0[row_inds]
        col_inds = val_inds_1[col_inds]

        match_matrix = np.zeros((fruitlet_ids_0.shape[0], fruitlet_ids_1.shape[0]),
                                dtype=np.float32)

        for ind in range(len(row_inds)):
            row = row_inds[ind]
            col = col_inds[ind]

            if dist[row, col] >= 0.01:
                continue

            match_matrix[row, col] = 1.0
        
        return torch.from_numpy(match_matrix)

    def test_step(self, batch, batch_idx):
        file_key_0, clouds_0, cloud_inds_0, \
               fruitlet_ids_0, \
               file_key_1, clouds_1, cloud_inds_1, \
               fruitlet_ids_1, \
               matches_gt, _, _ = batch
        
        if not clouds_0.shape[0] == 1:
            raise RuntimeError('only batch size 1 supported icp_assoc')

        data_0 = (clouds_0[0], cloud_inds_0[0], fruitlet_ids_0[0])
        data_1 = (clouds_1[0], cloud_inds_1[0], fruitlet_ids_1[0])

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

        if f1 > 0:
            f1 = f1.item()
            
        if f1 != 1:
            self.errors.append([file_key_0[0], file_key_1[0], f1])

        return 0
