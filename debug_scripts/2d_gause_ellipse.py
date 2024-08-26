import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import open3d
import distinctipy

# Following https://stackoverflow.com/questions/67718828/how-can-i-plot-an-ellipse-from-eigenvalues-and-eigenvectors-in-python-matplotl
# except using sklearn pca which has the same result

# https://math.stackexchange.com/questions/1447730/drawing-ellipse-from-eigenvalue-eigenvector
# may also be helpful

def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def create_point_cloud(cloud_path, points, colors, normals=None, estimate_normals=False):
    cloud = open3d.geometry.PointCloud()
    cloud.points = open3d.utility.Vector3dVector(points)
    cloud.colors = open3d.utility.Vector3dVector(colors)

    if normals is not None:
        cloud.normals = open3d.utility.Vector3dVector(normals)
    elif estimate_normals:
        cloud.estimate_normals(
            search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))

    open3d.io.write_point_cloud(
        cloud_path,
        cloud
    )
    
cloud_seg_path = '/home/frc-ag-3/Downloads/debug_fruitlet/debug_cloud/point_clouds/2021_54_5_left.pkl'
vis_dir = '/home/frc-ag-3/Downloads/debug_fruitlet/debug_cloud/vis_ellipses'

cloud_segmentaions = read_pickle(cloud_seg_path)
colors = distinctipy.get_colors(len(cloud_segmentaions))

full_cloud = []
full_colors = []
for ind in range(len(cloud_segmentaions)):
    cloud_points = cloud_segmentaions[ind]
    color = colors[ind]

    nan_inds = np.isnan(cloud_points).any(axis=1)
    cloud_points = cloud_points[~nan_inds]

    if cloud_points.shape[0] == 0:
        print('empty cloud')
        continue

    med_vals = np.median(cloud_points, axis=0)

    cloud_points = cloud_points[:, 0:2]



    # all 3 below get me the same results so I will use sklearn pca
    # cloud_points_cent = cloud_points - cloud_points.mean(axis=0)
    # cov = np.cov(cloud_points_cent.T)
    # eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # cov = np.cov(cloud_points.T)
    # eigenvalues_2, eigenvectors_2 = np.linalg.eig(cov)

    pca = PCA(n_components=2)
    _ = pca.fit_transform(cloud_points)
    eig_vals, eig_vecs = pca.explained_variance_, pca.components_

    # theta = np.linspace(0, 2*np.pi, 1000)
    # ellipsis = ((np.sqrt(eig_vals[None,:]) * eig_vecs) @ [np.sin(theta), np.cos(theta)]).T

    # plt.plot(ellipsis[:,0], ellipsis[:,1])
    # plt.show()

    # ellipse_points = np.zeros((ellipsis.shape[0], 3))
    # ellipse_points[:, 0] = ellipsis[:, 0]
    # ellipse_points[:, 1] = ellipsis[:, 1]
    
    # ellipse_points = ellipse_points + med_vals
    # ellipse_colors = np.zeros_like(ellipse_points) + color

    # full_cloud.append(ellipse_points)
    # full_colors.append(ellipse_colors)

    # # from gauss splat
    # A = eig_vecs @ np.diag(np.sqrt(eig_vals)) @ np.diag(np.sqrt(eig_vals)) @ eig_vecs.T
    # # this is always 0 so I guess we are doing something right
    # diff = np.max(np.abs(A - np.cov(cloud_points.T)))

    scale_0, scale_1 = np.sqrt(eig_vals)
    rot_mat = eig_vecs

    # we do this above but I want to match with my own code from gauss splat and not from links above
    theta = np.linspace(0, 2*np.pi, 1000)
    x_points = np.cos(theta)
    y_points = np.sin(theta)
    
    # it is the same as above hooray
    # my_ellipse = (rot_mat @ np.diag([scale_0, scale_1]) @ np.stack([y_points, x_points])).T
    # plt.plot(my_ellipse[:,0], my_ellipse[:,1], '--')
    # plt.show()

    # not x and y are swapped from above trials because I like it better like this
    ellipse = (rot_mat @ np.diag([scale_0, scale_1]) @ np.stack([x_points, y_points])).T

    ellipse_points = np.zeros((ellipse.shape[0], 3))
    ellipse_points[:, 0] = ellipse[:, 0]
    ellipse_points[:, 1] = ellipse[:, 1]
    
    ellipse_points = ellipse_points + med_vals
    ellipse_colors = np.zeros_like(ellipse_points) + color

    full_cloud.append(ellipse_points)
    full_colors.append(ellipse_colors)

    breakpoint()

full_cloud = np.vstack(full_cloud)
full_colors = np.vstack(full_colors)

vis_path = os.path.join(vis_dir, os.path.basename(cloud_seg_path).replace('.pkl', '.pcd'))
create_point_cloud(vis_path, full_cloud, full_colors)