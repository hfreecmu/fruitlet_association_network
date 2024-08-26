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
    
cloud_seg_path = '/home/frc-ag-3/Downloads/debug_fruitlet/debug_cloud/point_clouds/2021_42_13_left.pkl'
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

    cloud_points = cloud_points[:, 0:3]

    pca = PCA(n_components=3)
    _ = pca.fit_transform(cloud_points)
    eig_vals, eig_vecs = pca.explained_variance_, pca.components_

    scale_0, scale_1, scale_2 = np.sqrt(eig_vals)
    rot_mat = eig_vecs

    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(x, y, z)
    #plt.show()
    
    ellipse_points = (rot_mat @ np.diag([scale_0, scale_1, scale_2]) @ np.stack([x, y, z])).T
    ellipse_points = ellipse_points + med_vals
    ellipse_colors = np.zeros_like(ellipse_points) + color

    full_cloud.append(ellipse_points)
    full_colors.append(ellipse_colors)

full_cloud = np.vstack(full_cloud)
full_colors = np.vstack(full_colors)

vis_path = os.path.join(vis_dir, os.path.basename(cloud_seg_path).replace('.pkl', '.pcd'))
create_point_cloud(vis_path, full_cloud, full_colors)