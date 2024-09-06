import os
import numpy as np
import open3d
import distinctipy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import json

def cdf(x, ax, normed=True, *args, **kwargs):
    x = sorted(x)
    y = np.arange(len(x))
    if normed:
        y = y / len(x)

    ax.plot(x, y, *args, **kwargs)

    return x, y

def write_json(path, data, pretty=False):
    with open(path, 'w') as f:
        if not pretty:
            json.dump(data, f)
        else:
            json.dump(data, f, indent=4)

def sample_hull(pts, num_pts):
    shift_pts = np.roll(pts, -1, axis=0)

    dists = np.linalg.norm(shift_pts - pts, axis=1)
    total_length = dists.sum()

    seg_pcts = dists / total_length
    num_seg_pts = np.round(seg_pcts * num_pts).astype(int)
    
    while num_seg_pts.sum() > num_pts:
        densest_ind = np.argmax(num_seg_pts / dists)
        num_seg_pts[densest_ind] -= 1

    while num_seg_pts.sum() < num_pts:
        sparsest_ind = np.argmin(num_seg_pts / dists)
        num_seg_pts[sparsest_ind] += 1

    full_seg_pts = []
    for ind in range(pts.shape[0]):
        curr_pt = pts[ind]
        next_pt = shift_pts[ind]
        ns = num_seg_pts[ind]

        seg_pts = np.linspace(curr_pt, next_pt, ns + 1, endpoint=True)[0:-1]
        full_seg_pts.append(seg_pts)

    full_seg_pts = np.concatenate(full_seg_pts)

    return full_seg_pts

def process_file(filename, input_dir, output_dir, colors, cloud_max_pts=None, hull_num_pts = 500):
    input_path = os.path.join(input_dir, filename)
    
    data_array = np.loadtxt(input_path)

    pc_points = data_array[:,0:3]
    #-1 because maize labelled two ways
    labels = data_array[:, -1].astype(int)

    non_bg_inds = (labels > 1)
    pc_points = pc_points[non_bg_inds]
    labels = labels[non_bg_inds]

    color_inds = labels % colors.shape[0]
    pc_colors = colors[color_inds]
    
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(pc_points)
    pc.colors = open3d.utility.Vector3dVector(pc_colors)

    cloud_path = os.path.join(output_dir, filename.replace('.txt', '.pcd'))
    open3d.io.write_point_cloud(cloud_path, pc)

    full_proj_points = []
    full_proj_colors = []
    hull_points = []
    hull_colors = []
    leaf_data = {}

    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_inds = (labels == label)

        label_points = pc_points[label_inds]
        label_color = pc_colors[label_inds][0]

        pca = PCA(n_components=2)
        pca.fit(label_points)
        leaf_princ_0, leaf_princ_1 = pca.components_.tolist()

        projected_points = pca.transform(label_points)

        if cloud_max_pts is not None and projected_points.shape[0] > cloud_max_pts:
            proj_cloud = open3d.geometry.PointCloud()
            proj_cloud.points = open3d.utility.Vector3dVector(np.stack((projected_points[:, 0], projected_points[:, 1], np.zeros_like(projected_points[:, 0])), axis=1))
            proj_down = proj_cloud.farthest_point_down_sample(cloud_max_pts)

            projected_points = np.array(proj_down.points)[:, 0:2]
       
        hull = ConvexHull(projected_points)
        hull_pts = projected_points[hull.vertices]
        hull_pts = sample_hull(hull_pts, hull_num_pts)

        hull_min = hull_pts.min(axis=0)
        hull_max = hull_pts.max(axis=0)

        leaf_length, leaf_width = hull_max - hull_min

        reprojected_points = pca.inverse_transform(projected_points)
        reprojected_hull = pca.inverse_transform(hull_pts)

        leaf_location = reprojected_points.mean(axis=0) # hull or points?

        full_proj_points.append(reprojected_points)
        full_proj_colors.append(np.zeros_like(reprojected_points) + label_color)

        hull_points.append(reprojected_hull)
        hull_colors.append(np.zeros_like(reprojected_hull) + label_color)

        leaf_data[int(label)] = {
            'length': leaf_length,
            'width': leaf_width,
            'princ_0': leaf_princ_0,
            'prince_1': leaf_princ_1,
            'loc': leaf_location
        }

    full_proj_points = np.concatenate(full_proj_points)
    full_proj_colors = np.concatenate(full_proj_colors)
    hull_points = np.concatenate(hull_points)
    hull_colors = np.concatenate(hull_colors)

    mean_offset = full_proj_points.mean(axis=0)
    #full_proj_points -= mean_offset
    #full_proj_points -= mean_offset

    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(full_proj_points)
    pc.colors = open3d.utility.Vector3dVector(full_proj_colors)
    cloud_path = os.path.join(output_dir, filename.replace('.txt', '_proj.pcd'))
    open3d.io.write_point_cloud(cloud_path, pc)

    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(hull_points)
    pc.colors = open3d.utility.Vector3dVector(hull_colors)
    cloud_path = os.path.join(output_dir, filename.replace('.txt', '_hull.pcd'))
    open3d.io.write_point_cloud(cloud_path, pc)

    for label in leaf_data:
        leaf_data[label]['loc'] = (leaf_data[label]['loc'] - mean_offset).tolist()

    info_path = os.path.join(output_dir, filename.replace('.txt', '_info.json'))
    write_json(info_path, leaf_data)

    print('Done ' + filename)
    
def run(input_dir, output_dir, num_colors, type):
    colors = np.array(distinctipy.get_colors(num_colors))

    output_dir = os.path.join(output_dir, type)

    for dirname in os.listdir(input_dir):
        if not type in dirname.lower():
            continue

        subdir = os.path.join(input_dir, dirname)
        if not os.path.isdir(subdir):
            continue

        sub_output_dir = os.path.join(output_dir, dirname)
        if not os.path.exists(sub_output_dir):
            os.mkdir(sub_output_dir)

        for filename in os.listdir(subdir):

            if not filename.endswith('.txt'):
                continue

            if not '_a' in filename:
                continue
        
            process_file(filename, subdir, sub_output_dir, colors)

    

INPUT_DIR = '/home/frc-ag-3/Downloads/Pheno4D'
OUTPUT_DIR = 'labelling/pheno'
TYPE = 'tomato'
DISNCT_COLORS = 50
if __name__ == '__main__':
    run(INPUT_DIR, OUTPUT_DIR, DISNCT_COLORS, TYPE)