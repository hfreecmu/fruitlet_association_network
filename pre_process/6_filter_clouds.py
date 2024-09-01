import os
import pickle
import numpy as np
import open3d
import json
import cv2
import distinctipy

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def write_json(path, data, pretty=False):
    with open(path, 'w') as f:
        if not pretty:
            json.dump(data, f)
        else:
            json.dump(data, f, indent=4)

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

pointcloud_dir = 'labelling/point_clouds'
annotations_dir = 'labelling/id_annotations'
image_dir = 'labelling/selected_images/images'
det_dir = 'labelling/detections'
output_dir = 'labelling/vis_points_clouds_filtered'
anno_output_dir = 'labelling/id_annotations_filtered'

full_pos = []
for filename in os.listdir(annotations_dir):
    if not filename.endswith('.json'):
        continue 

    anno_path = os.path.join(annotations_dir, filename)
    pc_path = os.path.join(pointcloud_dir, filename.replace('.json', '.pkl'))
    image_path = os.path.join(image_dir, filename.replace('.json', '.png'))
    det_path = os.path.join(det_dir, filename.replace('.json', '.pkl'))

    full_annotations = read_json(anno_path)
    annotations = full_annotations['annotations']
    seg_clouds = read_pickle(pc_path)
    image = cv2.imread(image_path)
    segmentations = read_pickle(det_path)['segmentations']

    cluster_cloud = []
    cluster_colors = []
    orig_cloud = []
    orig_colors = []
    tracked_dets = []

    for (fruitlet_cloud, det, seg_inds) in zip(seg_clouds, annotations, segmentations):
        if det['fruitlet_id'] < 0:
            continue

        det['seg_inds'] = seg_inds.tolist()

        colors = image[seg_inds[:, 0], seg_inds[:, 1]]

        nan_inds = np.isnan(fruitlet_cloud).any(axis=1)
        fruitlet_cloud = fruitlet_cloud[~nan_inds]
        colors = colors[~nan_inds].astype(float) / 255

        orig_cloud.append(fruitlet_cloud)
        orig_colors.append(colors)

        cloud = open3d.geometry.PointCloud()
        cloud.points = open3d.utility.Vector3dVector(fruitlet_cloud)
        cloud.colors = open3d.utility.Vector3dVector(colors)

        cloud = cloud.voxel_down_sample(voxel_size=0.0005)
        
        # bug here cause not checking cloud but things are working
        # and don't want to naively change without testing
        # by cloud I mean the voxel down result but empty should be same?
        if fruitlet_cloud.shape[0] == 0:
            det['cloud_points'] = []
            continue
            
        cloud, radius_inds = cloud.remove_radius_outlier(nb_points=20, radius=0.002)
        
        if np.array(cloud.points).shape[0] == 0:
            det['cloud_points'] = []
            continue
        
        fruitlet_cloud = np.array(cloud.points)
        colors = np.array(cloud.colors)

        med_vals = np.median(fruitlet_cloud, axis=0)
        dists = np.linalg.norm(fruitlet_cloud - med_vals, axis=1)
        good_inds = dists < 0.02
        fruitlet_cloud = fruitlet_cloud[good_inds]
        colors = colors[good_inds]
        
        if fruitlet_cloud.shape[0] < 50:
            det['cloud_points'] = []
            continue

        cluster_cloud.append(fruitlet_cloud)
        cluster_colors.append(colors)
        tracked_dets.append(det)

    if len(orig_cloud) == 0:
        # happens when flagged and I did not labbel
        assert full_annotations['flagged'] == True
        continue

    anno_output_path = os.path.join(anno_output_dir, filename)
    write_json(anno_output_path, full_annotations)

    orig_cloud = np.concatenate(orig_cloud)
    orig_colors = np.concatenate(orig_colors)
    vis_path = os.path.join(output_dir, filename.replace('.json', '_orig.pcd'))

    create_point_cloud(vis_path, orig_cloud, orig_colors)

    cluster_cloud_backup = cluster_cloud
    cluster_cloud = np.concatenate(cluster_cloud)
    cluster_colors = np.concatenate(cluster_colors)

    if cluster_cloud.shape[0] == 0:
        raise RuntimeError('should not happen?')
    
    mean_vals = cluster_cloud.mean(axis=0)
    for det_ind in range(len(tracked_dets)):
        det = tracked_dets[det_ind]
        fruitlet_cloud = cluster_cloud_backup[det_ind]

        centered_fruitlet_cloud = fruitlet_cloud - mean_vals
        det['cloud_points'] = centered_fruitlet_cloud.tolist()

    # I do want to do this
    anno_output_path = os.path.join(anno_output_dir, filename)
    write_json(anno_output_path, full_annotations)

    vis_path = os.path.join(output_dir, filename.replace('.json', '.pcd'))
    create_point_cloud(vis_path, cluster_cloud, cluster_colors)

    cluster_cloud_to_procc = cluster_cloud - mean_vals
    full_pos.append(cluster_cloud_to_procc)
    vis_path = os.path.join(output_dir, filename.replace('.json', '_cent.pcd'))
    create_point_cloud(vis_path, cluster_cloud_to_procc, cluster_colors)

full_pos = np.concatenate(full_pos)
mins = full_pos.min(axis=0)
maxs = full_pos.max(axis=0)
means = np.mean(full_pos, axis=0)
stds = np.std(full_pos, axis=0)

print('MINS', mins)
print('MAXS', maxs)
print('MEANS', means)
print('STDS', stds)