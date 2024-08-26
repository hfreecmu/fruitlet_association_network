import os
import pickle
import numpy as np
import open3d
import json
import cv2
import distinctipy

# MINS [-0.04033629 -0.04759627 -0.25860758]
# MAXS [0.05186419 0.04141762 0.32618338]
# MEANS [-2.74611645e-18 -2.52243434e-17  1.65013076e-18]
# STDS [0.0127603  0.00885309 0.01517708]
# 1460


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

pointcloud_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/point_clouds'
annotations_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/id_annotations'
image_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/selected_images/images'
det_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/detections'
output_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/vis_point_clouds/cluster_full'
anno_output_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/id_annotations_filtered'

num_total_fruitlets = 0
num_processed_fruitlets = 0
filtered_set = set()
skipped_set = set()
never_had_set = set()
max_size = 0

full_pos = []
for filename in os.listdir(annotations_dir):
    if not filename.endswith('.json'):
        continue 

    # if not '2021_15_0_left' in filename:
    #     continue

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
    tracked_seg_inds = []

    for (fruitlet_cloud, det, seg_inds) in zip(seg_clouds, annotations, segmentations):
        if det['fruitlet_id'] < 0:
            continue

        num_total_fruitlets += 1

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
        
        # TODO bug here cause not checking cloud but things are working
        # and don't want to naively change without esting
        if fruitlet_cloud.shape[0] == 0:
            skipped_set.add(filename)
            # set as removed which we will use as -1
            # if we do augmenting later have to think about this
            det['fruitlet_id'] = -1
            continue

        #cloud = cloud.voxel_down_sample(voxel_size=0.0005)

        # if fruitlet_cloud.shape[0] == 0:
        #     filtered_set.add(filename)
        #     # set as removed which we will use as -1
        #     # if we do augmenting later have to think about this
        #     det['fruitlet_id'] = -1
        #     continue

            
        cloud, radius_inds = cloud.remove_radius_outlier(nb_points=20, radius=0.002)
        # seg_inds = seg_inds[radius_inds]
        
        if np.array(cloud.points).shape[0] == 0:
            filtered_set.add(filename)
            # set as removed which we will use as -1
            # if we do augmenting later have to think about this
            det['fruitlet_id'] = -1
            continue
        
        fruitlet_cloud = np.array(cloud.points)
        colors = np.array(cloud.colors)

        med_vals = np.median(fruitlet_cloud, axis=0)
        dists = np.linalg.norm(fruitlet_cloud - med_vals, axis=1)
        good_inds = dists < 0.02
        fruitlet_cloud = fruitlet_cloud[good_inds]
        colors = colors[good_inds]
        # seg_inds = seg_inds[good_inds]
        
        if fruitlet_cloud.shape[0] < 50:
            filtered_set.add(filename)
            # set as removed which we will use as -1
            # if we do augmenting later have to think about this
            det['fruitlet_id'] = -1
            continue

        # mean_vals = fruitlet_cloud.mean(axis=0)
        # fruitlet_cloud_to_procc = fruitlet_cloud - mean_vals
        # det['cloud_points'] = fruitlet_cloud_to_procc.tolist()
        # det['seg_inds'] = seg_inds.tolist()

        cluster_cloud.append(fruitlet_cloud)
        cluster_colors.append(colors)
        tracked_dets.append(det)
        tracked_seg_inds.append(seg_inds)

        num_processed_fruitlets += 1
        max_size = np.max([max_size, fruitlet_cloud.shape[0]])

    anno_output_path = os.path.join(anno_output_dir, filename)
    write_json(anno_output_path, full_annotations)

    orig_cloud = np.concatenate(orig_cloud)
    orig_colors = np.concatenate(orig_colors)
    vis_path = os.path.join(output_dir, filename.replace('.json', '_orig.pcd'))
    
    if len(orig_cloud) == 0 and filename not in skipped_set:
        never_had_set.add(filename)
        continue
    elif len(orig_cloud) == 0:
        continue

    create_point_cloud(vis_path, orig_cloud, orig_colors)

    if len(cluster_cloud) == 0:
        if not (filename in skipped_set or filename in filtered_set):
            raise RuntimeError('Should not happen')
        continue
        #breakpoint()

    cluster_cloud_backup = cluster_cloud
    cluster_cloud = np.concatenate(cluster_cloud)
    cluster_colors = np.concatenate(cluster_colors)

    if cluster_cloud.shape[0] == 0:
        raise RuntimeError('should not happen?')
        #breakpoint()
    
    # TODO should I use mean? I think so here
    mean_vals = cluster_cloud.mean(axis=0)
    for det_ind in range(len(tracked_dets)):
        det = tracked_dets[det_ind]
        fruitlet_cloud = cluster_cloud_backup[det_ind]
        seg_inds = tracked_seg_inds[det_ind]

        centered_fruitlet_cloud = fruitlet_cloud - mean_vals
        det['cloud_points'] = centered_fruitlet_cloud.tolist()
        det['seg_inds'] = seg_inds.tolist()

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

# max_abs = np.max(np.abs(full_pos))
# mins = -max_abs
# maxs = max_abs
# mins = full_pos.min(axis=0)
# maxs = full_pos.max(axis=0) 
# normed_pos = (full_pos - mins)/(maxs - mins)

# means = np.mean(normed_pos, axis=0)
# stds = np.std(normed_pos, axis=0)

print('MINS', mins)
print('MAXS', maxs)
print('MEANS', means)
print('STDS', stds)

# print(num_total_fruitlets, num_processed_fruitlets, num_total_fruitlets - num_processed_fruitlets)
print(max_size)
print('never had set: ')
print(never_had_set)
print('skipped set: ')
print(skipped_set)
print('filtered set: ')
print(filtered_set)