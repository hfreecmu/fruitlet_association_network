import os
import cv2
import numpy as np
import distinctipy
import open3d

import sys
sys.path.append('/home/hfreeman/harry_ws/repos/fruitlet_association_network')
from util.util import get_identifier, read_json

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

# TODO may have to fix cloud when add bounding box

def match_anno(fk_0, img_0, anno_0, fk_1, img_1, anno_1):
    anno_0_map = {}
    cloud_0 = []
    cloud_0_colors = []
    for ind_0, det in enumerate(anno_0):
        cloud_0.append(np.zeros((0, 3)))
        cloud_0_colors.append(np.zeros((0, 3)))

        if det['fruitlet_id'] < 0:
            continue

        if det['fruitlet_id'] in anno_0_map:
            raise RuntimeError('something wrong with ' + fk_0)

        anno_0_map[det['fruitlet_id']] = ind_0
        cloud_0[-1] = np.array(det['cloud_points'])
        cloud_0_colors[-1] = np.ones_like(cloud_0[-1])

    anno_1_map = {}
    cloud_1 = []
    cloud_1_colors = []
    for ind_1, det in enumerate(anno_1):
        cloud_1.append(np.zeros((0, 3)))
        cloud_1_colors.append(np.zeros((0, 3)))

        if det['fruitlet_id'] < 0:
            continue

        if det['fruitlet_id'] in anno_1_map:
            raise RuntimeError('something wrong with ' + fk_1)
        
        anno_1_map[det['fruitlet_id']] = ind_1
        cloud_1[-1] = np.array(det['cloud_points'])
        cloud_1_colors[-1] = np.ones_like(cloud_1[-1])

    cs = []
    for key in anno_0_map:
        if not key in anno_1_map:
            continue

        ind_0 = anno_0_map[key]
        ind_1 = anno_1_map[key]

        det_0 = anno_0[ind_0]
        det_1 = anno_1[ind_1]

        cx_0 = int((det_0['x0'] + det_0['x1']) / 2)
        cy_0 = int((det_0['y0'] + det_0['y1']) / 2)

        cx_1 = int((det_1['x0'] + det_1['x1']) / 2) + img_0.shape[1]
        cy_1 = int((det_1['y0'] + det_1['y1']) / 2)

        cs.append([ind_0, ind_1, cx_0, cy_0, cx_1, cy_1])
    
    comb_img = np.concatenate((img_0, img_1), axis=1)
    num_matches = len(cs)
    colors = distinctipy.get_colors(num_matches)

    for ind in range(len(cs)):
        ind_0, ind_1, cx_0, cy_0, cx_1, cy_1 = cs[ind]
        color = ([int(255*colors[ind][0]), int(255*colors[ind][1]), int(255*colors[ind][2])])

        cv2.line(comb_img, (cx_0, cy_0), (cx_1, cy_1), color, thickness=2)

        cloud_0_colors[ind_0][:] = colors[ind][0:3]
        cloud_1_colors[ind_1][:] = colors[ind][0:3]

    cloud_0 = np.vstack(cloud_0)
    cloud_0_colors = np.vstack(cloud_0_colors)

    cloud_1 = np.vstack(cloud_1)
    cloud_1_colors = np.vstack(cloud_1_colors)
    
    return comb_img, cloud_0, cloud_0_colors, cloud_1, cloud_1_colors

annotations_filtered_dir = 'labelling/id_annotations_filtered'
images_dir = 'labelling/selected_images/images'
vis_dir = 'labelling/vis_annotations/images'
cloud_dir = 'labelling/vis_annotations/clouds'
num_sample = 10

cluster_dict = {}
for filename in os.listdir(annotations_filtered_dir):
    if not filename.endswith('.json'):
        continue

    file_key = filename.split('.json')[0]

    annotations_path = os.path.join(annotations_filtered_dir, filename)
    image_path = os.path.join(images_dir, file_key + '.png')

    identifier = get_identifier(file_key)
    if not identifier in cluster_dict:
        cluster_dict[identifier] = []

    entry = (file_key, annotations_path, image_path)
    cluster_dict[identifier].append(entry)

entry_pairs = []
for identifier in cluster_dict:
    entries = cluster_dict[identifier]
    num_entries = len(entries)

    for ind_0 in range(num_entries):
        entry_0 = entries[ind_0]

        for ind_1 in range(ind_0 + 1, num_entries):
            entry_1 = entries[ind_1]

            entry_pairs.append([entry_0, entry_1])


rand_inds = np.random.choice(len(entry_pairs), size=num_sample, replace=False)
for rand_ind in rand_inds:
    entry_0, entry_1 = entry_pairs[rand_ind]

    fk_0 = entry_0[0]
    anno_0 = read_json(entry_0[1])['annotations']
    img_0 = cv2.imread(entry_0[2])

    fk_1 = entry_1[0]
    anno_1 = read_json(entry_1[1])['annotations']
    img_1 = cv2.imread(entry_1[2])

    comb_img, cloud_0, cloud_0_colors, cloud_1, cloud_1_colors = match_anno(fk_0, img_0, anno_0, fk_1, img_1, anno_1)

    output_path = os.path.join(vis_dir, '_'.join([fk_0, fk_1]) + '.png')
    cv2.imwrite(output_path, comb_img)

    cloud_path_0 = os.path.join(cloud_dir, '_'.join([fk_0, fk_1]) + '_0.pcd')
    cloud_path_1 = os.path.join(cloud_dir, '_'.join([fk_0, fk_1]) + '_1.pcd')

    create_point_cloud(cloud_path_0, cloud_0, cloud_0_colors)
    create_point_cloud(cloud_path_1, cloud_1, cloud_1_colors)






