import os
import json
import pickle
from omegaconf import OmegaConf
import cv2
import numpy as np
import open3d

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
    
def read_point_cloud(path):
    cloud = open3d.io.read_point_cloud(path)
    points = np.array(cloud.points)
    return points

def load_cfg(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    loss_cfg_path = cfg['loss_path']
    loss_cfg = OmegaConf.load(loss_cfg_path)

    cfg['loss_params'] = loss_cfg

    return cfg

def get_identifier(file_key):
    identifier = '_'.join(file_key.split('_')[0:2])

    return identifier

def get_checkpoint_path(checkpoint_dir, exp_name, checkpoint_metrics):
    checkpoint_dir = os.path.join(checkpoint_dir, exp_name, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        raise RuntimeError('checkpoint dir dne: ' + checkpoint_dir)
    
    metric_name = checkpoint_metrics['metric_type']
    is_min = checkpoint_metrics['is_min']

    best_filename = None
    best_val = None
    for filename in os.listdir(checkpoint_dir):
        if not filename.endswith('.ckpt'):
            continue

        metric_identifier = filename.split('-')[1].split('=')[0]
        if not metric_identifier == metric_name:
            continue

        metric_val = float(filename.split('=')[-1].split('.ckpt')[0])

        if best_val is None:
            best_filename = filename
            best_val = metric_val
        elif is_min and metric_val < best_val:
            best_filename = filename
            best_val = metric_val
        elif not is_min and metric_val > best_val:
            best_filename = filename
            best_val = metric_val

    if best_filename is None:
        raise RuntimeError('No valid checkpoint found in: ' + checkpoint_dir)
    
    checkpoint_path = os.path.join(checkpoint_dir, best_filename)
    return checkpoint_path

def unravel_clouds(clouds, cloud_inds, fruitlet_ids):
    fruitlet_clouds = []
    centroids = []
    has_points = []
    for fruitlet_id in fruitlet_ids:
        cloud_points = clouds[cloud_inds == fruitlet_id]
        if cloud_points.shape[0] == 0:
            has_points.append(False)
        else:
            has_points.append(True)

        fruitlet_clouds.append(cloud_points)

        centroids.append(cloud_points.mean(axis=0).numpy())
    
    centroids = np.array(centroids)
    has_points = np.array(has_points)
    
    return fruitlet_clouds, centroids, has_points

def vis_matches(matches, gt_matches, 
                im_path_0, anno_path_0, det_inds_0, 
                im_path_1, anno_path_1, det_inds_1,
                output_dir):
    im_0 = cv2.imread(im_path_0)
    im_1 = cv2.imread(im_path_1)

    anno_0 = read_json(anno_path_0)['annotations']
    anno_1 = read_json(anno_path_1)['annotations']

    file_key_0 = os.path.basename(anno_path_0).split('.json')[0]
    file_key_1 = os.path.basename(anno_path_1).split('.json')[0]
    file_id = '_'.join([file_key_0, file_key_1])

    match_inds = np.argwhere(matches == 1.0)
    prec_img = np.concatenate([im_0, im_1], axis=1)
    for m_ind_0, m_ind_1 in match_inds:
        ind_0 = det_inds_0[m_ind_0]
        ind_1 = det_inds_1[m_ind_1]

        cx_0 = int((anno_0[ind_0]['x0'] + anno_0[ind_0]['x1']) / 2)
        cy_0 = int((anno_0[ind_0]['y0'] + anno_0[ind_0]['y1']) / 2)

        cx_1 = int((anno_1[ind_1]['x0'] + anno_1[ind_1]['x1']) / 2) + im_0.shape[1]
        cy_1 = int((anno_1[ind_1]['y0'] + anno_1[ind_1]['y1']) / 2)

        if gt_matches[m_ind_0, m_ind_1] == 1.0:
            color = [0, 255, 0]
        else:
            color = [0, 0, 255]

        cv2.line(prec_img, (cx_0, cy_0), (cx_1, cy_1), color, thickness=2)

    gt_match_inds = np.argwhere(gt_matches == 1.0)
    rec_img = np.concatenate([im_0, im_1], axis=1)
    for m_ind_0, m_ind_1 in gt_match_inds:
        ind_0 = det_inds_0[m_ind_0]
        ind_1 = det_inds_1[m_ind_1]

        cx_0 = int((anno_0[ind_0]['x0'] + anno_0[ind_0]['x1']) / 2)
        cy_0 = int((anno_0[ind_0]['y0'] + anno_0[ind_0]['y1']) / 2)

        cx_1 = int((anno_1[ind_1]['x0'] + anno_1[ind_1]['x1']) / 2) + im_0.shape[1]
        cy_1 = int((anno_1[ind_1]['y0'] + anno_1[ind_1]['y1']) / 2)

        if matches[m_ind_0, m_ind_1] == 1.0:
            color = [0, 255, 0]
        else:
            color = [255, 0, 255]

        cv2.line(rec_img, (cx_0, cy_0), (cx_1, cy_1), color, thickness=2)

    comb_im = np.concatenate([prec_img, rec_img], axis=0)
        
    res_path = os.path.join(output_dir, file_id + '.png')
    cv2.imwrite(res_path, comb_im)
