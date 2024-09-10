import os
import torch
import numpy as np
from PIL import Image
import json
import yaml
import cv2
import pickle
import distinctipy
import open3d

import sys
sys.path.append('/home/hfreeman/harry_ws/repos/RAFT-Stereo')
sys.path.append('/home/hfreeman/harry_ws/repos/RAFT-Stereo/core')

from raft_stereo import RAFTStereo
from utils.utils import InputPadder

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def read_yaml(path):
    with open(path, 'r') as f:
        yaml_to_read = yaml.safe_load(f)

    return yaml_to_read

def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def write_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

class EmptyClass:
    def __init__(self):
        pass

def get_middle_model_args(restore_ckpt):
    args = EmptyClass()
    args.restore_ckpt = restore_ckpt
    args.corr_implementation = 'alt'
    args.mixed_precision = True

    args.shared_backbone = False
    args.n_downsample = 2
    args.n_gru_layers = 3
    args.slow_fast_gru = False
    args.valid_iters = 32
    args.hidden_dims = [128]*3
    args.corr_levels = 4
    args.corr_radius = 4
    args.context_norm = 'batch'

    return args 

def load_raft_model(args, device):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(device)
    model.eval()

    return model

def load_raft_image(imfile, device):
    pil_im = Image.open(imfile)

    img = np.array(pil_im).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

def extract_raft_disparity(model, left_im_path, right_im_path, valid_iters, device):
    image1 = load_raft_image(left_im_path, device)
    image2 = load_raft_image(right_im_path, device)

    with torch.no_grad():
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        _, flow_up = model(image1, image2, iters=valid_iters, test_mode=True)
        flow_up = padder.unpad(flow_up).squeeze()

    disparity = -flow_up.cpu().numpy().squeeze()
    return disparity

def get_camera_info_path(camera_info_dir, image_path):
    backwards_mapping = read_json(BACKWARDS_MAPPING_PATH)

    image_filename = os.path.basename(image_path)
    orig_filepath = backwards_mapping[image_filename]

    dir_identifier = os.path.basename(os.path.dirname(os.path.dirname(orig_filepath)))

    year, month, day, hour, _, _ = dir_identifier.split('_')[1].split('-')
    # early morning does to day before
    # since all days have 2 digits do not need to worry about str
    if (int(hour) < 4):
        day = str(int(day) - 1)
    
    camera_info_filename = '_'.join([year, month, day]) + '.yml'

    # robot images are in sub_dir
    camera_info_dir_to_use = camera_info_dir
    if 'field_data' in orig_filepath:
        camera_info_dir_to_use = os.path.join(camera_info_dir, 'robot_info') 

    camera_info_path = os.path.join(camera_info_dir_to_use, camera_info_filename)

    return camera_info_path

def get_intrinsics(camera_info, include_fy=False):
    P = camera_info['P']
    f_norm = P[0]
    f_norm_y = P[5]
    baseline = P[3] / P[0]
    cx = P[2]
    cy = P[6]

    if not include_fy:
        intrinsics = (baseline, f_norm, cx, cy)
    else:
        intrinsics = (baseline, f_norm, cx, cy, f_norm_y)

    return intrinsics

def bilateral_filter(disparity, intrinsics, 
                    #  bilateral_d=9, bilateral_sc=0.03, bilateral_ss=4.5):
                    bilateral_d=15, bilateral_sc=1, bilateral_ss=10):
    # baseline, f_norm, _, _ = intrinsics
    # stub = -baseline / disparity
    # z = stub * f_norm
    # z_new = cv2.bilateralFilter(z, bilateral_d, bilateral_sc, bilateral_ss)

    # stub_new = z_new / f_norm
    # disparity_new = -baseline / stub_new

    # on disparity image directly was much better
    disparity_new = cv2.bilateralFilter(disparity, bilateral_d, bilateral_sc, bilateral_ss)

    return disparity_new

def compute_points(disparity, intrinsics):
    baseline, f_norm, cx, cy = intrinsics
    stub = -baseline / disparity

    x_pts, y_pts = np.meshgrid(np.arange(disparity.shape[1]), np.arange(disparity.shape[0]))

    x = stub * (x_pts - cx)
    y = stub * (y_pts - cy)
    z = stub*f_norm

    points = np.stack((x, y, z), axis=2)

    return points

def extract_depth_discontuinities(disparity, intrinsics,
                                  disc_use_rat=True,
                                  disc_rat_thresh=0.01,
                                  disc_dist_thresh=0.001):
    baseline, f_norm, _, _ = intrinsics
    stub = -baseline / disparity
    z = stub * f_norm

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.dilate(z, element)
    erosion = cv2.erode(z, element)

    dilation -= z
    erosion = z - erosion

    max_image = np.max((dilation, erosion), axis=0)

    if disc_use_rat:
        ratio_image = max_image / z
        _, discontinuity_map = cv2.threshold(ratio_image, disc_rat_thresh, 1.0, cv2.THRESH_BINARY)
    else:
        _, discontinuity_map = cv2.threshold(max_image, disc_dist_thresh, 1.0, cv2.THRESH_BINARY)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    discontinuity_map = cv2.morphologyEx(discontinuity_map, cv2.MORPH_CLOSE, element)

    return discontinuity_map

def extract_point_cloud(left_path, disparity, camera_info_path,
                        should_bilateral_filter=True, 
                        should_depth_discon_filter=True,
                        should_distance_filter=True,
                        dist_thresh = 1.0):
    
    camera_info = read_yaml(camera_info_path)
    intrinsics = get_intrinsics(camera_info)

    pil_im = Image.open(left_path)
    im = np.array(pil_im).astype(np.uint8)

    if should_bilateral_filter:
        disparity = bilateral_filter(disparity, intrinsics)

    points = compute_points(disparity, intrinsics)
    colors = im.astype(float) / 255

    inf_inds = np.where(disparity <= 0)
    points[inf_inds] = np.nan
    colors[inf_inds] = np.nan

    if should_depth_discon_filter:
        discontinuity_map = extract_depth_discontuinities(disparity, intrinsics)
        discon_inds = np.where(discontinuity_map > 0) 
        points[discon_inds] = np.nan
        colors[discon_inds] = np.nan

    if should_distance_filter:
        dist_inds = np.where(np.linalg.norm(points, axis=2) > dist_thresh)
        points[dist_inds] = np.nan
        colors[dist_inds] = np.nan

    return points, colors

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

BACKWARDS_MAPPING_PATH = 'labelling/selected_images/backward_mappings.json'

anno_dir = 'labelling/id_annotations'
detections_dir = 'labelling/detections'
image_dir = 'labelling/selected_images/images'
raft_middleburry_path = 'trained_models/raftstereo-middlebury.pth'
camera_info_dir = 'camera_info'
disparity_dir = 'labelling/disparity'
output_dir = 'labelling/point_clouds'
full_cloud_dir = 'labelling/full_point_clouds'
device = 'cuda'

vis = True
num_vis = 20
vis_dir = 'labelling/vis_point_clouds'

if __name__ == "__main__":

    raft_args = get_middle_model_args(raft_middleburry_path)
    raft_model = load_raft_model(raft_args, device)

    for filename in os.listdir(anno_dir):
        if not filename.endswith('.json'):
            continue

        if not 'left' in filename:
            continue

        detections_path = os.path.join(detections_dir, filename.replace('.json', '.pkl'))
        left_image_path = os.path.join(image_dir, filename.replace('.json', '.png'))
        right_image_path = left_image_path.replace('left', 'right')

        output_path = os.path.join(output_dir, filename.replace('.json', '.pkl'))
        full_cloud_path = os.path.join(full_cloud_dir, filename.replace('.json', '.pcd'))
        if os.path.exists(output_path):
            print('Skipping: ', filename)
            continue

        disparity_path = os.path.join(disparity_dir, filename.replace('.json', '.npy'))
        if os.path.exists(disparity_path):
            disparity = np.load(disparity_path)
        else:
            disparity = extract_raft_disparity(raft_model, left_image_path, right_image_path,
                                               raft_args.valid_iters, device)
            np.save(disparity_path, disparity)
        
        camera_info_path = get_camera_info_path(camera_info_dir, left_image_path)

        points, colors = extract_point_cloud(left_image_path, disparity, camera_info_path)

        segmentations = read_pickle(detections_path)['segmentations']

        cloud_points = []
        for ind in range(len(segmentations)):
            seg_inds = segmentations[ind]

            fruitlet_cloud_points = points[seg_inds[:, 0], seg_inds[:, 1]]
            # for now leaving nans because why not
            cloud_points.append(fruitlet_cloud_points)

        ### now for full cloud here
        nan_inds = np.isnan(points).any(axis=2)
        full_cloud_points = points[~nan_inds]
        full_cloud_colors = colors[~nan_inds]
        ###

        write_pickle(output_path, cloud_points)
        create_point_cloud(full_cloud_path, full_cloud_points, full_cloud_colors)

        if vis and num_vis > 0:
            num_vis -= 1

            num_fruitlets = len(cloud_points)
            colors = distinctipy.get_colors(num_fruitlets)

            full_cloud = []
            full_colors = []

            for ind in range(num_fruitlets):
                fruitlet_cloud_points = cloud_points[ind]

                # flatten and make not nan
                nan_inds = np.isnan(fruitlet_cloud_points).any(axis=1)
                fruitlet_cloud_points = fruitlet_cloud_points[~nan_inds]
                
                
                fruitlet_color = np.zeros_like(fruitlet_cloud_points) + colors[ind]

                full_cloud.append(fruitlet_cloud_points)
                full_colors.append(fruitlet_color)
            

            full_cloud = np.vstack(full_cloud)
            full_colors = np.vstack(full_colors)

            vis_path = os.path.join(vis_dir, filename.replace('.json', '.pcd'))
            create_point_cloud(vis_path, full_cloud, full_colors)


        print('Done: ', filename)


