import sys
sys.path.append('/home/frc-ag-3/harry_ws/fruitlet/repos/RAFT-Stereo')
sys.path.append('/home/frc-ag-3/harry_ws/fruitlet/repos/RAFT-Stereo/core')

import os
import numpy as np
import shutil
import torch
from PIL import Image
import yaml
import cv2
import json
import open3d
import torch.nn.functional as F

from raft_stereo import RAFTStereo
from utils.utils import InputPadder

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

class EmptyClass:
    def __init__(self):
        pass

def get_irvc_model_args(restore_ckpt):
    args = EmptyClass()
    args.restore_ckpt = restore_ckpt
    args.context_norm = 'instance'

    args.shared_backbone = False
    args.n_downsample = 2
    args.n_gru_layers = 3
    args.slow_fast_gru = False
    args.valid_iters = 32
    args.corr_implementation = 'reg'
    args.mixed_precision = False
    args.hidden_dims = [128]*3
    args.corr_levels = 4
    args.corr_radius = 4

    return args

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

def load_raft_image(imfile, resize_dims, device):
    pil_im = Image.open(imfile)

    if resize_dims is not None:
        pil_im = pil_im.resize(resize_dims)

    img = np.array(pil_im).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

def extract_raft_disparity(model, left_im_path, right_im_path, resize_dims, valid_iters, device):
    image1 = load_raft_image(left_im_path, resize_dims, device)
    image2 = load_raft_image(right_im_path, resize_dims, device)

    with torch.no_grad():
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        _, flow_up = model(image1, image2, iters=valid_iters, test_mode=True)
        flow_up = padder.unpad(flow_up).squeeze()

    disparity = -flow_up.cpu().numpy().squeeze()
    return disparity

def extract_disparities(image_dir, output_dir, raft_details, resize_dims, device):
    raft_arg_func, raft_model_path = raft_details

    raft_args = raft_arg_func(raft_model_path)
    raft_model = load_raft_model(raft_args, device)

    for filename in os.listdir(image_dir):
        if not filename.endswith('.png'):
            continue

        if not (('left' in filename) or ('LEFT' in filename)):
            continue

        left_im_path = os.path.join(image_dir, filename)
        right_im_path = left_im_path.replace('left', 'right').replace('LEFT', 'RIGHT')

        disparity = extract_raft_disparity(raft_model, left_im_path, right_im_path,
                                           resize_dims, raft_args.valid_iters, device)
        
        output_path = os.path.join(output_dir, filename.replace('.png', '.npy'))
        np.save(output_path, disparity)

def copy_random_pairs(input_dir, output_dir, num=10):
    filenames = []
    for filename in os.listdir(input_dir):
        if not filename.endswith('.png'):
            continue

        if not (('left' in filename) or ('LEFT' in filename)):
            continue

        filenames.append(filename)

    rand_inds = np.random.choice(len(filenames), size=num, replace=False)

    for ind in rand_inds:
        left_src = os.path.join(input_dir, filenames[ind])
        left_dst = os.path.join(output_dir, filenames[ind])

        right_src = left_src.replace('left', 'right').replace('LEFT', 'RIGHT')
        right_dst = left_dst.replace('left', 'right').replace('LEFT', 'RIGHT')

        shutil.copyfile(left_src, left_dst)
        shutil.copyfile(right_src, right_dst)

def read_yaml(path):
    with open(path, 'r') as f:
        yaml_to_read = yaml.safe_load(f)

    return yaml_to_read

def get_intrinsics(camera_info, resize_dims, include_fy=False):
    P = camera_info['P']
    f_norm = P[0]
    f_norm_y = P[5]
    baseline = P[3] / P[0]
    cx = P[2]
    cy = P[6]

    if resize_dims is not None:
        scale_x = resize_dims[0] / ORIG_DIMS[0]
        scale_y = resize_dims[1] / ORIG_DIMS[1]

        if scale_x != scale_y:
            raise RuntimeError('scales must be same')
    
        f_norm *= scale_x
        f_norm_y *= scale_y
        cx *= scale_x
        cy *= scale_y
    
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

def extract_point_cloud(left_path, disparity_path, camera_info_path,
                        resize_dims,
                        should_bilateral_filter=True, 
                        should_depth_discon_filter=True,
                        should_distance_filter=True,
                        dist_thresh = 1.0):
    
    camera_info = read_yaml(camera_info_path)
    intrinsics = get_intrinsics(camera_info, resize_dims)

    disparity = np.load(disparity_path)
    #inf_inds = np.where(disparity <= 0)

    #disparity[inf_inds] = 1e-6

    pil_im = Image.open(left_path)
    if resize_dims is not None:
        pil_im = pil_im.resize(resize_dims)
    im = np.array(pil_im).astype(np.uint8) 

    if should_bilateral_filter:
        disparity = bilateral_filter(disparity, intrinsics)

    points = compute_points(disparity, intrinsics)
    colors = im.astype(float) / 255

    #points[inf_inds] = np.nan
    #colors[inf_inds] = np.nan

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

def extract_point_clouds(disp_dir, image_dir, camera_info_dir, resize_dims,
                         output_dir):
    backwards_mapping = read_json(BACKWARDS_MAPPING_PATH)
    for filename in os.listdir(disp_dir):
        if not filename.endswith('.npy'):
            continue

        disp_path = os.path.join(disp_dir, filename)
        im_path = os.path.join(image_dir, filename.replace('.npy', '.png'))

        # get camera info path
        im_filename = os.path.basename(im_path)
        orig_filepath = backwards_mapping[im_filename]

        dir_identifier = os.path.basename(os.path.dirname(os.path.dirname(orig_filepath)))

        have_data = False
        if '2021' in orig_filepath:
            have_data = True
        elif 'in-hand_images' in orig_filepath:
            have_data = True
        
        if not have_data:
            continue

        year, month, day, _, _, _ = dir_identifier.split('_')[1].split('-')
        camera_info_filename = '_'.join([year, month, day]) + '.yml'
        camera_info_path = os.path.join(camera_info_dir, camera_info_filename)
        
        # end get camera info path

        points, colors = extract_point_cloud(im_path, disp_path, camera_info_path,
                                             resize_dims)

        points = points.reshape((-1, 3))
        colors = colors.reshape((-1, 3))

        nan_inds = np.isnan(points).any(axis=1)

        points = points[~nan_inds]
        colors = colors[~nan_inds]

        output_path = os.path.join(output_dir, filename.replace('.npy', '.pcd'))
        create_point_cloud(output_path, points, colors)

raft_middleburry_path = '/home/frc-ag-3/harry_ws/viewpoint_planning/segment_exp/src/fruitlet_disparity/models/raftstereo-middlebury.pth'
raft_irvc_path = '/home/frc-ag-3/harry_ws/viewpoint_planning/segment_exp/src/fruitlet_disparity/models/iraftstereo_rvc.pth'

ORIG_DIMS=(1440, 1080)

RAFT_DICT = {"middleburry": (get_middle_model_args, raft_middleburry_path),
             'irvc': (get_irvc_model_args, raft_irvc_path)}

BACKWARDS_MAPPING_PATH = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/selected_images/backward_mappings.json'

src_image_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/selected_images/images'
debug_image_dir = '/home/frc-ag-3/Downloads/debug/images'

raft_midd_disp_dir = '/home/frc-ag-3/Downloads/debug/middle_disp'
raft_irvc_disp_dir = '/home/frc-ag-3/Downloads/debug/irvc_disp'
raft_type = "middleburry"
resize_dims = None# (720, 540)
device = 'cuda'

raft_midd_cloud_dir = '/home/frc-ag-3/Downloads/debug/middle_cloud'
raft_irvc_cloud_dir = '/home/frc-ag-3/Downloads/debug/irvc_cloud'
camera_info_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/final_pipeline/camera_info'


if __name__ == '__main__':
    #copy_random_pairs(src_image_dir, debug_image_dir)

    # extract_disparities(debug_image_dir, raft_midd_disp_dir, RAFT_DICT['middleburry'],
    #                     resize_dims, device)
    # extract_disparities(debug_image_dir, raft_irvc_disp_dir, RAFT_DICT['irvc'],
    #                     resize_dims, device)

    extract_point_clouds(raft_midd_disp_dir, debug_image_dir, camera_info_dir,
                         resize_dims, raft_midd_cloud_dir)
    extract_point_clouds(raft_irvc_disp_dir, debug_image_dir, camera_info_dir,
                         resize_dims, raft_irvc_cloud_dir)
