import os
import numpy as np
import open3d
import distinctipy

def process_file(filename, input_dir, output_dir, colors, max_num):
    input_path = os.path.join(input_dir, filename)
    
    data_array = np.loadtxt(input_path)

    pc_points = data_array[:,0:3]
    #-1 because maize labelled two ways
    labels = data_array[:, -1].astype(int)

    non_bg_inds = (labels > 0)
    pc_points = pc_points[non_bg_inds]
    labels = labels[non_bg_inds]

    color_inds = labels % colors.shape[0]
    pc_colors = colors[color_inds]
    
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(pc_points)
    pc.colors = open3d.utility.Vector3dVector(pc_colors)

    cloud_path = os.path.join(output_dir, filename.replace('.txt', '.pcd'))
    #open3d.io.write_point_cloud(cloud_path, pc)

    np_path = cloud_path.replace('.pcd', '.npy')
    np_to_save = np.concatenate([pc_points, labels.reshape((-1, 1))], axis=1)
    #np.save(np_path, np_to_save)

    print('Done ' + filename)

    max_num = max(max_num, np.unique(labels).shape[0])
    return max_num

def run(input_dir, output_dir, num_colors, type):
    colors = np.array(distinctipy.get_colors(num_colors))

    output_dir = os.path.join(output_dir, type)

    max_num = -1
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
        
            max_num = process_file(filename, subdir, sub_output_dir, colors, max_num)

    print('Max num is: ', max_num)

INPUT_DIR = '/home/hfreeman/Downloads/datasets/Pheno4D'
OUTPUT_DIR = 'labelling/pheno'
TYPE = 'tomato'
DISNCT_COLORS = 50
if __name__ == '__main__':
    run(INPUT_DIR, OUTPUT_DIR, DISNCT_COLORS, TYPE)