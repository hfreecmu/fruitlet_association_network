import os
import numpy as np
import open3d as o3d
import distinctipy

def process_file(filename, input_dir, output_dir):
    input_path = os.path.join(input_dir, filename)
    
    data_array = np.loadtxt(input_path, comments='//')
    points = data_array[:,0:3]
    colors = data_array[:, 3:6] / 255
    class_labels = data_array[:, 6]
    instance_labels = data_array[:, 7]

    berry_inds = (class_labels == 3)
    berry_instances = np.unique(instance_labels[berry_inds])
    
    print(filename, berry_instances)


def run(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if not filename.endswith('.xyz'):
            continue

        if not '_a' in filename:
            continue

        if not 'A2' in filename:
            continue
        
        process_file(filename, input_dir, output_dir)

INPUT_DIR = '/home/hfreeman/Downloads/datasets/LAST-Straw/'
OUTPUT_DIR = '/home/hfreeman/Downloads/icra_debug/'
if __name__ == '__main__':
    run(INPUT_DIR, OUTPUT_DIR)