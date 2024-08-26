import os
import json
import numpy as np
from torchvision.io.image import read_image

# seg inds
# Mean:  [0.43726605, 0.4358562, 0.42501393]
# Std:  [0.17180225, 0.1634072, 0.16061011]
# full im
# Mean:  [0.3224376, 0.3340579, 0.3343975]
# Std:  [0.20352836, 0.19546622, 0.21104322]

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

anno_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/id_annotations_filtered'

r_cols = []
g_cols = []
b_cols = []

for filename in os.listdir(anno_dir):
    if not filename.endswith('json'):
        continue

    if not 'left' in filename:
        continue

    anno_path = os.path.join(anno_dir, filename)

    full_annotations = read_json(anno_path)
    image_path = full_annotations['image_path']
    annotations = full_annotations['annotations']

    image = read_image(image_path).float().numpy() / 255.0

    for det in annotations:
        if det['fruitlet_id'] < 0:
            continue

        x0 = det['x0']
        y0 = det['y0']
        x1 = det['x1']
        y1 = det['y1']

        round_x0 = int(np.round(x0))
        round_y0 = int(np.round(y0))
        round_x1 = int(np.round(x1))
        round_y1 = int(np.round(y1))

        fruitlet_im = image[:, round_y0:round_y1, round_x0:round_x1]
        
        r_cols.append(fruitlet_im[0].flatten())
        g_cols.append(fruitlet_im[1].flatten())
        b_cols.append(fruitlet_im[2].flatten())

        # seg_inds = np.array(det['seg_inds'])
        # fruitlet_vals = image[:, seg_inds[:, 0], seg_inds[:, 1]]

        # r_cols.append(fruitlet_vals[0].flatten())
        # g_cols.append(fruitlet_vals[1].flatten())
        # b_cols.append(fruitlet_vals[2].flatten())

r_cols = np.concatenate(r_cols)
g_cols = np.concatenate(g_cols)
b_cols = np.concatenate(b_cols)

r_mean = np.mean(r_cols)
r_std = np.std(r_cols)
g_mean = np.mean(g_cols)
g_std = np.std(g_cols)
b_mean = np.mean(b_cols)
b_std = np.std(b_cols)

print('Mean: ', [r_mean, g_mean, b_mean])
print('Std: ', [r_std, g_std, b_std])