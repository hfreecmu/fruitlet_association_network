import os
import json
import numpy as np

# widths mean:  62.566184450668
# heights mean:  83.43935303454538
# widths med:  59.60137939453125
# heights med:  81.23833465576172

# will have to re-run this once all data labelled

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

anno_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/id_annotations'

widths = []
heights = []

for filename in os.listdir(anno_dir):
    if not filename.endswith('json'):
        continue

    if not 'left' in filename:
        continue

    anno_path = os.path.join(anno_dir, filename)

    annotations = read_json(anno_path)['annotations']

    for det in annotations:
        if det['fruitlet_id'] < 0:
            continue

        x0 = det['x0']
        y0 = det['y0']
        x1 = det['x1']
        y1 = det['y1']

        width = x1 - x0
        height = y1 - y0

        widths.append(width)
        heights.append(height)

print('widths mean: ', np.mean(widths))
print('heights mean: ', np.mean(heights))
print('widths med: ', np.median(widths))
print('heights med: ', np.median(heights))