import os
import cv2
import pickle
import distinctipy

def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    

im_path = 'labelling/selected_images/images/2023_374_8_left.png'
det_dir = 'labelling/detections'

det_path = os.path.join(det_dir, os.path.basename(im_path).replace('.png', '.pkl'))

im = cv2.imread(im_path)
det_info = read_pickle(det_path)
boxes = det_info['boxes']
segmentations = det_info['segmentations']

num_matches = len(boxes)
colors = distinctipy.get_colors(num_matches)

for ind in range(num_matches):
    x0, y0, x1, y1, _ = boxes[ind]
    seg_inds = segmentations[ind]
    color = ([int(255*colors[ind][0]), int(255*colors[ind][1]), int(255*colors[ind][2])])

    im[seg_inds[:, 0], seg_inds[:, 1]] = color
    cv2.rectangle(im, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness=2)

cv2.imshow('vis', im)
cv2.waitKey(0)