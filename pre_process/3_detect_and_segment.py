import os
import numpy as np
import cv2
import torch
import pickle
import distinctipy

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model

def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def write_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_seg_model(model_file, score_thresh, nms_thresh=0.3):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_file 
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.INPUT.MIN_SIZE_TEST = 1080
    cfg.INPUT.MAX_SIZE_TEST = 1440

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_thresh

    model = build_model(cfg)
    model.load_state_dict(torch.load(model_file)['model'])
    model.eval()

    return model

def detect_image(model, image_path):
    im = cv2.imread(image_path)
    im_boxes = []
    segmentations = []

    image = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))
    inputs = {"image": image}
    with torch.no_grad():
        outputs = model([inputs])[0]

    masks = outputs['instances'].get('pred_masks').to('cpu').numpy()
    boxes = outputs['instances'].get('pred_boxes').to('cpu')
    scores = outputs['instances'].get('scores').to('cpu').numpy()

    num = len(boxes)

    for i in range(num):
        x0, y0, x1, y1 = boxes[i].tensor.numpy()[0]
        seg_inds = np.argwhere(masks[i, :, :] > 0)
        score = scores[i]

        im_boxes.append([float(x0), float(y0), float(x1), float(y1), float(score)])
        segmentations.append(seg_inds)

    return im_boxes, segmentations


image_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/selected_images/images'
model_path = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/segmentation/turk/mask_rcnn/mask_best.pth'
output_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/detections'
score_thresh = 0.4

vis = True
num_vis = 20
vis_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/vis_detections'

if __name__ == "__main__":
    model = load_seg_model(model_path, score_thresh)


    for filename in os.listdir(image_dir):
        if not filename.endswith('.png'):
            continue

        if not 'left' in filename:
            continue

        image_path = os.path.join(image_dir, filename)

        im_boxes, segmentations = detect_image(model, image_path)

        output_dict = {
            'boxes': im_boxes,
            'segmentations': segmentations
        }

        output_path = os.path.join(output_dir, filename.replace('.png', '.pkl'))
        write_pickle(output_path, output_dict)

        if vis and num_vis > 0:
            vis_im = cv2.imread(image_path).copy()

            im_boxes = output_dict['boxes']
            segmentations = output_dict['segmentations']

            num_boxes = len(im_boxes)
            colors = distinctipy.get_colors(num_boxes)

            for ind in range(num_boxes):
                x0, y0, x1, y1, _ = im_boxes[ind]
                seg_inds = segmentations[ind]

                color = ([int(255*colors[ind][0]), int(255*colors[ind][1]), int(255*colors[ind][2])])
                cv2.rectangle(vis_im, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness=2)

                vis_im[seg_inds[:, 0], seg_inds[:, 1]] = color

            vis_path = os.path.join(vis_dir, filename)
            cv2.imwrite(vis_path, vis_im)

            num_vis -= 1

