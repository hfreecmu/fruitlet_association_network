import os
import json
import numpy as np
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.io.image import read_image
import torchvision
import torch
import torch.nn.functional as F
import cv2
import uuid

FRUITLET_MEAN = [0.30344757, 0.3133871, 0.32248256]
FRUITLET_STD = [0.051711865, 0.0505018, 0.056481156]

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def vis_padded_resized_im(padded_im, identifier, det_ind, output_dir):
    padded_im = padded_im.numpy().transpose(1, 2, 0)
    padded_im = padded_im * FRUITLET_STD + FRUITLET_MEAN

    padded_im = (padded_im * 255).astype(np.uint8)

    padded_im = cv2.cvtColor(padded_im, cv2.COLOR_RGB2BGR)

    if identifier is not None:
        output_name = identifier + '_' + str(det_ind) + '.png'
    else:
        output_name = str(det_ind) + '.png'

    output_path = os.path.join(output_dir, output_name)

    cv2.imwrite(output_path, padded_im)


#annotations_path = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/id_annotations/2021_15_0_left.json'
annotations_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/id_annotations/'
vis_padding_dir = '/home/frc-ag-3/Downloads/debug_fruitlet/vis_paddings'
vis_resize_dir = '/home/frc-ag-3/Downloads/debug_fruitlet/vis_resizes'
fruitlet_im_size = 64
device = 'cuda'
output_dim = 512

if True:
    filenames = []
    for filename in os.listdir(annotations_dir):
        if not filename.endswith('.json'):
            continue

        filenames.append(filename)
    
    rand_ind = np.random.choice(len(filenames))
    filename = filenames[rand_ind]
    annotations_path = os.path.join(annotations_dir, filename)
    identifier = uuid.uuid4().hex[0:8]
else:
    identifier = None

full_annotations = read_json(annotations_path)
image_path = full_annotations['image_path']
annotations = full_annotations['annotations']

# We have options. First we can you mask rcnn or keypoint rcnn backbone.
# There is a difference in frozen batch norm and a couple others at the end
# https://pytorch.org/vision/stable/models.html
# ma_model = maskrcnn_resnet50_fpn_v2().backbone
# kp_model = keypointrcnn_resnet50_fpn().backbone
# Then we can use resnet 50 out of the box
# res_model = torchvision.models.resnet50()

# I would say do backbone for individual fruitlets. If you get from image need fpn.
# ALSO if you use pretrained rains, use imagenet std. not your own.

image_transform = torchvision.transforms.Compose([
    torchvision.transforms.ConvertImageDtype(torch.float32),
    torchvision.transforms.Normalize(mean=FRUITLET_MEAN, std=FRUITLET_STD),
])

fruitlet_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((fruitlet_im_size, fruitlet_im_size)),
])

image = read_image(image_path)
image = image_transform(image)

fruitlet_model_ims = []
for det_ind in range(len(annotations)):
    det = annotations[det_ind]

    if det['fruitlet_id'] == -1:
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
    _, fruitlet_height, fruitlet_width = fruitlet_im.shape
    # pad fruitlet_im
    # (left, right, top, bottom)
    if fruitlet_height > fruitlet_width:
        num_horiz_pad = fruitlet_height - fruitlet_width
        left_pad = num_horiz_pad // 2
        right_pad = num_horiz_pad - left_pad
        padding = (left_pad, right_pad, 0, 0)
    elif fruitlet_width > fruitlet_height:
        num_vert_pad = fruitlet_width - fruitlet_height
        top_pad = num_vert_pad // 2
        bottom_pad = num_vert_pad - top_pad
        padding = (0, 0, top_pad, bottom_pad)
    else:
        padding = (0, 0, 0, 0)

    padded_im = F.pad(fruitlet_im, padding, mode='constant', value=0)
    # vis_padded_resized_im(padded_im, identifier, det_ind, vis_padding_dir)

    fruitlet_model_im = fruitlet_transform(padded_im)
    # vis_padded_resized_im(fruitlet_model_im, identifier, det_ind, vis_resize_dir)

    fruitlet_model_ims.append(fruitlet_model_im)

fruitlet_model_ims = torch.stack(fruitlet_model_ims)

encoder = torchvision.models.resnet18()
encoder.fc = torch.nn.Linear(512, output_dim)
encoder.to(device)

fruitlet_model_ims = fruitlet_model_ims.to(device)
encodings = encoder(fruitlet_model_ims)
