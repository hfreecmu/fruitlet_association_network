import os
import numpy as np
import json

def write_json(path, data, pretty=False):
    with open(path, 'w') as f:
        if not pretty:
            json.dump(data, f)
        else:
            json.dump(data, f, indent=4)

### input dirs
inhand_dir_2021 = '/media/frc-ag-3/umass_1/umass/umass_2021_data/umass_2021_bags'
inhand_dir_2023_70 = '/media/frc-ag-3/umass_1/umass/umass_2023_data/in-hand_images/rectified_images/70_clusters'
robot_dir_2023_70 = '/media/frc-ag-3/umass_1/umass/umass_2023_data/field_data/rectified_images/70_clusters'
#inhand_dir_2023_30 = '/media/frc-ag-3/umass_1/umass/umass_2023_data/in-hand_images/rectified_images/30_clusters'
#robot_dir_2023_30 = '/media/frc-ag-3/umass_1/umass/umass_2023_data/field_data/rectified_images/30_clusters'
###

### output dirs
images_output = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/selected_images.json'
image_dist_output = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/selected_image_dist.json'
###

# num images per tag
images_per_tag = 4
#

training_dirs = [inhand_dir_2021,
                 inhand_dir_2023_70,
                 robot_dir_2023_70,
                 #inhand_dir_2023_30,
                 #robot_dir_2023_30,
                 ]

filter_dict = {inhand_dir_2021: (True, {245, 250, 170, 172, 173, 174, 176,
                                        177, 190, 193, 229, 238, 247, 251,
                                        8, 15, 42, 48, 51, 54, 58,
                                        60, 62, 67, 69, 71, 81, 82,
                                        95, 97, 100, 105, 106, 120, 121,
                                        125, 128, 137, 154, 160, 161, 163}),
               inhand_dir_2023_70: (False, set()),
               robot_dir_2023_70: (False, set()),
               inhand_dir_2023_30: (False, set()),
               robot_dir_2023_30: (False, set())
               }

year_dict = {inhand_dir_2021: '2021',
             inhand_dir_2023_70: '2023',
             robot_dir_2023_70: '2023',
             inhand_dir_2023_30: '2023',
             robot_dir_2023_30: '2023'
            }

selected_image_paths = []
count_dict = {}

for training_dir in training_dirs:
    should_filter, target_tags = filter_dict[training_dir]
    year = year_dict[training_dir]

    if not str(year) + "_tags" in count_dict:
        count_dict[str(year) + "_tags"] = set()
        count_dict[str(year) + "_images"] = 0

    for tag_dirname in os.listdir(training_dir):
        tag_dir = os.path.join(training_dir, tag_dirname)

        if not os.path.isdir(tag_dir):
            continue

        tag_id = int(tag_dirname.split('_')[0])

        if should_filter:
            if not tag_id in target_tags:
                continue

        count_dict[str(year) + "_tags"].add(tag_id)

        image_dir = os.path.join(tag_dir, 'COLOR')
        if not os.path.exists(image_dir):
            raise RuntimeError('No COLOR dir in ' + tag_dir)
        
        # get all image candidates
        full_image_candidates = []
        for filename in os.listdir(image_dir):
            if not filename.endswith('.png'):
                continue

            #only select left images for now. can get right ones later
            if not 'LEFT' in filename:
                continue

            image_path = os.path.join(image_dir, filename)

            #don't include if no right image
            right_image_path = image_path.replace('LEFT', 'RIGHT')
            if not os.path.exists(right_image_path):
                continue

            full_image_candidates.append(image_path)
        
        # this can happen
        if len(full_image_candidates) < images_per_tag:
            continue

        image_candidate_inds = np.random.choice(len(full_image_candidates), images_per_tag, replace=False)

        for ind in image_candidate_inds:
            selected_image_paths.append(full_image_candidates[ind])
            count_dict[str(year) + "_images"] += 1

write_json(images_output, selected_image_paths, pretty=True)

for key in count_dict:
    if not "tags" in key:
        continue
    count_dict[key] = len(count_dict[key])

write_json(image_dist_output, count_dict, pretty=True)

print('Done')        