import os
import json
import shutil

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def write_json(path, data, pretty=False):
    with open(path, 'w') as f:
        if not pretty:
            json.dump(data, f)
        else:
            json.dump(data, f, indent=4)

# used to create mappings file and group images by same cluster

selected_image_paths = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/selected_images.json'
output_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/selected_images'
include_right = True

image_paths = read_json(selected_image_paths)

index_dict = {}
forward_mappings = {}
backward_mappings = {}

for image_path in image_paths:
    if not 'LEFT' in image_path:
        raise RuntimeError('should not happen')

    date_info = image_path.split('/')[-3]

    #e.g. 558_2023-05-19-22-14-35_tsdfroi_1
    tag_id_year, month, day, _, _, _ = date_info.split('-')

    tag_id, year = tag_id_year.split('_')

    cluster_identifier = '_'.join([year, tag_id])
    if cluster_identifier not in index_dict:
        index_dict[cluster_identifier] = 0

    new_filename = cluster_identifier + '_' + str(index_dict[cluster_identifier]) + '_left.png'
    new_path = os.path.join(output_dir, 'images', new_filename)
    shutil.copyfile(image_path, new_path)

    forward_mappings[image_path] = new_filename
    backward_mappings[new_filename] = image_path

    if include_right:
        right_image_path = image_path.replace('LEFT', 'RIGHT')
        #sometimes this will happen
        if not os.path.exists(right_image_path):
            continue

        right_new_filename = new_filename.replace('left', 'right')
        right_new_path = os.path.join(output_dir, 'images', right_new_filename)
        shutil.copyfile(right_image_path, right_new_path)

        forward_mappings[right_image_path] = os.path.basename(right_new_filename)
        backward_mappings[right_new_filename] = right_image_path

    index_dict[cluster_identifier] += 1

print(len(forward_mappings))
print(len(backward_mappings))

write_json(os.path.join(output_dir, 'forward_mappings.json'), forward_mappings, pretty=True)
write_json(os.path.join(output_dir, 'backward_mappings.json'), backward_mappings, pretty=True)
