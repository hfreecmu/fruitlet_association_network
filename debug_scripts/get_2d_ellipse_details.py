import os
import json
import pickle
import numpy as np
from sklearn.decomposition import PCA

# TODO will have to re-run this once all data labelled
# TODO include right images?
# TODO need to center to get norms!!!
#Mean:  [[0.01515367144828234, -0.014630158413319364, 0.4377650249983249], 0.004248618083974694, 0.0021318646462382008]
#Std:  [[0.03693850598871499, 0.028846982679794086, 0.07828516204898538], 0.0021931332935247344, 0.000561814052844269]


def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

#TODO will have to re-run this once all data is labelled
anno_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/id_annotations'
cloud_dir = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/point_clouds'

scale_0s = []
scale_1s = []
fruitlet_cluster_meds = []
for filename in os.listdir(anno_dir):
    if not filename.endswith('json'):
        continue

    if not 'left' in filename:
        continue

    anno_path = os.path.join(anno_dir, filename)
    full_annotations = read_json(anno_path)
    annotations = full_annotations['annotations']

    cloud_path = os.path.join(cloud_dir, filename.replace('.json', '.pkl'))
    seg_clouds = read_pickle(cloud_path)

    assert len(seg_clouds) == len(annotations)

    for ind in range(len(annotations)):
        det = annotations[ind]
        cloud_points = seg_clouds[ind]

        if det['fruitlet_id'] < 0:
            continue

        nan_inds = np.isnan(cloud_points).any(axis=1)
        cloud_points = cloud_points[~nan_inds]

        if cloud_points.shape[0] == 0:
            print('empty cloud')
            continue

        # TODO med or median?
        fruitlet_cluster_meds.append(np.median(cloud_points, axis=0))

        cloud_points = cloud_points[:, 0:2]
        pca = PCA(n_components=2)
        _ = pca.fit_transform(cloud_points)
        eig_vals, eig_vecs = pca.explained_variance_, pca.components_

        scale_0, scale_1 = np.sqrt(eig_vals)
        # don't need to scale theta as limited already
        #rot_mat = eig_vecs

        assert scale_0 > scale_1

        scale_0s.append(scale_0)
        scale_1s.append(scale_1)

fruitlet_cluster_meds = np.stack(fruitlet_cluster_meds)

scale_0s = np.array(scale_0s)
scale_1s = np.array(scale_1s)

scale_0_mean = scale_0s.mean()
scale_1_mean = scale_1s.mean()
med_dists_mean =  np.mean(fruitlet_cluster_meds, axis=0)

scale_0_std = scale_0s.std()
scale_1_std = scale_1s.std()
med_dists_std = np.std(fruitlet_cluster_meds, axis=0)

print('Mean: ', [med_dists_mean.tolist(), scale_0_mean, scale_1_mean])
print('Std: ', [med_dists_std.tolist(), scale_0_std, scale_1_std])
