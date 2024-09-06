import os

id_dir_0 = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/v0/id_annotations'
id_dir_1 = '/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/inhand/tro_final/id_annotations'

ids_0 = {}
for filename in os.listdir(id_dir_0):
    if not filename.endswith('.json'):
        continue

    identifier = '_'.join(filename.split('_')[0:2])
    if identifier not in ids_0:
        ids_0[identifier] = []

    ids_0[identifier].append(filename)

ids_1 = {}
for filename in os.listdir(id_dir_1):
    if not filename.endswith('.json'):
        continue

    identifier = '_'.join(filename.split('_')[0:2])
    if identifier not in ids_1:
        ids_1[identifier] = []

    ids_1[identifier].append(filename)

overlap = set()
for id in ids_0:
    if id in  ids_1:
        overlap.add(id)

print('num overlaps: ', len(overlap))
print('num_total: ', len(ids_0) + len(ids_1) - len(overlap))

missings_0s = []
for id in ids_0:
    if id not in overlap:
        missings_0s.append(id)

missings_1s = []
for id in ids_1:
    if id not in overlap:
        missings_1s.append(id)

print('missing 0s: ', len(missings_0s))
print('missing 1s: ', len(missings_1s))

count_missing_0 = 0
for id in missings_0s:
    count_missing_0 += len(ids_0[id])

print(count_missing_0)

# print('num 1s: ', len(ids_1))

missings_0s = sorted(missings_0s)
for m in missings_0s:
    print(m)