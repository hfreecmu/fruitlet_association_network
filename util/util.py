import os
import json
import pickle
import torch

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

def read_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_identifier(file_key):
    identifier = '_'.join(file_key.split('_')[0:2])

    return identifier

# def save_checkpoint(iter_num, checkpoint_dir, model):
#     path = os.path.join(checkpoint_dir, 'iter_%d.pth' % iter_num)
#     torch.save(model.state_dict(), path)

# def load_checkpoint(iter_num, checkpoint_dir, model):
#     if iter_num == -1:
#         max_iter = -1
#         for filename in os.listdir(checkpoint_dir):
#             if not filename.endswith('.pth'):
#                 continue
            
#             checkpoint_iter = int(filename.split('_')[1].split('.pth')[0])
#             if checkpoint_iter > max_iter:
#                 max_iter = checkpoint_iter

#         iter_num = max_iter
    
#     path = os.path.join(checkpoint_dir, 'iter_%d.pth' % iter_num)
#     model.load_state_dict(torch.load(path))