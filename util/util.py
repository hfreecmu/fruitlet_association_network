import os
import json
import pickle
from omegaconf import OmegaConf


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

def load_cfg(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    loss_cfg_path = cfg['loss_path']
    loss_cfg = OmegaConf.load(loss_cfg_path)

    cfg['loss_params'] = loss_cfg

    return cfg

def get_identifier(file_key):
    identifier = '_'.join(file_key.split('_')[0:2])

    return identifier

def get_checkpoint_path(checkpoint_dir, exp_name, checkpoint_metrics):
    checkpoint_dir = os.path.join(checkpoint_dir, exp_name, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        raise RuntimeError('checkpoint dir dne: ' + checkpoint_dir)
    
    metric_name = checkpoint_metrics['metric_type']
    is_min = checkpoint_metrics['is_min']

    best_filename = None
    best_val = None
    for filename in os.listdir(checkpoint_dir):
        if not filename.endswith('.ckpt'):
            continue

        metric_identifier = filename.split('-')[1].split('=')[0]
        if not metric_identifier == metric_name:
            continue

        metric_val = float(filename.split('=')[-1].split('.ckpt')[0])

        if best_val is None:
            best_filename = filename
            best_val = metric_val
        elif is_min and metric_val < best_val:
            best_filename = filename
            best_val = metric_val
        elif not is_min and metric_val > best_val:
            best_filename = filename
            best_val = metric_val

    if best_filename is None:
        raise RuntimeError('No valid checkpoint found in: ' + checkpoint_dir)
    
    checkpoint_path = os.path.join(checkpoint_dir, best_filename)
    return checkpoint_path