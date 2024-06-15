from omegaconf import OmegaConf
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from data.dataloader import get_data_loader
from model.associator import FruitletAssociator

from util.util import load_checkpoint

from train import get_infonce_loss
from data.dataloader import ELLIPSE_MEAN, ELLIPSE_STD

from train import get_ellipse_2d_loss
def infer(cfg):
    dataloader = get_data_loader(shuffle=False, **cfg['data'])

    model = FruitletAssociator(**cfg['model']).to(cfg['device'])

    infer_cfg = cfg['infer']
    load_checkpoint(infer_cfg['checkpoint_iter'], infer_cfg['checkpoint_dir'], model)

    model.eval()

    for _, batch_data in enumerate(dataloader):

        _, fruitlet_images_0, fruitlet_ellipses_0, \
        fruitlet_ids_0, is_pad_0, matches_0, \
        _, fruitlet_images_1, fruitlet_ellipses_1, \
        fruitlet_ids_1, is_pad_1, matches_1 = batch_data

        fruitlet_images_0 = fruitlet_images_0.to(cfg['device'])
        fruitlet_images_1 = fruitlet_images_1.to(cfg['device'])
        fruitlet_ellipses_0 = fruitlet_ellipses_0.to(cfg['device'])
        fruitlet_ellipses_1 = fruitlet_ellipses_1.to(cfg['device'])
        is_pad_0 = is_pad_0.to(cfg['device'])
        is_pad_1 = is_pad_1.to(cfg['device'])

        data_0 = (fruitlet_images_0, torch.clone(fruitlet_ellipses_0), is_pad_0)
        data_1 = (fruitlet_images_1, torch.clone(fruitlet_ellipses_1), is_pad_1)
        
        model_output = model(data_0, data_1)

        _, positions_0, _, positions_1, _, all_offsets = model_output

        # feats_0 = feats_0[torch.arange(feats_0.shape[0])[:, None], matches_0]
        # positions_0 = positions_0[torch.arange(feats_0.shape[0])[:, None], matches_0]
        # is_pad_0 = is_pad_0[torch.arange(feats_0.shape[0])[:, None], matches_0]

        # feats_1 = feats_1[torch.arange(feats_1.shape[0])[:, None], matches_1]
        # positions_1 = positions_1[torch.arange(feats_1.shape[0])[:, None], matches_1]
        # is_pad_1 = is_pad_1[torch.arange(feats_1.shape[0])[:, None], matches_1]

        total_ellipse_loss = 0.0
        for ind in range(len(all_offsets)):
            offset_1 = all_offsets[ind]

            fruitlet_ellipses_0_sort = fruitlet_ellipses_0[torch.arange(offset_1.shape[0])[:, None], matches_0]
            is_pad_0_sort = is_pad_0[torch.arange(offset_1.shape[0])[:, None], matches_0]

            offset_1 = offset_1[torch.arange(offset_1.shape[0])[:, None], matches_1]
            fruitlet_ellipses_1_sort = fruitlet_ellipses_1[torch.arange(offset_1.shape[0])[:, None], matches_1]
            is_pad_1_sort = is_pad_1[torch.arange(offset_1.shape[0])[:, None], matches_1]

            ellipse_loss = get_ellipse_2d_loss(fruitlet_ellipses_0_sort, None,
                                               fruitlet_ellipses_1_sort, offset_1,
                                               cfg['model']['offset_scaling'],
                                               is_pad_0_sort, is_pad_1_sort)
            
            total_ellipse_loss += ellipse_loss
        
        total_ellipse_loss = total_ellipse_loss / len(all_offsets)

        # infonce_loss = get_infonce_loss(feats_0, feats_1,
        #                                 is_pad_0, is_pad_1)
        
        positions_0 = positions_0[torch.arange(offset_1.shape[0])[:, None], matches_0]
        is_pad_0 = is_pad_0[torch.arange(offset_1.shape[0])[:, None], matches_0]

        positions_1 = positions_1[torch.arange(offset_1.shape[0])[:, None], matches_1]
        is_pad_1 = is_pad_1[torch.arange(offset_1.shape[0])[:, None], matches_1]

        for batch_ind in range(positions_0.shape[0]):
            # f_0 = feats_0[batch_ind]
            # f_1 = feats_1[batch_ind]

            # f_0 = f_0[~is_pad_0[batch_ind]].cpu().numpy()
            # f_1 = f_1[~is_pad_1[batch_ind]].cpu().numpy()

            # dists = cdist(f_0, f_1)
            # row_ind, col_ind = linear_sum_assignment(dists)

            # for r, c in zip(row_ind, col_ind):
            #     print(r, c)

            

            p_0 = positions_0[batch_ind, :, :-1]
            p_1 = positions_1[batch_ind, :, :-1]

            p_0 = p_0[~is_pad_0[batch_ind]]
            p_1 = p_1[~is_pad_1[batch_ind]]

            mean = ELLIPSE_MEAN[0]
            std = ELLIPSE_STD[0]

            p_0_unnorm = p_0.cpu().numpy()*std + mean
            p_1_unnorm = p_1.cpu().numpy()*std + mean

            dists = cdist(p_0_unnorm, p_1_unnorm)

            row_ind, col_ind = linear_sum_assignment(dists)

            for r, c in zip(row_ind, col_ind):
                print(r, c)




if __name__ == "__main__":
    cfg = OmegaConf.load('cfg/v0.yml')
    print(OmegaConf.to_yaml(cfg))
    with torch.no_grad():
        infer(cfg)