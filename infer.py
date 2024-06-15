from omegaconf import OmegaConf
import torch
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from data.dataloader import get_data_loader
from model.associator import FruitletAssociator

from util.util import load_checkpoint

from train import get_infonce_loss

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

        data_0 = (fruitlet_images_0, fruitlet_ellipses_0, is_pad_0)
        data_1 = (fruitlet_images_1, fruitlet_ellipses_1, is_pad_1)
        
        model_output = model(data_0, data_1)

        feats_0, _, feats_1, _, _, _ = model_output

        feats_0 = feats_0[torch.arange(feats_0.shape[0])[:, None], matches_0]
        is_pad_0 = is_pad_0[torch.arange(feats_0.shape[0])[:, None], matches_0]

        feats_1 = feats_1[torch.arange(feats_1.shape[0])[:, None], matches_1]
        is_pad_1 = is_pad_1[torch.arange(feats_1.shape[0])[:, None], matches_1]


        # infonce_loss = get_infonce_loss(feats_0, feats_1,
        #                                 is_pad_0, is_pad_1)
        
        for batch_ind in range(feats_0.shape[0]):
            f_0 = feats_0[batch_ind]
            f_1 = feats_1[batch_ind]

            f_0 = f_0[~is_pad_0[batch_ind]].cpu().numpy()
            f_1 = f_1[~is_pad_1[batch_ind]].cpu().numpy()

            dists = cdist(f_0, f_1)
            row_ind, col_ind = linear_sum_assignment(dists)

            for r, c in zip(row_ind, col_ind):
                print(r, c)




if __name__ == "__main__":
    cfg = OmegaConf.load('cfg/v0.yml')
    print(OmegaConf.to_yaml(cfg))
    with torch.no_grad():
        infer(cfg)