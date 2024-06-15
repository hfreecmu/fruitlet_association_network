from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim

from data.dataloader import get_data_loader
from model.associator import FruitletAssociator
from util.util import save_checkpoint

epsilon = 1e-8
#TODO may want cosine distance
#TODO make more stable by not using exponents? Kumar used exponents.
def get_infonce_loss(feats_0, feats_1, is_pad_0, is_pad_1):
    # negative l2
    
    scale = feats_0.shape[-1]**0.5
    dists = -torch.cdist(feats_0/scale, feats_1/scale)

    #TODO should info nce loss for multiple pairs be added or multiplied?
    dists_exp = torch.exp(dists)

    num_fruitlets = torch.min(is_pad_0.sum(dim=1), is_pad_1.sum(dim=1))

    #TODO any averaging here?
    total_loss_0 = 0
    total_loss_1 = 0
    for ind in range(feats_0.shape[1]):
        pos = dists_exp[:, ind, ind]

        pos = torch.where(is_pad_0[:, ind], torch.zeros_like(pos), pos)
        pos = torch.where(is_pad_1[:, ind], torch.zeros_like(pos), pos)
        # pos[is_pad_0[:, ind]] = 0
        # pos[is_pad_1[:, ind]] = 0

        neg_0 = dists_exp[:, ind, :].sum(-1)
        neg_1 = dists_exp[:, :, ind].sum(-1)

        neg_0 = torch.where(is_pad_1[:, ind], torch.zeros_like(neg_0), neg_0)
        neg_1 = torch.where(is_pad_0[:, ind], torch.zeros_like(neg_1), neg_1)
        # neg_0[is_pad_1[:, ind]] = 0
        # neg_1[is_pad_0[:, ind]] = 0

        loss_0 = -torch.log((epsilon + pos) / (epsilon + neg_0))
        loss_1 = -torch.log((epsilon + pos) / (epsilon + neg_1))

        total_loss_0 += loss_0
        total_loss_1 += loss_1

        # TODO this is not right but want a hack for now
        #total_loss = total_loss + loss_0.mean() + loss_1.mean()

    total_loss = ((total_loss_0 + total_loss_1)/num_fruitlets).sum()


    return total_loss

# TODO maybe only move one and keep other fixed?
# TODO angle should be in range -pi/2 to pi/2 which we could scale 
# to -1 to 1
l2_loss = nn.MSELoss(reduction='none')
smooth_l1_loss = torch.nn.SmoothL1Loss(reduction='none')
def get_ellipse_2d_loss(pos_orig_0, offset_0, 
                       pos_orig_1, offset_1, 
                       offset_scaling,
                       is_pad_0, is_pad_1):
    
    pos_0 = pos_orig_0 + torch.tanh(offset_0)*offset_scaling
    pos_1 = pos_orig_1 + torch.tanh(offset_1)*offset_scaling

    positions_0 = pos_0[:, :, 0:3]
    positions_1 = pos_1[:, :, 0:3]
    scales_0 = pos_0[:, :, 3:5]
    scales_1 = pos_1[:, :, 3:5]
    angle_0 = pos_0[:, :, 5]
    angle_1 = pos_1[:, :, 5]

    pos_loss = l2_loss(positions_0, positions_1)
    scale_loss = l2_loss(scales_0, scales_1)

    phi = (torch.pi / 2) * (angle_0 - angle_1)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    rho = torch.where(phi >= 0, torch.atan2(sin_phi, cos_phi), torch.atan2(-sin_phi, -cos_phi))
    angle_loss = smooth_l1_loss(rho, torch.zeros_like(rho))

    valid_inds = torch.bitwise_and(~is_pad_0, ~is_pad_1)
    pos_loss = pos_loss[valid_inds]
    scale_loss = scale_loss[valid_inds]
    angle_loss = angle_loss[valid_inds]

    #TODO also not right
    return pos_loss.mean(dim=0).sum(), scale_loss.mean(dim=0).sum(), angle_loss.mean()

def train(cfg):
    dataloader = get_data_loader(shuffle=True, **cfg['data'])

    model = FruitletAssociator(**cfg['model']).to(cfg['device'])

    train_cfg = cfg['train']

    #TODO weight decay?
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['lr'])
    
    dataloader_iterator = iter(dataloader)
    for iter_num in range(train_cfg['num_iters']):
        try:
            batch_data = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(dataloader)
            batch_data = next(dataloader_iterator)

        optimizer.zero_grad()

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

        _, _, _, _, all_feats, all_offsets = model_output
        
        total_infonce_loss = 0.0
        total_pos_loss = 0.0
        total_scale_loss = 0.0
        total_angle_loss = 0.0
        # for ind in range(len(all_feats)):
        if False:
            feats_0, feats_1 = all_feats[ind]
            offset_0, offset_1 = all_offsets[ind]

            feats_0 = feats_0[torch.arange(feats_0.shape[0])[:, None], matches_0]
            offset_0 = offset_0[torch.arange(feats_0.shape[0])[:, None], matches_0]
            fruitlet_ellipses_0_sort = fruitlet_ellipses_0[torch.arange(feats_0.shape[0])[:, None], matches_0]
            is_pad_0_sort = is_pad_0[torch.arange(feats_0.shape[0])[:, None], matches_0]

            feats_1 = feats_1[torch.arange(feats_1.shape[0])[:, None], matches_1]
            offset_1 = offset_1[torch.arange(feats_1.shape[0])[:, None], matches_1]
            fruitlet_ellipses_1_sort = fruitlet_ellipses_1[torch.arange(feats_1.shape[0])[:, None], matches_1]
            is_pad_1_sort = is_pad_1[torch.arange(feats_1.shape[0])[:, None], matches_1]

            infonce_loss = get_infonce_loss(feats_0, feats_1,
                                            is_pad_0_sort, is_pad_1_sort)
            
            # ellipse_loss = get_ellipse_2d_loss(fruitlet_ellipses_0_sort, offset_0,
            #                                    fruitlet_ellipses_1_sort, offset_1,
            #                                    cfg['model']['offset_scaling'],
            #                                    is_pad_0_sort, is_pad_1_sort)
            # pos_loss, scale_loss, angle_loss = ellipse_loss

            infonce_loss = infonce_loss*train_cfg["infonce_scale"]
            # pos_loss = pos_loss*train_cfg["pos_scale"]
            # scale_loss = scale_loss*train_cfg["scale_scale"]
            # angle_loss = angle_loss*train_cfg["angle_scale"]

            total_infonce_loss += infonce_loss 
            # total_pos_loss += pos_loss
            # total_scale_loss += scale_loss 
            # total_angle_loss += angle_loss
        if True:
            feats_0, feats_1 = all_feats[-1]

            feats_0 = feats_0[torch.arange(feats_0.shape[0])[:, None], matches_0]
            is_pad_0_sort = is_pad_0[torch.arange(feats_0.shape[0])[:, None], matches_0]

            feats_1 = feats_1[torch.arange(feats_1.shape[0])[:, None], matches_1]
            is_pad_1_sort = is_pad_1[torch.arange(feats_1.shape[0])[:, None], matches_1]

            total_infonce_loss = get_infonce_loss(feats_0, feats_1,
                                            is_pad_0_sort, is_pad_1_sort)

        total_infonce_loss = total_infonce_loss / len(all_feats)
        # total_pos_loss = total_pos_loss / len(all_feats)
        # total_scale_loss = total_scale_loss / len(all_feats)
        # total_angle_loss = total_angle_loss / len(all_feats)

        total_loss = total_infonce_loss# + total_pos_loss + total_scale_loss + total_angle_loss
        total_loss.backward()
        optimizer.step()

        save_iter = iter_num + 1
        if save_iter % train_cfg['step_iter_log'] == 0:
            # loss_array = [total_loss.item(), total_infonce_loss.item(), total_pos_loss.item(),
            #               total_scale_loss.item(), total_angle_loss.item()]
            print('Iter loss at iteration ', iter_num, ' is: ', total_loss.item())

        if save_iter % train_cfg['step_iter_save'] == 0:
            print('Saving checkpoint', iter_num)
            save_checkpoint(iter_num, train_cfg['checkpoint_dir'], model)


if __name__ == "__main__":
    cfg = OmegaConf.load('cfg/v0.yml')
    print(OmegaConf.to_yaml(cfg))
    train(cfg)