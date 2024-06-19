import torch
from torch import optim
import lightning as L
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from model.associator import FruitletAssociator

#max(||f(a) - f(p)||_2 - ||f(a) - f(n)||_2 + a, 0)
def get_triplet_losses(enc_0, enc_1, dist_type, alpha,
                       matches_gt, masks_gt):
    if dist_type == 'l2':
        dists = torch.cdist(enc_0, enc_1)
    else:
        raise RuntimeError('Invalid dist type: ' + dist_type)
    
    breakpoint()

# reproduces https://stats.stackexchange.com/questions/573581/why-does-contrastive-loss-and-triplet-loss-have-the-margin-element-in-them
# except the pow
def contrastive_loss(features1, features2, gt_match, gt_mask, dist_type, margin=1.0,
                     return_dist = False):
    # Ensure features are normalized
    features1 = F.normalize(features1, p=2, dim=-1)
    features2 = F.normalize(features2, p=2, dim=-1)

    if dist_type == 'l2':
        distances = torch.cdist(features1, features2)
    else:
        raise RuntimeError('Invalid dist type: ' + dist_type)
    
    # Compute the contrastive loss
    match_loss = gt_match * torch.square(distances)
    non_match_loss = (1 - gt_match) * torch.square(torch.clamp(margin - distances, min=0.0))
    #non_match_loss = (1 - gt_match) * torch.clamp(margin - distances, min=0.0)
    
    # Apply the gt_mask to ignore padded objects
    match_loss = match_loss * gt_mask
    non_match_loss = non_match_loss * gt_mask

    pos_vals = gt_match*gt_mask
    neg_vals = (1-gt_match)*gt_mask

    pos_loss = match_loss.sum(dim=(1,2)) / pos_vals.sum(dim=(1,2))
    neg_loss = non_match_loss.sum(dim=(1,2)) / neg_vals.sum(dim=(1,2))
    loss = pos_loss + neg_loss

    # Sum the losses over the object dimensions and average over the batch
    # total_loss = match_loss + non_match_loss

    # loss = total_loss.sum(dim=(1, 2)) / gt_mask.sum(dim=(1, 2))
    
    if not return_dist:
        return loss.mean()
    else:
        return loss.mean(), distances
    

class LightningAssociator(L.LightningModule):
    def __init__(self,
                 loss_params,
                 lr=None,
                 weight_decay=None, 
                 **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.dist_type = loss_params['dist_type']
        self.alpha = loss_params['alpha']

        self.associator = FruitletAssociator(**kwargs)

    def training_step(self, batch, batch_idx):
        file_keys_0, fruitlet_ims_0, cloud_ims_0, \
               is_pad_0, fruitlet_ids_0, \
               file_keys_1, fruitlet_ims_1, cloud_ims_1, \
               is_pad_1, fruitlet_ids_1, \
               matches_gt, masks_gt = batch
        
        data_0 = (fruitlet_ims_0, cloud_ims_0, is_pad_0)
        data_1 = (fruitlet_ims_1, cloud_ims_1, is_pad_1)

        enc_0, enc_1 = self.associator(data_0, data_1)

        loss = contrastive_loss(enc_0, enc_1, 
                                matches_gt, masks_gt,
                                self.dist_type,
                                margin=self.alpha)
        
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_index):
        file_keys_0, fruitlet_ims_0, cloud_ims_0, \
               is_pad_0, fruitlet_ids_0, \
               file_keys_1, fruitlet_ims_1, cloud_ims_1, \
               is_pad_1, fruitlet_ids_1, \
               matches_gt, masks_gt = batch
        
        data_0 = (fruitlet_ims_0, cloud_ims_0, is_pad_0)
        data_1 = (fruitlet_ims_1, cloud_ims_1, is_pad_1)

        enc_0, enc_1 = self.associator(data_0, data_1)

        loss = contrastive_loss(enc_0, enc_1, 
                                matches_gt, masks_gt,
                                self.dist_type,
                                margin=self.alpha)
        
        self.log("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        file_keys_0, fruitlet_ims_0, cloud_ims_0, \
               is_pad_0, fruitlet_ids_0, \
               file_keys_1, fruitlet_ims_1, cloud_ims_1, \
               is_pad_1, fruitlet_ids_1, \
               matches_gt, masks_gt = batch
        
        data_0 = (fruitlet_ims_0, cloud_ims_0, is_pad_0)
        data_1 = (fruitlet_ims_1, cloud_ims_1, is_pad_1)

        enc_0, enc_1 = self.associator(data_0, data_1)

        loss, dists = contrastive_loss(enc_0, enc_1, 
                                matches_gt, masks_gt,
                                self.dist_type,
                                margin=self.alpha,
                                return_dist=True)
        
        dists = dists.cpu().numpy()
        padded_0s = is_pad_0.cpu().numpy()
        padded_1s = is_pad_1.cpu().numpy()
        full_true_pos = 0
        full_false_pos = 0
        full_false_neg = 0
        for batch_dists, batch_matches_gt, batch_pad_0s, batch_pad_1s \
            in zip(dists, matches_gt, padded_0s, padded_1s):

            batch_dists = batch_dists[~batch_pad_0s][:, ~batch_pad_1s]
            matches = batch_matches_gt[~batch_pad_0s][:, ~batch_pad_1s]
            
            row_inds, col_inds = linear_sum_assignment(batch_dists)

            true_pos = matches[row_inds, col_inds].sum()
            false_pos = (1-matches)[row_inds, col_inds].sum()
            false_neg = matches.sum() - true_pos

            full_true_pos += true_pos
            full_false_pos += false_pos
            full_false_neg += false_neg
            
        if full_true_pos == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = full_true_pos / (full_true_pos + full_false_pos)
            recall = full_true_pos / (full_true_pos + full_false_neg)
            f1 = 2*precision*recall / (precision + recall)
            
        self.log("test_loss", loss)
        self.log('precision', precision)
        self.log('recall', recall)
        self.log('f1', f1)
        return loss


    def configure_optimizers(self):
        if self.weight_decay is None:
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        else:
            optimizer = optim.Adam(self.parameters(), 
                                   lr=self.lr,
                                   weight_decay=self.weight_decay)
        return optimizer

    
        
