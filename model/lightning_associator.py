import torch
from torch import optim
import lightning as L
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from model.associator import FruitletAssociator

# reproduces https://stats.stackexchange.com/questions/573581/why-does-contrastive-loss-and-triplet-loss-have-the-margin-element-in-them
# except the pow
# also https://medium.com/@maksym.bekuzarov/losses-explained-contrastive-loss-f8f57fe32246
def contrastive_loss(features1, features2, gt_match, gt_mask, dist_type, margin=1.0,
                     return_dist=False):
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
    
def match_loss(sim, z0, z1, gt_match, is_pad_0, is_pad_1, return_dist=False):
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)

    if return_dist:
        dists = torch.zeros_like(sim)

    losses = []
    for ind in range(sim.shape[0]):
        b_sim = sim[ind:ind+1, ~is_pad_0[ind]][:, :, ~is_pad_1[ind]]
        b_cert = certainties[ind:ind+1, ~is_pad_0[ind]][:, :, ~is_pad_1[ind]]
        b_z0 = z0[ind:ind+1, ~is_pad_0[ind]]
        b_z1 = z1[ind:ind+1, ~is_pad_1[ind]]

        b, m, n = b_sim.shape

        scores0 = F.log_softmax(b_sim, 2)
        scores1 = F.log_softmax(b_sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
        scores = b_sim.new_full((b, m + 1, n + 1), 0)

        scores[:, :m, :n] = scores0 + scores1 + b_cert
        scores[:, :-1, -1] = F.logsigmoid(-b_z0.squeeze(-1))
        scores[:, -1, :-1] = F.logsigmoid(-b_z1.squeeze(-1))

        if return_dist:
            tmp = dists[ind:ind+1, ~is_pad_0[ind]]
            tmp[:, :, ~is_pad_1[ind]] = torch.exp(scores[:, 0:-1, 0:-1])
            dists[ind:ind+1, ~is_pad_0[ind]] = tmp

        scores = scores[0]
        matches = gt_match[ind, ~is_pad_0[ind]][:, ~is_pad_1[ind]]

        M_inds = torch.argwhere(matches == 1.0)
        if M_inds.shape[0] > 0:
            M_scores = scores[M_inds[:, 0], M_inds[:, 1]].mean()
        else:
            M_scores = 0

        A_inds = torch.argwhere(matches.sum(dim=1) == 0.0)
        if A_inds.shape[0] > 0:
            A_scores = scores[A_inds[:, 0], -1].mean() / 2
        else:
            A_scores = 0

        B_inds = torch.argwhere(matches.sum(dim=0) == 0.0)
        if B_inds.shape[0] > 0:
            B_scores = scores[-1, B_inds[:, 0]].mean() / 2
        else:
            B_scores = 0

        b_loss = -(M_scores + A_scores + B_scores)
        losses.append(b_loss)
    
    loss = torch.stack(losses).mean()
    
    if not return_dist:
        return loss
    else:
        return loss, dists

# def sigmoid_log_double_softmax(
#     sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
# ) -> torch.Tensor:
#     """create the log assignment matrix from logits and similarity"""
#     b, m, n = sim.shape
#     certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
#     scores0 = F.log_softmax(sim, 2)
#     scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
#     scores = sim.new_full((b, m + 1, n + 1), 0)
#     scores[:, :m, :n] = scores0 + scores1 + certainties
#     scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
#     scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
#     return scores

def get_metrics(dists, is_pad_0, is_pad_1, matches_gt, match_thresh):
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
            
        #removing below
        # row_inds, col_inds = linear_sum_assignment(batch_dists)
        # is_match = torch.zeros_like(matches)
        # is_match[row_inds, col_inds] = 1.0
        # is_match[batch_dists > match_thresh] = 0.0

        #replacing
        is_match = torch.zeros_like(matches)
        is_match[batch_dists > match_thresh] = 1.0

        # true_pos = matches[row_inds, col_inds].sum()
        # false_pos = (1-matches)[row_inds, col_inds].sum()
        # false_neg = matches.sum() - true_pos
        true_pos = (matches * is_match).sum()
        false_pos = ((1 - matches) * is_match).sum()
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

    return precision, recall, f1

class LightningAssociator(L.LightningModule):
    def __init__(self,
                 loss_params,
                 match_thresh,
                 lr=None,
                 gamma=None,
                 train_step=None,
                 weight_decay=None, 
                 **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.train_step = train_step
        self.match_thresh = match_thresh
        self.dist_type = loss_params['dist_type']
        self.alpha = loss_params['alpha']

        self.associator = FruitletAssociator(**kwargs)

    def training_step(self, batch, batch_idx):
        file_keys_0, fruitlet_ims_0, fruitlet_clouds_0, \
        is_pad_0, fruitlet_ids_0, \
        file_keys_1, fruitlet_ims_1, fruitlet_clouds_1, \
        is_pad_1, fruitlet_ids_1, \
        matches_gt, masks_gt = batch
        
        data_0 = (fruitlet_ims_0, fruitlet_clouds_0, is_pad_0)
        data_1 = (fruitlet_ims_1, fruitlet_clouds_1, is_pad_1)

        enc_0, enc_1, sim, z0, z1 = self.associator(data_0, data_1)

        # loss = contrastive_loss(enc_0, enc_1, 
        #                         matches_gt, masks_gt,
        #                         self.dist_type,
        #                         margin=self.alpha)

        loss = match_loss(sim, z0, z1, matches_gt, is_pad_0, is_pad_1)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_index):
        file_keys_0, fruitlet_ims_0, fruitlet_clouds_0, \
        is_pad_0, fruitlet_ids_0, \
        file_keys_1, fruitlet_ims_1, fruitlet_clouds_1, \
        is_pad_1, fruitlet_ids_1, \
        matches_gt, masks_gt = batch
        
        data_0 = (fruitlet_ims_0, fruitlet_clouds_0, is_pad_0)
        data_1 = (fruitlet_ims_1, fruitlet_clouds_1, is_pad_1)

        enc_0, enc_1, sim, z0, z1 = self.associator(data_0, data_1)

        # loss, dists = contrastive_loss(enc_0, enc_1, 
        #                         matches_gt, masks_gt,
        #                         self.dist_type,
        #                         margin=self.alpha,
        #                         return_dist=True)

        loss, dists = match_loss(sim, z0, z1, matches_gt, is_pad_0, is_pad_1, return_dist=True)
        
        precision, recall, f1 = get_metrics(dists, is_pad_0, is_pad_1,
                                            matches_gt, self.match_thresh)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log('precision', precision)
        self.log('recall', recall)
        self.log('f1', f1, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        file_keys_0, fruitlet_ims_0, fruitlet_clouds_0, \
        is_pad_0, fruitlet_ids_0, \
        file_keys_1, fruitlet_ims_1, fruitlet_clouds_1, \
        is_pad_1, fruitlet_ids_1, \
        matches_gt, masks_gt = batch
        
        data_0 = (fruitlet_ims_0, fruitlet_clouds_0, is_pad_0)
        data_1 = (fruitlet_ims_1, fruitlet_clouds_1, is_pad_1)

        enc_0, enc_1, sim, z0, z1 = self.associator(data_0, data_1)

        # loss, dists = contrastive_loss(enc_0, enc_1, 
        #                         matches_gt, masks_gt,
        #                         self.dist_type,
        #                         margin=self.alpha,
        #                         return_dist=True)

        loss, dists = match_loss(sim, z0, z1, matches_gt, is_pad_0, is_pad_1, return_dist=True)
        
        precision, recall, f1 = get_metrics(dists, is_pad_0, is_pad_1,
                                            matches_gt, self.match_thresh)
            
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
        
        if self.gamma is not None:
            # sch = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)   
            scheduler = torch.optim.lr_scheduler.StepLR(step_size=self.train_step,
                                                        gamma=self.gamma) 
            return [optimizer], [scheduler]
        else:
            return optimizer

    
        
