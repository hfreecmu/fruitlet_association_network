import torch
from torch import optim
import lightning as L
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np

from model.associator import FruitletAssociator
from util.util import vis_matches

# reproduces https://stats.stackexchange.com/questions/573581/why-does-contrastive-loss-and-triplet-loss-have-the-margin-element-in-them
# except the pow
# also https://medium.com/@maksym.bekuzarov/losses-explained-contrastive-loss-f8f57fe32246
# also https://lilianweng.github.io/posts/2021-05-31-contrastive/
# but different from https://towardsdatascience.com/contrastive-loss-explaned-159f2d4a87ec
def contrastive_loss_fn(features1, features2, gt_match, gt_mask, 
                        loss_params):
    
    dist_type = loss_params['dist_type']
    margin = loss_params['margin']

    # Ensure features are normalized
    features1 = F.normalize(features1, p=2, dim=-1)
    features2 = F.normalize(features2, p=2, dim=-1)

    if dist_type == 'l2':
        distances = torch.cdist(features1, features2)
        # Compute the contrastive loss
        match_loss = gt_match * torch.square(distances)
        non_match_loss = (1 - gt_match) * torch.square(torch.clamp(margin - distances, min=0.0))
    elif dist_type == 'cos':
        cosines = torch.einsum("bmd,bnd->bmn", features1, features2)
        distances = 1 - cosines
        # Compute the contrastive loss
        match_loss = gt_match * distances
        non_match_loss = (1 - gt_match) * torch.clamp(cosines - margin, min=0.0)
    else:
        raise RuntimeError('Invalid dist type: ' + dist_type)
    
        
    # Apply the gt_mask to ignore padded objects
    match_loss = match_loss * gt_mask
    non_match_loss = non_match_loss * gt_mask

    #raise RuntimeError('shoudl I be weighting it like this or just summing equal weight?')

    pos_vals = gt_match*gt_mask
    neg_vals = (1-gt_match)*gt_mask

    pos_loss = match_loss.sum(dim=(1,2)) / pos_vals.sum(dim=(1,2))
    neg_loss = non_match_loss.sum(dim=(1,2)) / neg_vals.sum(dim=(1,2))
    loss = pos_loss + neg_loss
    
    return loss.mean(), distances

def n_pair_loss_fn(features1, features2, gt_match, gt_mask, loss_params):
    distances = torch.where(gt_mask > 0.0, torch.cdist(features1, features2), (1-gt_mask)*torch.inf)
    sim = -distances
    
    match_inds = torch.argwhere(gt_match == 1.0)
    if match_inds.shape[0] == 0:
        raise RuntimeError('this should not happen')
    
    pos = sim[match_inds[:, 0], match_inds[:, 1], match_inds[:, 2]]
    neg_row = sim[match_inds[:, 0], match_inds[:, 1]]
    neg_col = sim[match_inds[:, 0], :, match_inds[:, 2]]

    neg_row[torch.arange(neg_row.shape[0]), match_inds[:, 2]] = -torch.inf
    neg_col[torch.arange(neg_col.shape[0]), match_inds[:, 1]] = -torch.inf

    soft_batch = torch.concatenate((pos[:, None], neg_row, neg_col), dim=-1)

    log_softmax = F.log_softmax(soft_batch, 1)

    loss = -log_softmax[:, 0].mean()

    return loss, distances
    
def match_loss_fn(sim, z0, z1, gt_match, gt_mask, is_pad_0, is_pad_1, include_dist):
    sim_orig = sim

    sim = torch.where(gt_mask > 0.0, sim, torch.zeros_like(sim) - torch.inf)

    scores0 = F.log_softmax(sim, 2)
    scores1 =F.log_softmax(sim, 1)
    scores = torch.where(gt_mask > 0.0, scores0 + scores1, torch.zeros_like(sim) - torch.inf)

    M_inds = torch.argwhere(gt_match == 1.0)
    if M_inds.shape[0] > 0:
        M_scores = scores[M_inds[:, 0], M_inds[:, 1], M_inds[:, 2]].mean()
    else:
        M_scores = sim.new_full((1,), 0)
    
    loss = -M_scores

    # dists
    if include_dist:
        dists = scores[:, 0:-1, 0:-1].exp()
        dists2 = -sim_orig
    else:
        dists = None

    return loss, dists, dists2

def get_loss(loss_params, features_0, features_1, 
             sim, z0, z1, is_pad_0, is_pad_1,
             matches_gt, masks_gt,
             pred_confidences, gt_confidences, include_bce, bce_loss_fn,
             include_dist=True):
    
    if 'contrastive' in loss_params['loss_type']:
        match_loss, dists = contrastive_loss_fn(features_0, features_1, matches_gt, masks_gt, 
                                                loss_params)
    elif loss_params['loss_type'] == 'matching':
        match_loss, dists, dists2 = match_loss_fn(sim, z0, z1, matches_gt, masks_gt, is_pad_0, is_pad_1, include_dist)
    elif loss_params['loss_type'] == 'n_pair':
        match_loss, dists = n_pair_loss_fn(features_0, features_1, matches_gt, masks_gt, 
                                           loss_params)
    else:
        raise RuntimeError('Invalid loss type: ' + loss_params['loss_type'])
    
    if include_bce and len(pred_confidences) > 0:
        bce_loss = bce_loss_fn(pred_confidences, gt_confidences)
    else:
        bce_loss = 0.0

    return match_loss, bce_loss, dists, dists2

# TODO don't like how vis is in loss metrics. Move it.
# probably have a common function to find matches
def get_loss_metrics(dists, dists2, is_pad_0, is_pad_1, matches_gt, loss_params,
                     vis=False, 
                     im_paths_0=None, anno_paths_0=None, det_inds_0=None, 
                     im_paths_1=None, anno_paths_1=None, det_inds_1=None, 
                     vis_dir=None):
    dists = dists.cpu().numpy()
    padded_0s = is_pad_0.cpu().numpy()
    padded_1s = is_pad_1.cpu().numpy()

    full_true_pos = 0
    full_false_pos = 0
    full_false_neg = 0
    inst_precs = []
    inst_recs = []
    inst_f1s = []
    for ind, batch_data in enumerate(zip(dists, dists2, matches_gt, padded_0s, padded_1s)):
        batch_dists, batch_dists2, batch_matches_gt, batch_pad_0s, batch_pad_1s = batch_data

        batch_dists = batch_dists[~batch_pad_0s][:, ~batch_pad_1s]
        batch_dists2 = batch_dists2[~batch_pad_0s][:, ~batch_pad_1s]
        matches = batch_matches_gt[~batch_pad_0s][:, ~batch_pad_1s]
            
        if loss_params['use_linear_sum']:
            row_inds, col_inds = linear_sum_assignment(batch_dists2)
            is_match = torch.zeros_like(matches)
            is_match[row_inds, col_inds] = 1.0
            is_match[batch_dists2 > loss_params['match_thresh']] = 0.0
        else:
            is_match = torch.zeros_like(matches)

            m0, m1 = batch_dists.argmax(1), batch_dists.argmax(0)
            max0_exp, _ = batch_dists.max(1), batch_dists.max(0)
            
            indices0 = np.arange(m0.shape[0])
            #indices1 = np.arange(m1.shape[0])
            mutual0 = indices0 == m1[m0]
            #mutual1 = indices1 == m0[m1]

            mscores0 = np.where(mutual0, max0_exp, np.zeros_like(max0_exp))
            #mscores1 = np.where(mutual1, mscores0[m1], np.zeros_like(max1_exp))
            valid0 = mutual0 & (mscores0 > loss_params['match_thresh'])
            #valid1 = mutual1 & valid0[m1]
            
            is_match[indices0[valid0], m0[valid0]] = 1.0

        if vis:
            vis_matches(is_match.cpu().numpy(), matches.cpu().numpy(), 
                        im_paths_0[ind], anno_paths_0[ind], det_inds_0[ind, ~batch_pad_0s].cpu().numpy(),
                        im_paths_1[ind], anno_paths_1[ind], det_inds_1[ind, ~batch_pad_1s].cpu().numpy(), 
                        vis_dir)

        true_pos = (matches * is_match).sum()
        false_pos = ((1 - matches) * is_match).sum()
        false_neg = matches.sum() - true_pos

        full_true_pos += true_pos
        full_false_pos += false_pos
        full_false_neg += false_neg

        if true_pos == 0:
            inst_prec = 0
            inst_rec = 0
            inst_f1 = 0
        else:
            inst_prec = true_pos / (true_pos + false_pos)
            inst_rec = true_pos / (true_pos + false_neg)
            inst_f1 = 2*inst_prec*inst_rec / (inst_prec + inst_rec)

        inst_precs.append(inst_prec)
        inst_recs.append(inst_rec)
        inst_f1s.append(inst_f1)

    if full_true_pos == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = full_true_pos / (full_true_pos + full_false_pos)
        recall = full_true_pos / (full_true_pos + full_false_neg)
        f1 = 2*precision*recall / (precision + recall)

    inst_precs = torch.as_tensor(inst_precs, dtype=matches_gt.dtype, device=matches_gt.device)
    inst_recs = torch.as_tensor(inst_recs, dtype=matches_gt.dtype, device=matches_gt.device)
    inst_f1s = torch.as_tensor(inst_f1s, dtype=matches_gt.dtype, device=matches_gt.device)

    return precision, recall, f1, inst_precs, inst_recs, inst_f1s

def get_bce_metrics(pred_confidences, gt_confidences, matches_gt, bce_thresh):
    pred_confidences = torch.sigmoid(pred_confidences)
    full_true_pos = gt_confidences[pred_confidences > bce_thresh].sum()
    full_false_pos = (1-gt_confidences[pred_confidences > bce_thresh]).sum()
    full_false_neg = matches_gt.sum() - full_true_pos

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
                 model_params,
                 include_bce,
                 lr,
                 weight_decay, 
                 scheduler,
                 vis=False,
                 vis_dir=None,
                 ):
        super().__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        
        self.loss_params = loss_params

        self.associator = FruitletAssociator(loss_params=loss_params, 
                                             include_bce=include_bce, 
                                             **model_params)
        
        self.include_bce = include_bce
        if include_bce:
            self.bce_loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            self.bce_loss_fn = None

        self.vis = vis
        self.vis_dir = vis_dir

        self.validation_prec_outputs = []
        self.validation_rec_outputs = []
        self.validation_f1_outputs = []

    def training_step(self, batch, batch_idx):
        _, fruitlet_ims_0, fruitlet_clouds_0, \
        is_pad_0, _, pos_2ds_0, \
        _, fruitlet_ims_1, fruitlet_clouds_1, \
        is_pad_1, _, pos_2ds_1, \
        matches_gt, masks_gt = batch
        
        data_0 = (fruitlet_ims_0, fruitlet_clouds_0, is_pad_0, pos_2ds_0)
        data_1 = (fruitlet_ims_1, fruitlet_clouds_1, is_pad_1, pos_2ds_1)

        enc_0, enc_1, sim, z0, z1, pred_confidences, gt_confidences = self.associator(data_0, data_1, matches_gt)

        match_loss, bce_loss, _, _ = get_loss(self.loss_params, enc_0, enc_1, 
                                           sim, z0, z1, is_pad_0, is_pad_1,
                                           matches_gt, masks_gt,
                                           pred_confidences, gt_confidences, 
                                           self.include_bce, self.bce_loss_fn, include_dist=False)

        loss = match_loss + bce_loss
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_match_loss", match_loss)

        if self.include_bce:
            self.log("train_bce_loss", bce_loss)

        return loss

    def validation_step(self, batch, batch_index):
        _, fruitlet_ims_0, fruitlet_clouds_0, \
        is_pad_0, _, pos_2ds_0, \
        _, fruitlet_ims_1, fruitlet_clouds_1, \
        is_pad_1, _, pos_2ds_1, \
        matches_gt, masks_gt = batch
        
        data_0 = (fruitlet_ims_0, fruitlet_clouds_0, is_pad_0, pos_2ds_0)
        data_1 = (fruitlet_ims_1, fruitlet_clouds_1, is_pad_1, pos_2ds_1)

        enc_0, enc_1, sim, z0, z1, pred_confidences, gt_confidences = self.associator(data_0, data_1, matches_gt)
        
        match_loss, bce_loss, dists, dists2 = get_loss(self.loss_params, enc_0, enc_1, 
                                           sim, z0, z1, is_pad_0, is_pad_1,
                                           matches_gt, masks_gt,
                                           pred_confidences, gt_confidences, 
                                           self.include_bce, self.bce_loss_fn)
        

        loss = match_loss + bce_loss
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_match_loss", match_loss)

        precision, recall, f1, inst_precs, inst_recs, inst_f1s  = get_loss_metrics(dists, dists2, is_pad_0, is_pad_1,
                                                 matches_gt, self.loss_params)
        
        self.log('precision', precision)
        self.log('recall', recall)
        self.log('f1', f1)

        if self.include_bce:
            self.log("val_bce_loss", bce_loss)

            bce_precision, bce_recall, bce_f1 = get_bce_metrics(pred_confidences, gt_confidences,
                                                                matches_gt, self.loss_params['bce_thresh'])
            
            self.log('bce_precision', bce_precision)
            self.log('bce_recall', bce_recall)
            self.log('bce_f1', bce_f1, prog_bar=True)
        
        if len(self.validation_prec_outputs) == 0:
            self.validation_prec_outputs = inst_precs
            self.validation_rec_outputs = inst_recs
            self.validation_f1_outputs = inst_f1s
        else:
            self.validation_prec_outputs = torch.concatenate((self.validation_prec_outputs, inst_precs))
            self.validation_rec_outputs = torch.concatenate((self.validation_rec_outputs, inst_recs))
            self.validation_f1_outputs = torch.concatenate((self.validation_f1_outputs, inst_f1s))
    
    def on_validation_epoch_end(self):
        self.log('inst_prec', torch.mean(self.validation_prec_outputs))
        self.log('inst_rec', torch.mean(self.validation_rec_outputs))
        self.log('inst_f1', torch.mean(self.validation_f1_outputs))
    
    def test_step(self, batch, batch_idx):
        _, fruitlet_ims_0, fruitlet_clouds_0, \
        is_pad_0, _, pos_2ds_0, \
        _, fruitlet_ims_1, fruitlet_clouds_1, \
        is_pad_1, _, pos_2ds_1, \
        matches_gt, masks_gt, \
        im_paths_0, anno_paths_0, det_inds_0, im_paths_1, anno_paths_1, det_inds_1 = batch
        
        data_0 = (fruitlet_ims_0, fruitlet_clouds_0, is_pad_0, pos_2ds_0)
        data_1 = (fruitlet_ims_1, fruitlet_clouds_1, is_pad_1, pos_2ds_1)

        enc_0, enc_1, sim, z0, z1, pred_confidences, gt_confidences = self.associator(data_0, data_1, matches_gt)
        
        _, _, dists, dists2 = get_loss(self.loss_params, enc_0, enc_1, 
                                           sim, z0, z1, is_pad_0, is_pad_1,
                                           matches_gt, masks_gt,
                                           pred_confidences, gt_confidences, 
                                           self.include_bce, self.bce_loss_fn)
        

        #loss = match_loss + bce_loss

        precision, recall, f1, inst_precs, inst_recs, inst_f1s = get_loss_metrics(dists, dists2, is_pad_0, is_pad_1,
                                                 matches_gt, self.loss_params,
                                                 self.vis, 
                                                 im_paths_0, anno_paths_0, det_inds_0, 
                                                 im_paths_1, anno_paths_1, det_inds_1, 
                                                 self.vis_dir)
        
        self.log('precision', precision)
        self.log('recall', recall)
        self.log('f1', f1)

        if self.include_bce:
            bce_precision, bce_recall, bce_f1 = get_bce_metrics(pred_confidences, gt_confidences,
                                                                matches_gt, self.loss_params['bce_thresh'])
            
            self.log('bce_precision', bce_precision)
            self.log('bce_recall', bce_recall)
            self.log('bce_f1', bce_f1)

        if len(self.validation_prec_outputs) == 0:
            self.validation_prec_outputs = inst_precs
            self.validation_rec_outputs = inst_recs
            self.validation_f1_outputs = inst_f1s
        else:
            self.validation_prec_outputs = torch.concatenate((self.validation_prec_outputs, inst_precs))
            self.validation_rec_outputs = torch.concatenate((self.validation_rec_outputs, inst_recs))
            self.validation_f1_outputs = torch.concatenate((self.validation_f1_outputs, inst_f1s))
        
    
    def on_test_epoch_end(self):
        self.log('inst_prec', torch.mean(self.validation_prec_outputs))
        self.log('inst_rec', torch.mean(self.validation_rec_outputs))
        self.log('inst_f1', torch.mean(self.validation_f1_outputs))
        
    def configure_optimizers(self):
        
        if self.weight_decay is None:
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        else:
            optimizer = optim.Adam(self.parameters(), 
                                   lr=self.lr,
                                   weight_decay=self.weight_decay)

        if self.scheduler is None:
            return optimizer
        

        if self.scheduler.type == 'lr_step':
            gamma = self.scheduler.gamma
            train_step = self.scheduler.train_step
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=train_step,
                                                        gamma=gamma) 
        elif self.scheduler.type == 'warmup_cosine':
            warmup_epochs = self.scheduler.warmup_epochs
            max_epochs = self.scheduler.max_epochs
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warmup_epochs, max_epochs=max_epochs)
        else:
            raise RuntimeError('Invalid scheduler type: ' + self.scheduler.type)

        return [optimizer], [scheduler]
    
        
