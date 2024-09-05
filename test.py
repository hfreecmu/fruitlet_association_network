import argparse
import os
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
#import uuid
import lightning as L

from data.dataset import AssociationDataset
from data.icp_dataset import ICPAssociationDataset
from model.lightning_associator import LightningAssociator
from model.icp_associator import ICPAssociator
from model.pheno_lightning_associator import PhenoLightningAssociator
from data.pheno_dataset import PhenoDataset
from model.bonn_associator import BonnAssociator
from util.util import get_checkpoint_path, load_cfg

def test(is_pheno,
         test_params, 
         model_params, 
         loss_params, 
         exp_name, 
         include_bce,
         batch_size,
         checkpoint_dir,
         checkpoint_metrics,
         **kwargs):
    
    if not is_pheno:
        data_class = AssociationDataset
        model_class = LightningAssociator
    else:
        data_class = PhenoDataset
        model_class = PhenoLightningAssociator

    if test_params['use_icp'] and test_params['use_bonn']:
        raise RuntimeError('cannot icp and bonn')

    if test_params['use_icp']:
        test_dataset = ICPAssociationDataset(**test_params)
        test_loader = DataLoader(dataset=test_dataset, 
                                 batch_size=1,
                                 shuffle=False)
        model = ICPAssociator()
        trainer = L.Trainer(enable_checkpointing=False, 
                            logger=False,
                            accelerator="cpu")
    elif test_params['use_bonn']:
        test_dataset = ICPAssociationDataset(**test_params)
        test_loader = DataLoader(dataset=test_dataset, 
                                 batch_size=1,
                                 shuffle=False)
        model = BonnAssociator(**test_params)
        trainer = L.Trainer(enable_checkpointing=False, 
                            logger=False,
                            accelerator="cpu")
    else:
        test_dataset = data_class(**test_params)
        test_loader = model_class(dataset=test_dataset, 
                                 batch_size=batch_size,
                                 shuffle=False)
        
        if include_bce:
            exp_name = exp_name + '_bce'

        checkpoint_path = get_checkpoint_path(checkpoint_dir, exp_name, 
                                              checkpoint_metrics)
    
        print('Using checkpoint: ' + checkpoint_path)

        model = LightningAssociator.load_from_checkpoint(checkpoint_path,
                                                         vis=test_params['vis'],
                                                         vis_dir=test_params['vis_dir'])
        
        trainer = L.Trainer(enable_checkpointing=False, logger=False)
        

    trainer.test(model=model, dataloaders=test_loader)

    # if test_params['use_icp']:
    #     error_files = model.errors
    #     #identifier = uuid.uuid4().hex[0:6]
    #     identifier = test_params['anno_subdir']
    #     output_path = os.path.join(checkpoint_dir, f'error_files_{identifier}.json')
    #     write_json(output_path, error_files)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_path')
    parser.add_argument('--include_bce', action='store_true')
    parser.add_argument('--use_icp', action='store_true')
    parser.add_argument('--use_bonn', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--vis_dir', default=None)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    cfg_path = args.cfg_path
    include_bce = args.include_bce
    use_icp = args.use_icp
    use_bonn = args.use_bonn
    vis = args.vis
    vis_dir = args.vis_dir

    cfg = load_cfg(cfg_path)
    cfg['include_bce'] = include_bce
    cfg['test_params']['use_icp'] = use_icp
    cfg['test_params']['use_bonn'] = use_bonn
    cfg['test_params']['vis'] = vis
    cfg['test_params']['vis_dir'] = vis_dir

    test(**cfg)
