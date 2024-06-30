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
from util.util import get_checkpoint_path, write_json, load_cfg

def test(test_params, 
         model_params, 
         loss_params, 
         exp_name, 
         include_bce,
         batch_size,
         checkpoint_dir,
         checkpoint_metrics,
         **kwargs):

    if not test_params['use_icp']:
        test_dataset = AssociationDataset(**test_params)
        test_loader = DataLoader(dataset=test_dataset, 
                                 batch_size=batch_size,
                                 shuffle=False)
        
        if include_bce:
            exp_name = exp_name + '_bce'

        checkpoint_path = get_checkpoint_path(checkpoint_dir, exp_name, 
                                              checkpoint_metrics)
    
        print('Using checkpoint: ' + checkpoint_path)

        model = LightningAssociator.load_from_checkpoint(checkpoint_path, 
                                                         loss_params=loss_params,
                                                         include_bce=include_bce, 
                                                         model_params=model_params,
                                                         )
        trainer = L.Trainer(enable_checkpointing=False, logger=False)

    else:
        test_dataset = ICPAssociationDataset(**test_params)
        test_loader = DataLoader(dataset=test_dataset, 
                                 batch_size=1,
                                 shuffle=False)
        model = ICPAssociator()
        trainer = L.Trainer(enable_checkpointing=False, 
                            logger=False,
                            accelerator="cpu")

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

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    cfg_path = args.cfg_path
    include_bce = args.include_bce
    use_icp = args.use_icp

    cfg = load_cfg(cfg_path)
    cfg['include_bce'] = include_bce
    cfg['test_params']['use_icp'] = use_icp

    test(**cfg)
