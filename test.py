import os
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
#import uuid
import lightning as L

from data.dataset import AssociationDataset
from data.icp_dataset import ICPAssociationDataset
from model.lightning_associator import LightningAssociator
from model.icp_associator import ICPAssociator
from util.util import get_checkpoint_path, write_json

def test(test_params, 
         model_params, 
         loss_params, 
         exp_name, 
         checkpoint_dir,
         **kwargs):

    if not test_params['use_icp']:
        test_dataset = AssociationDataset(**test_params)
        test_loader = DataLoader(dataset=test_dataset, 
                                 batch_size=test_params['batch_size'],
                                 shuffle=False)

        checkpoint_path = get_checkpoint_path(checkpoint_dir, exp_name, 
                                              test_params['checkpoint_metrics'])
    
        print('Using checkpoint: ' + checkpoint_path)


        model = LightningAssociator.load_from_checkpoint(checkpoint_path, 
                                                         loss_params=loss_params, 
                                                         match_thresh=test_params['match_thresh'],
                                                         **model_params)
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

    if test_params['use_icp']:
        error_files = model.errors
        #identifier = uuid.uuid4().hex[0:6]
        identifier = test_params['anno_subdir']
        output_path = os.path.join(checkpoint_dir, f'error_files_{identifier}.json')
        write_json(output_path, error_files)

if __name__ == "__main__":
    cfg = OmegaConf.load('cfg/v0_overfit.yml')
    print(OmegaConf.to_yaml(cfg))
    test(**cfg)
