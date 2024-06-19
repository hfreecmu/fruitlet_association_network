from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import lightning as L

from data.dataset import AssociationDataset
from model.lightning_associator import LightningAssociator
from util.util import get_checkpoint_path

def test(test_params, 
         model_params, 
         loss_params, 
         exp_name, 
         checkpoint_dir,
         **kwargs):

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
    trainer.test(model=model, dataloaders=test_loader)

if __name__ == "__main__":
    cfg = OmegaConf.load('cfg/v0_full.yml')
    print(OmegaConf.to_yaml(cfg))
    test(**cfg)
