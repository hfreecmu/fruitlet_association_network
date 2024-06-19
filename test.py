from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import lightning as L

from data.dataset import AssociationDataset
from model.lightning_associator import LightningAssociator

def test(test_params, model_params, loss_params, exp_name, **kwargs):

    test_dataset = AssociationDataset(**test_params)
    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=test_params['batch_size'],
                            shuffle=False)

    model = LightningAssociator.load_from_checkpoint(test_params['checkpoint'], 
                                                     loss_params=loss_params, **model_params)
    trainer = L.Trainer(enable_checkpointing=False, logger=False)
    trainer.test(model=model, dataloaders=test_loader)

if __name__ == "__main__":
    cfg = OmegaConf.load('cfg/v0.yml')
    print(OmegaConf.to_yaml(cfg))
    test(**cfg)