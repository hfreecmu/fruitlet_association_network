from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from data.dataset import AssociationDataset
from model.lightning_associator import LightningAssociator

def train(train_params, 
          val_params, 
          model_params, 
          loss_params,
          exp_name,
          **kwargs):
    train_dataset = AssociationDataset(**train_params)
    val_dataset = AssociationDataset(**val_params)

    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=train_params['batch_size'],
                              shuffle=True)
    
    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=val_params['batch_size'],
                            shuffle=False)
    
    model = LightningAssociator(lr=train_params['lr'],
                                weight_decay=train_params['weight_decay'],
                                loss_params=loss_params,
                                **model_params)
    
    filename = exp_name + '/lighsample-mnist_{epoch:02d}-{val_loss:.2f}'
    checkpoint_callback = ModelCheckpoint(save_top_k=2, monitor="val_loss", filename=filename)

    trainer = L.Trainer(max_epochs=train_params['num_epochs'], 
                        callbacks=[checkpoint_callback],
                        log_every_n_steps=5)

    trainer.fit(model=model, 
                train_dataloaders=train_loader, 
                val_dataloaders=val_loader)

if __name__ == "__main__":
    cfg = OmegaConf.load('cfg/v0.yml')
    print(OmegaConf.to_yaml(cfg))
    train(**cfg)