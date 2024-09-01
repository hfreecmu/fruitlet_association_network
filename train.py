import argparse
import os
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from data.dataset import AssociationDataset
from model.lightning_associator import LightningAssociator
from util.util import load_cfg, get_checkpoint_path

def train(train_params, 
          val_params, 
          model_params, 
          loss_params,
          exp_name,
          include_bce,
          batch_size,
          checkpoint_dir,
          checkpoint_metrics,
          **kwargs):
    
    train_dataset = AssociationDataset(**train_params)
    val_dataset = AssociationDataset(**val_params)

    print('Len train dataset: ', len(train_dataset))
    print('Len val dataset: ', len(val_dataset))

    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=batch_size,
                              shuffle=True)
    
    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=batch_size,
                            shuffle=False)

    if not include_bce:
        model = LightningAssociator(loss_params=loss_params,
                                    include_bce=include_bce,
                                    model_params=model_params,
                                    lr=train_params['lr'],
                                    weight_decay=train_params['weight_decay'],
                                    scheduler=train_params['scheduler'],
                                    )
    else:
        prev_exp_name = exp_name
        exp_name = exp_name + '_bce'
        checkpoint_path = get_checkpoint_path(checkpoint_dir, prev_exp_name, 
                                              checkpoint_metrics)
        
        #TODO alternatively use this below to restore everything 
        # automatically restores model, epoch, step, LR schedulers, apex, etc...
        #trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")
        #https://pytorch-lightning.readthedocs.io/en/1.6.5/common/checkpointing.html
        model = LightningAssociator.load_from_checkpoint(
                                    checkpoint_path, 
                                    loss_params=loss_params,
                                    include_bce=include_bce,
                                    model_params=model_params,
                                    lr=train_params['lr'],
                                    weight_decay=train_params['weight_decay'],
                                    gamma=train_params['gamma'],
                                    train_step=train_params['train_step'],
                                    )
    
    filename_val = '{epoch:02d}-{val_loss:.2f}'
    checkpoint_callback_val = ModelCheckpoint(save_top_k=2, monitor="val_loss", filename=filename_val)
    filename_f1 = '{epoch:02d}-{f1:.2f}'
    checkpoint_callback_f1 = ModelCheckpoint(save_top_k=2, monitor="f1", mode='max', filename=filename_f1)

    logger = TensorBoardLogger(save_dir=os.getcwd(), version=exp_name)
    
    trainer = L.Trainer(max_epochs=train_params['num_epochs'], 
                        callbacks=[checkpoint_callback_val, checkpoint_callback_f1],
                        logger=logger, log_every_n_steps=5,
                        accelerator="gpu",
                        devices=-1,
                        strategy="ddp",
                        #num_nodes=?
                        )

    trainer.fit(model=model, 
                train_dataloaders=train_loader, 
                val_dataloaders=val_loader)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_path')
    parser.add_argument('--include_bce', action='store_true')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    cfg_path = args.cfg_path
    include_bce = args.include_bce

    cfg = load_cfg(cfg_path)
    cfg['include_bce'] = include_bce

    print(OmegaConf.to_yaml(cfg))
    train(**cfg)
