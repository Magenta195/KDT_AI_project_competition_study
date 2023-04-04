'''
File containing various utility functions for Pytorch model training.
'''

import torch
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

from typing import Dict, List, Tuple
from datetime import datetime
import os

def save_model(
    model: torch.nn.Module,
    target_dir: str,
    model_name: str
):
    '''Saves a Pytorch model to a target directory.

    Args:
        model: A target Pytorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include either '.pth' or '.pt' as the file extension.

    Example usage:
        save_model(
            model=model_0,
            target_dir='models',
            model_name='some_model_name.pth'
        )
    '''

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith('.pth') or model_name.endswith('.pt'), "model_name should end with '.pt' or '.pth'."
    model_save_path = target_dir_path / model_name

    print(f'[INFO] Saving model to: {model_save_path}')
    torch.save(
        obj=model.state_dict(),
        f=model_save_path
    )

def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str = None):
    timestamp = datetime.now().strftime('%Y-%m-%d')

    # when using colab: /content/drive/MyDrive/Colab Notebooks/Paper Replicating/runs
    # else: ../runs
    if extra:
        log_dir = os.path.join('../runs', timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join('../runs', timestamp, experiment_name, model_name)
    print(f'[INFO] Creating SummaryWriter saving to {log_dir}')

    return SummaryWriter(log_dir=log_dir)
