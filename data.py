import os.path
import numpy as np
from typing import Any, Callable, Optional, Tuple

import ignite.distributed as idist
import torchvision.transforms as T

from torchvision.datasets import ImageFolder


def setup_data(config: Any):
    """Download datasets and create dataloaders

    Parameters
    ----------
    config: needs to contain `data_path`, `train_batch_size`, `eval_batch_size`, and `num_workers`
    """
    transform = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset_train = ImageFolder(
        root=os.path.join(config.data_path, 'train'),
        transform=transform,
    )
    
    dataset_eval = ImageFolder(
        root=os.path.join(config.data_path, 'test'),
        transform=transform,
    )

    dataloader_train = idist.auto_dataloader(
        dataset_train,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    
    dataloader_eval = idist.auto_dataloader(
        dataset_eval,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return dataloader_train, dataloader_eval
