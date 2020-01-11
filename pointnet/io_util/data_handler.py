from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from io_util.dataset import WChPointnetDataset
import numpy as np 

def get_loaders(path, cols_to_include, indices_file, batch_size, workers, scale=True):
    """
    Returns training and validation dataloaders according to indices specified in indices file.
    cols_to_include: columns of training data to be used in a list (i.e. [0 1 2] for x,y,z data)
    indices file: file which specifies training/validation indices
    batch_size: number of examples (pointclouds) per batch
    workers: number of threads for dataloader
    scale: if True, scale position values by 1/100
    """
    
    dataset = WChPointnetDataset(path, cols_to_include, scale)

    all_indices = np.load(indices_file)
    train_indices = all_indices["train_idxs"]
    val_indices = all_indices["val_idxs"]


    # change pin_memory back to true
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                              pin_memory=True, sampler=SubsetRandomSampler(train_indices))

    val_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                              pin_memory=True, sampler=SubsetRandomSampler(val_indices))

    
    return train_loader, val_loader, dataset
