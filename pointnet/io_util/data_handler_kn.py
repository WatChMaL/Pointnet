from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from io_util.dataset2_kn import WChPointnetDataset
import numpy as np 

def get_loaders(path, cols_to_include, indices_file, batch_size, workers, device):
    
    all_indices = np.load(indices_file)
    train_indices = all_indices["train_idxs"]
    val_indices = all_indices["val_idxs"]

    dataset = WChPointnetDataset(path, cols_to_include, train_indices, val_indices, device=device)


    # change pin_memory back to true
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                              pin_memory=False, sampler=SubsetRandomSampler(train_indices))

    val_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                              pin_memory=False, sampler=SubsetRandomSampler(val_indices))

    
    return train_loader, val_loader, dataset
