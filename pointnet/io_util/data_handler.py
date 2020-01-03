from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from io_util.dataset import WChPointnetDataset, WChPointnetDataset_trainval
import numpy as np 

def get_loaders(path, cols_to_include, indices_file, batch_size, workers):

	dataset = WChPointnetDataset_trainval(path, cols_to_include, indices_file)

	train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
							  pin_memory=True, sampler=SubsetRandomSampler(dataset.train_indices))

	val_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
							  pin_memory=True, sampler=SubsetRandomSampler(dataset.val_indices))

	return train_loader, val_loader, dataset