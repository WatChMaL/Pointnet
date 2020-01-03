import torch
import h5py
import numpy as np

from torch.utils.data import Dataset, DataLoader

class WChPointnetDataset(Dataset):

	def __init__(self, path, cols_to_include):
		"""
		Pointnet dataset object for WatChMaL data.
		path: location of hdf5 file
		cols_to_include: list containing index numbers of which columns to use. 
		"""
		self.cols_to_include

		f = h5py.File(path, 'r')
		hdf5_event_data = f["event_data"]
		hdf5_labels = f["labels"]

		assert hdf5_event_data.shape[0] == hdf5_labels.shape[0]

		event_data_shape = hdf5_event_data.shape
        event_data_offset = hdf5_event_data.id.get_offset()
        event_data_dtype = hdf5_event_data.dtype

        #this creates a memory map - i.e. events are not loaded in memory here
        #only on get_item
        self.point_clouds = np.memmap(path, mode='r', shape=event_data_shape,
                                    offset=event_data_offset, dtype=event_data_dtype)
        self.labels = np.array(hdf5_labels)

    def get(self, idx):
    	x = torch.from_numpy(self.point_clouds[idx][:, self.cols_to_include])
    	y = torch.tensor([self.labels[idx]], dtype=torch.int64)

    	return x, y

   	def __len__(self):
   		return self.labels.shape[0]


class WChPointnetDataset_trainval(WCH5Dataset):

    def __init__(self, path, cols_to_include, indices_file):

        super(WChPointnetDataset_trainval, self).__init__(path, cols_to_include)

        all_indices = np.load(indices_file)

        self.train_indices = all_indices["train_idxs"]
        self.val_indices = all_indices["val_idxs"]
