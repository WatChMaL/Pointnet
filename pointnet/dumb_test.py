from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from io_util.dataset import WChPointnetDataset
from io_util.dataset2 import WChPointnetDataset as DSet2
import numpy as np

path = "/fast_scratch/WatChMaL/data/pointnet/splits/pointnet_trainval.h5"
cols_to_include = [0,1,2,3]
batch_size = 32
indices_file = "/fast_scratch/WatChMaL/data/pointnet/splits/pointnet_trainval_idxs.npz"
workers = 0
device = 'cuda:7'

dataset = WChPointnetDataset(path, cols_to_include, device=device)
dataset2 = DSet2(path, cols_to_include, device=device)

for data, label in dataset:
    print(data)
    break

for data, label in dataset2:
    print(data)
    break



