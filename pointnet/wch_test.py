import torch
from torch.utils.data import DataLoader
import kaolin as kal
from kaolin import ClassificationEngine
from kaolin.models.PointNet import PointNetClassifier as PointNet
import kaolin.transforms as tfs
import sys
from io_util.data_handler_kn import get_loaders

path = "/fast_scratch/WatChMaL/data/pointnet/splits/pointnet_trainval.h5"
cols = [0,1,2,3]
batch_size = 32
indices_file = "/fast_scratch/WatChMaL/data/pointnet/splits/pointnet_trainval_idxs.npz"
workers = 0
device = 'cuda:6'

train_loader, val_loader, dataset = get_loaders(path, cols, indices_file, batch_size, workers, device=device)

engine = ClassificationEngine(PointNet(num_classes=3, in_channels=4),
                              train_loader, val_loader, device=device)

engine.fit()
