#primary training code

import numpy as np
import torch
import os.path as osp

from training_utils.engine_pointnet import EnginePointnet
from training_utils.find_top_models import find_top_models
from kaolin.models.PointNet import PointNetClassifier as Pointnet

from config.config_triumf_pointnet_adam import config

if __name__ == '__main__':
    # Initialization
    model = Pointnet(**config.model_kwargs) 
    engine = EnginePointnet(model, config)

    # Training
    engine.train()

    # Save network
    engine.save_state()

