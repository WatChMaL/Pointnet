#primary training code

import numpy as np
import torch
import os.path as osp

from training_utils.engine_pointnet import EnginePointnet
from training_utils.find_top_models import find_top_models
from kaolin.models.PointNet2 import PointNet2Classifier as Pointnet2

from config.config_triumf_pointnet2_adam import config

if __name__ == '__main__':
    # Initialization
    model = Pointnet2(**config.model_kwargs) 
    engine = EnginePointnet(model, config)

    # Training
    engine.train()

    # Save network
    engine.save_state()

    #Validation
    models = find_top_models(engine.dirpath, 5)
    for model in models:
        engine.load_state(osp.join(engine.dirpath, model))
        engine.validate("validation", name=osp.splitext(model)[0])
