from config.easy_dict import EasyDict

config = EasyDict()

config.cols_to_use = [0,1,2,3]
# may want to change: feat_size, layer_dims, etc. 
config.model_kwargs = {"in_channels": len(config.cols_to_use), 
					   "num_classes": 3}

config.data_path = "/data/WatChMaL/data/pointnet/splits/pointnet_trainval.h5"
config.indices_file = "/data/WatChMaL/data/pointnet/splits/pointnet_trainval_idxs.npz"

config.dump_path = "/home/dgreen/training_outputs"

config.num_data_workers = 0 # Sometime crashes if we do multiprocessing
config.device = 'gpu'
config.gpu_list = [7]

config.optimizer = "SGD"
config.optimizer_kwargs = {"lr":0.01, "weight_decay":1e-3, "momentum":0.9, "nesterov":True}

config.scheduler_kwargs = {"mode":"min", "min_lr":1e-6, "patience":1, "verbose":True}
config.scheduler_step = 190

config.batch_size = 32
config.epochs = 10

config.report_interval = 50
config.num_val_batches  = 32
config.valid_interval   = 200

config.validate_batch_size = 32
config.validate_dump_interval = 256