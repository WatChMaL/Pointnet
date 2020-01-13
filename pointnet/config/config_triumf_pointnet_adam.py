from config.easy_dict import EasyDict

config = EasyDict()

config.model_name = "Pointnet"

config.cols_to_use = [0,1,2,3]
# may want to change: feat_size, layer_dims, etc. 
config.model_kwargs = {"in_channels": len(config.cols_to_use), 
					   "num_classes": 3}
config.data_path = "/data/WatChMaL/data/pointnet/splits/pointnet_trainval.h5"
config.indices_file = "/data/WatChMaL/data/pointnet/splits/pointnet_trainval_idxs.npz"

#make sure to change this
config.dump_path = "/home/dgreen/training_outputs/pointnet/no_time/adam/"

config.num_data_workers = 0 # Sometime crashes if we do multiprocessing
config.device = 'gpu'
config.gpu_list = [6]

config.optimizer = "Adam"
config.optimizer_kwargs = {"lr":1e-3, "betas": (0.9, 0.999)}

config.use_scheduler = False
config.scheduler_kwargs = {"mode":"min", "min_lr":1e-6, "patience":1, "verbose":True}
config.scheduler_step = 190

config.batch_size = 32
config.epochs = 10

config.report_interval = 50
config.num_val_batches  = 128
config.valid_interval   = 10000

config.validate_batch_size = 32
config.validate_dump_interval = 256
