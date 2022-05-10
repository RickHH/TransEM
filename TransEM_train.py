import numpy as np
from BuildGeometry import BuildGeometry
from modellib import  Trainer, TransEM,dotstruct, PETMrDataset
import os
import torch


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

save_training_dir = '...' # training_dataset_dir

g = dotstruct()
g.is3d = False
g.temPath = '...' #system matrix path
g.radialBinCropFactor = 0.5
g.psf_cm = 0.25
g.niters = 8
g.nsubs = 6
g.training_flname = [save_training_dir+os.sep,'data-']
g.save_dir = '...'+os.sep
g.device = torch.device("cuda:0")
g.num_workers = 0
g.batch_size = 4
g.test_size = 0.1
g.valid_size = 0.1
g.num_train = 100
g.depth =1
g.in_channels = 1 # with or without mrImg
g.lr = 1e-5
g.epochs = 100
g.model_name = 'TransEM'
g.save_from_epoch = 0
g.crop_factor = 0.3
g.do_validation = True


# build PET object
PET = BuildGeometry('mmr',g.radialBinCropFactor)
PET.loadSystemMatrix(g.temPath,is3d=False )

# load dataloaders
train_loader, valid_loader, test_loader = PETMrDataset(g.training_flname, num_train=g.num_train, is3d=g.is3d, \
                                                       batch_size=g.batch_size, test_size=g.test_size, valid_size=g.valid_size, num_workers = g.num_workers)

# build model
model = TransEM(g.depth, g.num_kernels, g.kernel_size, g.in_channels, g.is3d, g.reg_ccn_model).to(g.device, dtype=torch.float32)

# train
Trainer(PET,model, g, train_loader, valid_loader)