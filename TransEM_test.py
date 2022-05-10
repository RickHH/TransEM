import numpy as np
from BuildGeometry import BuildGeometry
from modellib import  Trainer, TransEM,dotstruct, PETMrDataset,Test,toNumpy
import os
import torch
from matplotlib import pyplot as plt
import scipy.io as sio



def Norm(img):
    batch = img.shape[0]
    channel = img.shape[1]
    H = img.shape[2]
    W = img.shape[3]
    img_r = img.reshape(batch,-1)
    max_value,_ = img_r.max(1)
    min_value,_ = img_r.min(1)
    max_value = max_value.unsqueeze(1).repeat(1,H*W)
    min_value = min_value.unsqueeze(1).repeat(1,H*W)
    out = (img_r-min_value)/(max_value-min_value)
    return out.reshape(batch,channel,H,W)

save_training_dir = '...' # training_dataset_dir

g = dotstruct()
g.is3d = False
g.temPath = '...'
g.radialBinCropFactor = 0.5
g.psf_cm = 0.4
g.niters = 8
g.nsubs = 6
g.training_flname = [save_training_dir+os.sep,'data-']
g.save_dir = '...'+os.sep
g.device = torch.device("cuda:0")
g.num_workers = 0
g.batch_size = 50
g.test_size = 1
g.valid_size = 0
g.num_train = 100
g.depth =1
g.in_channels = 1
g.lr = 1e-5
g.epochs = 22
g.crop_factor = 0.3
g.shuffle = False
PET = BuildGeometry('mmr',g.radialBinCropFactor)
PET.loadSystemMatrix(g.temPath,is3d=False )
train_loader, valid_loader, test_loader = PETMrDataset(g.training_flname, num_train=g.num_train, is3d=g.is3d, \
                                                       batch_size=g.batch_size, test_size=g.test_size, valid_size=g.valid_size, num_workers = g.num_workers,shuffle=g.shuffle)
model_flname = '...'


num_bathces = len(test_loader)

i=0
for sinoLD, sinoHD, imgHD, AN, RS,imgLD, imgLD_psf, mrImg, counts, imgGT,index in test_loader:
    print('start recon batch {}'.format(i))
    AN = toNumpy(AN)
    sinoLD_n = toNumpy(sinoLD)
    print('start TransEM recon')
    img_TransEM = Test(model_flname, PET, sinoLD_n, AN, mrImg, niters=10, nsubs =6)
    print('start mapem recon')
    sinoLD = sinoLD.squeeze(1)
    sinoLD,imgLD,imgHD = toNumpy(sinoLD),toNumpy(imgLD),toNumpy(imgHD)
    img_mapem = PET.MAPEM2DBatch(sinoLD,AN,imgHD,beta=0.005,niters = 10, nsubs=6,psf=0.4)
    print('start OSEM recon')
    img_osem = PET.OSEM2D(sinoLD,None,None,AN,None,niter = 10, nsubs =6, tof = False, psf = 0.4)
    if i == 0:
        img_TransEm_total = img_TransEM
        img_mapem_total = img_mapem
        imgGT_total = imgGT
        imgHD_total = imgHD
        imgLD_total = imgLD
        img_osem_total = img_osem
    else:
        img_TransEm_total = np.concatenate((img_TransEm_total,img_TransEM),axis=0)
        img_mapem_total = np.concatenate((img_mapem_total,img_mapem),axis=0)
        imgGT_total = np.concatenate((imgGT_total,imgGT),axis=0)
        imgHD_total = np.concatenate((imgHD_total,imgHD),axis=0)
        imgLD_total = np.concatenate((imgLD_total,imgLD),axis=0)
        img_osem_total = np.concatenate((img_osem_total,img_osem),axis=0)

    i=i+1

