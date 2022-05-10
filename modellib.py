import torch
import torch.nn as nn
import numpy as np
from RSTR import RSTR

from torch.utils.data import Dataset
from numpy import load, ceil

class DatasetPetMr_v2(Dataset):
    def __init__(self, filename, num_train, transform=None, target_transform=None, is3d=False, imgLD_flname = None,crop_factor = 0, allow_pickle=True):
        """
        filename = ['save_dir,'prefix']
        num_train =number of traning datasets
        set "has_gtruth=False" for invivo data
        """
        self.transform = transform
        self.target_transform=target_transform
        self.is3d = is3d
        self.filename = filename
        self.num_train = num_train
        self.imgLD_flname = imgLD_flname
        self.crop_factor = crop_factor
        self.allow_pickle =allow_pickle
        
    def crop_sino(self,sino):
         if self.crop_factor!=0:
              i =  int(ceil(sino.shape[0]*self.crop_factor/2.0)*2)//2
              sinOut = sino[i:sino.shape[0]-i]
         else:
              sinOut = sino
         return sinOut
    def crop_img(self,img):
         if self.crop_factor!=0:
              i =  int(ceil(img.shape[0]*self.crop_factor/2.0)*2)//2    
              imgOut = img[i:img.shape[0]-i, i:img.shape[1]-i]
         else:
              imgOut = img
         return imgOut
       
    def __len__(self):
        return self.num_train
   
    def __getitem__(self, index):
        dset = load(self.filename[0]+self.filename[1]+str(index)+'.npy',allow_pickle=self.allow_pickle).item()
        
        sinoLD =  self.crop_sino(dset['sinoLD'])
        sinoHD = self.crop_sino(dset['sinoHD'])
        AN = self.crop_sino(dset['AN'])
        imgHD = self.crop_img(dset['imgHD'])
        mrImg = self.crop_img(dset['mrImg'])
        counts = dset['counts']

        if 'RS' in dset and type(dset['RS'])!=list:
             RS = self.crop_sino(dset['RS'])
        else:
             RS = 0        
        if 'imgGT' in dset and type(dset['imgGT'])!=list:
             imgGT = self.crop_img(dset['imgGT'])
        else:
             imgGT = 0
        if 'imgLD' in dset and type(dset['imgLD'])!=list:
            imgLD = self.crop_img(dset['imgLD'])
        elif self.imgLD_flname is not None:
             dset = load(self.imgLD_flname[0]+self.imgLD_flname[1]+str(index)+'.npy').item()
             imgLD = self.crop_img(dset['imgLD'])
        else:
             imgLD = 0
        if 'imgLD_psf' in dset  and type(dset['imgLD_psf'])!=list:
            imgLD_psf = self.crop_img(dset['imgLD_psf'])
        elif self.imgLD_flname is not None:
             dset = load(self.imgLD_flname[0]+self.imgLD_flname[1]+str(index)+'.npy').item()
             imgLD_psf = self.crop_img(dset['imgLD_psf'])
        else:
             imgLD_psf = 0
        if self.transform is not None:
            sinoLD = self.transform(sinoLD)
            sinoHD = self.transform(sinoHD)
            AN = self.transform(AN)   
            if not np.isscalar(RS):
                 RS = self.transform(RS) 
        if self.target_transform is not None:
            imgHD = self.target_transform(imgHD)
            mrImg = self.target_transform(mrImg)
            if not np.isscalar(imgLD):
                 imgLD = self.target_transform(imgLD)
            if not np.isscalar(imgLD_psf):
                 imgLD_psf = self.target_transform(imgLD_psf)
            if not np.isscalar(imgGT):
                 imgGT = self.target_transform(imgGT)
     #    return sinoLD, imgHD, AN, RS,imgLD, imgLD_psf, mrImg, counts, imgGT,index #simu return
        return sinoLD, sinoHD,imgHD, AN, RS,imgLD, imgLD_psf, mrImg, counts, imgGT,index
 
def train_test_split(dset, num_train, batch_size, test_size, valid_size=0, num_workers = 0, shuffle=True):
    from torch.utils.data.sampler import SubsetRandomSampler
    from torch.utils.data import DataLoader
    from numpy import floor, random
    
    random.seed(1)
    
    indx = list(range(num_train))
    if shuffle:
        random.shuffle(indx)
    split = int(floor(num_train*(test_size)))
    train_idx,test_idx = indx[split:],indx[:split]
    
    valid_loader = None
    valid_idx = None
    if valid_size:
        if shuffle:
            random.shuffle(train_idx)
        split = int(floor(len(train_idx)*valid_size))
        train_idx,valid_idx = train_idx[split:],train_idx[:split]
        valid_sampler = SubsetRandomSampler(valid_idx)
        valid_loader = DataLoader(dset,batch_size = batch_size, sampler = valid_sampler, num_workers = num_workers, pin_memory=False)
  
    test_sampler = SubsetRandomSampler(test_idx)
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = DataLoader(dset,batch_size = batch_size, sampler = train_sampler, num_workers = num_workers, pin_memory=False)
    test_loader = DataLoader(dset,batch_size = batch_size, sampler = test_sampler, num_workers = num_workers,pin_memory=False)
 
    return train_loader, test_loader, valid_loader#, train_idx, test_idx, valid_idx

def PETMrDataset(filename, num_train, batch_size, test_size, valid_size=0, num_workers = 0, \
                 transform=None, target_transform=None, is3d=False, imgLD_flname = None,   shuffle=True, crop_factor = 0):
 
     dset = DatasetPetMr_v2(filename,num_train, transform, target_transform, is3d, imgLD_flname,crop_factor)
     train_loader, test_loader, valid_loader = train_test_split(dset, num_train, batch_size, test_size, valid_size, num_workers, shuffle)
     return train_loader, valid_loader, test_loader

def crop(img, crop_factor=0, is3d = False): 
    if crop_factor!=0:
         from torch import is_tensor
         from numpy import ceil
         isTensor = is_tensor(img)
         round_int = lambda x: int(ceil(x/2.0)*2)
         if isTensor:
             i = round_int(img.shape[2]*(crop_factor))//2 
             j = round_int(img.shape[3]*(crop_factor))//2 
             imgOut = img[:,:,i:img.shape[2]-i, j:img.shape[3]-j] 
         else:
              if img.ndim==4 or (img.ndim==3 and not is3d):
                   i = round_int(img.shape[1]*(crop_factor))//2
                   j = round_int(img.shape[2]*(crop_factor))//2
                   imgOut = img[:,i:img.shape[1]-i, j:img.shape[2]-j]
              elif img.ndim==2 or (img.ndim==3 and is3d):
                   i = round_int(img.shape[0]*(crop_factor))//2
                   j = round_int(img.shape[1]*(crop_factor))//2
                   imgOut = img[i:img.shape[0]-i, j:img.shape[1]-j]
    else: 
        imgOut = img 
    return imgOut  

def uncrop(img,W,H=None,is3d = False):
    from torch import is_tensor
    
    if H is None: H = W
    if is_tensor(img):
         if (img.shape[2]!=W or img.shape[3]!=H):
              from torch import zeros
              i = (W - img.shape[2])//2 
              j = (H - img.shape[3])//2 
              dims = [img.shape[0],img.shape[1],W,H]
              if img.dim()==5: dims.append(img.shape[4])
              imgOut = zeros(dims,dtype=img.dtype,device=img.device)
              imgOut[:,:,i:W-i, j:H-j] = img
         else:
             imgOut = img 
    else: 
         from numpy import zeros
         if img.ndim==4 and (img.shape[1]!=W or img.shape[2]!=H): #(nBatch,W,H,D)
              i = (W - img.shape[1])//2 
              j = (H - img.shape[2])//2 
              imgOut = zeros((img.shape[0],W,H,img.shape[3]),dtype=img.dtype)
              imgOut[:,i:W-i, j:H-j,:] = img
         elif (img.ndim==3 and not is3d) and (img.shape[1]!=W or img.shape[2]!=H): #(nBatch,W,H)
              i = (W - img.shape[1])//2 
              j = (H - img.shape[2])//2 
              imgOut = zeros((img.shape[0],W,H),dtype=img.dtype)
              imgOut[:,i:W-i, j:H-j] = img 
         elif (img.ndim==3 and is3d) and (img.shape[0]!=W or img.shape[1]!=H): #(W,H,D)
              i = (W - img.shape[0])//2 
              j = (H - img.shape[1])//2 
              imgOut = zeros((W,H,img.shape[2]),dtype=img.dtype)
              imgOut[i:W-i, j:H-j,:] = img 
         elif img.ndim==2  and (img.shape[0]!=W or img.shape[1]!=H): #(W,H)
              i = (W - img.shape[0])//2 
              j = (H - img.shape[1])//2 
              imgOut = zeros((W,H),dtype=img.dtype)
              imgOut[i:W-i, j:H-j] = img  
         else:
              imgOut = img 
    return imgOut 

def toNumpy(x):
    return x.detach().cpu().numpy().astype('float32')


def zeroNanInfs(x):
     from torch import is_tensor,isnan,isinf, Tensor,float
     if is_tensor(x):
          x.data[isnan(x)]= Tensor([0]).to('cuda:0',dtype=float)
          x.data[isinf(x)]= Tensor([0]).to('cuda:0',dtype=float)
     else:
          x[np.isnan(x)]=0
          x[np.isinf(x)]=0
     return x

class dotstruct():
    def __setattr__(self, name, value):
         self.__dict__[name] = value
    def __getitem__(self, name):
        return self[name]
    def as_dict(self):
        dic = {}
        for item in self.__dict__.keys(): 
             dic[item] = self.__dict__.get(item)
        return dic

def setOptions(arg,opt,trasnfer=True):
    # update common items of arg from opt
    for item in arg.__dict__.keys(): 
        if item in opt.__dict__.keys():
            arg.__dict__[item] = opt.__dict__.get(item)
    # trasnfer unique items of opt into arg
    if trasnfer:
        for item in opt.__dict__.keys():
            if item not in arg.__dict__.keys():
                arg.__dict__[item] = opt.__dict__.get(item)
    return arg

class TransEM(nn.Module):
    def __init__(self, in_channels=1,is3d=False):
        super(TransEM,self).__init__()
        self.regularize = RSTR(input_dim = in_channels, dim=96,input_resolution=(120,120),depth=1,num_heads=12,window_size=4,mlp_ratio= 2)
        self.gamma = nn.Parameter(torch.rand(1),requires_grad=True)
        self.is3d = is3d
        
    def forward(self,PET,prompts,img=None,RS=None, AN=None, iSensImg = None, mrImg=None, niters = 10, nsubs=1, tof=False, psf=0,device ='cuda', crop_factor = 0):
         # e.g. crop_factor = 0.667
         
         batch_size = prompts.shape[0]
         device = torch.device(device)
         matrixSize = PET.image.matrixSize
         if 0<crop_factor<1: 
             Crop    = lambda x: crop(x,crop_factor,is3d=self.is3d)
             unCrop  = lambda x: uncrop(x,matrixSize[0],is3d=self.is3d)
         else:
             Crop    = lambda x: x
             unCrop  = lambda x: x         
         if self.is3d:  
             toTorch = lambda x: zeroNanInfs(Crop(torch.from_numpy(x).unsqueeze(1).to(device=device, dtype=torch.float32)))
             toNumpy = lambda x: zeroNanInfs(unCrop(x)).detach().cpu().numpy().squeeze(1).astype('float32')
             if iSensImg is None:
                  iSensImg = PET.iSensImageBatch3D(AN, nsubs, psf).astype('float32') 
             if img is None:
                  img =  np.ones([batch_size,matrixSize[0],matrixSize[1],matrixSize[2]],dtype='float32')
             if batch_size ==1:
                 if iSensImg.ndim==4:  iSensImg = iSensImg[None].astype('float32') 
                 if img.ndim==3:  img = img[None].astype('float32')
         else:
             reShape = lambda x: x.reshape([batch_size,matrixSize[0],matrixSize[1]],order='F')
             Flatten = lambda x: x.reshape([batch_size,matrixSize[0]*matrixSize[1]],order='F')
             toTorch = lambda x: zeroNanInfs(Crop(torch.from_numpy(reShape(x)).unsqueeze(1).to(device=device, dtype=torch.float)))
             toNumpy = lambda x: zeroNanInfs(Flatten((unCrop(x)).detach().cpu().numpy().squeeze(1)))
             if iSensImg is None:
                  iSensImg,_ = PET.iSensImageBatch2D(prompts,AN, nsubs, psf)
             if img is None:
                  img =  np.ones([batch_size,matrixSize[0]*matrixSize[1]],dtype='float32')  
             if batch_size ==1:
                 if iSensImg.ndim==2:  iSensImg = iSensImg[None].astype('float32') 
                 if img.ndim==1:  img = img[None].astype('float32')
         if mrImg is not None:
              mrImg = Crop(mrImg)
         imgt = toTorch(img)
         
         for i in range(niters):
             for s in range(nsubs):
                   if self.is3d:
                        img_em = img*PET.forwardDivideBackwardBatch3D(img, prompts, RS, AN, nsubs, s, psf)*iSensImg[:,s,:,:,:]
                        img_emt = toTorch(img_em) 
                        img_regt = zeroNanInfs(self.regularize(imgt,mrImg)) 
                        S = toTorch(iSensImg[:,s,:,:,:])
                   else:
                        img_em = img*PET.forwardDivideBackwardBatch2D(img, prompts, RS, AN, nsubs, s, tof, psf)*iSensImg[:,s,:]
                        img_emt = toTorch(img_em) 
                        img_regt = zeroNanInfs(self.regularize(imgt,mrImg))
                        S = toTorch(iSensImg[:,s,:])
                   imgt = 2*img_emt/((1 - self.gamma*S*img_regt) + torch.sqrt((1 - self.gamma*S*img_regt)**2 + 4*self.gamma*S*img_emt)) 
                   img = toNumpy(imgt)
                   del img_em, img_emt, img_regt, S
            
         del iSensImg, prompts, RS, AN, PET, img

         return unCrop(imgt)

def Trainer(PET, model, opts, train_loader, valid_loader=None):
    import torch.optim as optim
    import os
    
    g = dotstruct()
    g.psf_cm = 0.15
    g.niters = 10
    g.nsubs = 6
    g.lr = 0.001
    g.epochs = 100
    g.in_channels = 1
    g.save_dir = os.getcwd()
    g.model_name = 'TransEM-pm-01'
    g.display = True
    g.disp_figsize=(20,10)
    g.save_from_epoch = None
    g.crop_factor = 0.3
    g.do_validation = True
    g.device = torch.device('cuda:0')
    g.mr_scale = 5

    g = setOptions(g,opts)

    if not os.path.exists(g.save_dir):
        os.makedirs(g.save_dir)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=g.lr )
    toNumpy = lambda x: x.detach().cpu().numpy().astype('float32')

    train_losses = []
    valid_losses = []
    gamma = []
    
    print('start training')
    for e in range(g.epochs):
         
         running_loss = 0
         for sinoLD, sinoHD,imgHD, AN, _,_, _, mrImg, _, _,index in train_loader: 
             AN=toNumpy(AN)
             RS = None
             sinoLD=toNumpy(sinoLD)
             imgHD = imgHD.to(g.device,dtype=torch.float32).unsqueeze(1)
             if g.in_channels==2:
                  mrImg = g.mr_scale*mrImg/mrImg.max()
                  mrImg = mrImg.to(g.device,dtype=torch.float32).unsqueeze(1)
             else:
                  mrImg = None
             optimizer.zero_grad()
             img = model.forward(PET,prompts = sinoLD,AN=AN, mrImg = mrImg,\
                                 niters=g.niters, nsubs = g.nsubs, psf=g.psf_cm, device=g.device, crop_factor=g.crop_factor)#, 
             loss = loss_fn(img,imgHD)
             loss.backward()
             optimizer.step()
             running_loss+=loss.item()
             
             if torch.isnan(model.gamma) or model.gamma.data<0:
                 model.gamma.data= model.gamma.data= torch.Tensor([0.01]).to(g.device,dtype=torch.float32)
             del sinoLD, AN, RS, mrImg, index
    
         else:
             train_losses.append(running_loss/len(train_loader))
             if g.do_validation:
                 valid_loss = 0
                 with torch.no_grad():
                     model.eval()
                     for sinoLD, sinoHD,imgHD, AN, _,_, _, mrImg, _, _,index in valid_loader:
                         AN=toNumpy(AN)
                         RS = None
                         sinoLD=toNumpy(sinoLD)
                         imgHD = imgHD.to(g.device,dtype=torch.float32).unsqueeze(1)
                         if g.in_channels==2:
                             mrImg = g.mr_scale*mrImg/mrImg.max()
                             mrImg = mrImg.to(g.device,dtype=torch.float32).unsqueeze(1)
                         else:
                             mrImg = None
                         img = model.forward(PET,prompts = sinoLD,AN=AN, mrImg = mrImg, niters=g.niters, nsubs = g.nsubs, psf=g.psf_cm, device=g.device, crop_factor=g.crop_factor)#, 
                         valid_loss +=loss_fn(img,imgHD).item()
                 valid_losses.append(valid_loss/len(valid_loader))
                 model.train()
                 output_data = f"Epoch: {e+1}/{g.epochs},Training loss: {train_losses[e]:.3f}, Validation loss: {valid_losses[e]:.3f}\n"
                 print(output_data)
             if ((g.save_from_epoch is not None) and (g.save_from_epoch <=e)) or e==(g.epochs - 1):
                  g.state_dict = model.state_dict()
                  g.train_losses = train_losses
                  g.valid_losses = valid_losses
                  g.training_idx = train_loader.sampler.indices
                  g.gamma = gamma
                  
                  checkpoint = g.as_dict()
                  torch.save(checkpoint,g.save_dir+g.model_name+'-epo-'+str(e)+'.pth')

             torch.cuda.empty_cache()
             output_file = open('{}/loss.txt'.format(g.save_dir), 'a')
             output_file.write(output_data)
             output_file.close()
    import matplotlib.pyplot as plt
    plt.figure()
    l1 , = plt.plot(list(range(g.epochs)),train_losses,color = 'r')
    l2 , = plt.plot(list(range(g.epochs)),valid_losses,color='g')

    plt.legend(handles = [l1,l2],labels = ['Training_loss','Val_loss'],loc = 'upper right')
    plt.xlabel('epoch_num')
    plt.ylabel('Loss_num')
    plt.title('Loss')
    plt.savefig('{}/Loss'.format(g.save_dir))
    plt.show()
 
             
def Test(dl_model_flname, PET, sinoLD, AN, mrImg, niters=None, nsubs = None, device='cuda:0'):

    toNumpy = lambda x: x.detach().cpu().numpy().astype('float32')

    g = torch.load(dl_model_flname, map_location=torch.device(device))
    
    model = TransEM(g['depth'], g['in_channels'], g['is3d']).to(device)
    model.load_state_dict(g['state_dict'])
    
    AN=toNumpy(AN)
    RS = None
    sinoLD = toNumpy(sinoLD)

    if g['in_channels']==2:
         mrImg = g['mr_scale']*mrImg/mrImg.max()
         mrImg = mrImg.to(device,dtype=torch.float32).unsqueeze(1)
    else:
         mrImg = None
    niters = niters or g['niters']
    nsubs = nsubs or g['nsubs']
        
    with torch.no_grad():
        model.eval()
        img = model.forward(PET,prompts = sinoLD,AN=AN, mrImg = mrImg,\
                        niters=niters, nsubs = nsubs, psf=g['psf_cm'], device=device, crop_factor=g['crop_factor'])#, 
    return toNumpy(img).squeeze()