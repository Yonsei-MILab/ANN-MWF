# import libraries

import torch 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from matplotlib import pyplot as plt
from scipy import signal as sg
import numpy as np

########################### network model ##########################

def call_MWF_model(te):
    
    use_cuda = torch.cuda.is_available()
    
    class MWF_Model(nn.Module):

        def __init__(self):
            super(MWF_Model, self).__init__()       
            self.fc1 = nn.Linear(len(te), 250)
            self.fc2 = nn.Linear(250, 250)
            self.fc3 = nn.Linear(250, 250)
            self.fc4 = nn.Linear(250, 250)
            self.fc5 = nn.Linear(250, 1)
            self.dropout = nn.Dropout(p=0.1)

            # gpu setting
            if use_cuda:
                self.fc1 = self.fc1.cuda()
                self.fc2 = self.fc2.cuda()
                self.fc3 = self.fc3.cuda()
                self.fc4 = self.fc4.cuda()
                self.fc5 = self.fc5.cuda()

        def forward(self,x):
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.dropout(F.relu(self.fc2(x)))
            x = self.dropout(F.relu(self.fc3(x)))
            x = self.dropout(F.relu(self.fc4(x)))
            x = self.fc5(x)

            return x

    model = MWF_Model().cuda().float()
    
    return model

########################### Testing  #########################

def test_invivo(im,model):
    
    [ay,ax,az,ae] = im.shape
    
    MWF = np.zeros([ay,ax,az])

    for sslice in range(az):
        im1 = np.squeeze(im[:,:,sslice:sslice+1,:17])

        im2 = np.zeros([ay,ax,ae])
        for i in range(ay):
            for j in range(ax):
                im2[i,j,:] = im1[i,j,:]/np.abs(im1[i,j,0])

        im3 = np.reshape(im2,[ay*ax,ae])
        im3 = torch.from_numpy(im3)
        im3 = im3.type(torch.FloatTensor)
        im3 = im3.cuda()

        res_temp=model(im3)

        MWF[:,:,sslice] = np.reshape(res_temp.cpu().detach().numpy(),[ay,ax])

    return MWF

######################### plot & create mask ###############################

def create_mask(im):

    [ay,ax,az,ae] = im.shape
    im_mask = np.ones([ay,ax,az])
    
    for zz in range(az):
        aim = np.squeeze(im[:,:,zz,:]);
        [ay,ax,ae] = aim.shape  
        mask=np.abs(aim[:,:,0]); mask[mask<np.mean(mask[:])*.5]=0; mask[mask>0]=1;
        mask = np.float32(mask)
        mask=sg.medfilt2d(sg.medfilt2d(mask,kernel_size=17),kernel_size=17)
        im_mask[:,:,zz] = mask
  
    return im_mask


def MWF_plot(MWF,sslice):
    
    fig = plt.figure(figsize=(7,7))
    plt.imshow(np.squeeze(MWF[:,:,sslice]), cmap='hot',vmin=0,vmax=0.3)
    plt.axis('off')

    plt.show()
