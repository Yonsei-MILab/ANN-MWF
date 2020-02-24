# import libraries

import os
from scipy import io

import matplotlib.pyplot as plt
import numpy as np

from utils4test import *

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2"  # specify which GPU(s) to be used

# Load in-vivo test data
meas = io.loadmat('Data/im_JKJ_1.5mm.mat')

# TE of in-vivo data [s]
te = meas['te']
te = np.squeeze(te[:17])

# mGRE data & create mask
im = np.abs(meas['aimc'])
im = im[:,:,:,:17]
im_mask = create_mask(im)

[ay,ax,az,ae] = im.shape

# model define
model = call_MWF_model(te)

# trained weight upload
model.load_state_dict(torch.load("./Model/model(17echo).pth"))
model.eval()

MWF = test_invivo(im,model)
MWF = MWF*mask

sslice = 23;
MWF_plot(MWF,sslice)

io.savemat('./Result.mat',{'MWF':MWF})
