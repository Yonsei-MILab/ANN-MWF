# Import libraries

import os
from scipy import io
import argparse
from utils4train import *

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2"  # specify which GPU(s) to be used

meas = io.loadmat('Data/im_JKJ_1.5mm.mat')

# TE of in-vivo data [s]
te = meas['te']
te = np.squeeze(te[:17])

# slice thickness of in-vivo data [cm]
zs = 0.15

# model define
model = call_MWF_model(te)

# Parameters setting
parse = argparse.ArgumentParser()
args = parse.parse_args("")

args.optim = 'Adam'
args.lr = 0.0001
args.epoch = 300
args.batch_size = 1000

if args.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
else:
    raise ValueError('Optimiser Error')
    
# loss function
criterion = nn.MSELoss()

# Signal generation Parameters
params = {'shuffle': True, 'batch_size' : args.batch_size}

# Generators
training_set = Datagen(te,snr=100,s_thickness=zs)
training_generator = data.DataLoader(training_set,**params)

validation_set = Datagen(te,snr=100,s_thickness=zs)
validation_generator = data.DataLoader(validation_set, **params)

# Model Train
model = train(model, optimizer, criterion, args, training_generator, validation_generator)

# save model weights
savePath = "./Model/test.pth"
torch.save(model.state_dict(), savePath)
 
