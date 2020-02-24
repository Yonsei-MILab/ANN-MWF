# import libraries

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets

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
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = self.fc5(x)

            return x

    model = MWF_Model().cuda().float()
    
    return model

########################### Signal generation  #########################

class Datagen(data.Dataset):
    def __init__(self,te,snr,s_thickness):
        self.te = te
        self.snr = snr
        self.s_thickness = s_thickness
    
    def __len__(self):
        return 3000

    def __getitem__(self, idx):
        
        TE = self.te
      
        # Generate MWF 0 to 0.5 with random scaling of each amplitudes    
        MWF = np.random.rand(1)*0.5
        M0 = MWF
        scaling = 0.5+1.5*np.random.rand(1)
        M2 = (1-MWF)/(1+scaling)             
        M1 = M2*scaling
        
        # Generate T2* and frequency offsets 
        T20, T21, T22 = (np.multiply(0.001,np.random.normal(10, 2)),np.multiply(0.001,np.random.normal(72, 10)),np.multiply(0.001,np.random.normal(48, 6)))
        R20, R21, R22 = (1/T20,1/T21,1/T22)
        
        fme = np.random.randint(-10,20)
        fae = np.random.randint(-10,10)

        cmpx_my = np.complex(R20,2*np.pi*fme)
        cmpx_ax = np.complex(R21,2*np.pi*fae)
        cmpx_ex = R22

        signal = (M0*np.exp(-TE*cmpx_my)+M1*np.exp(-TE*cmpx_ax)+M2*np.exp(-TE*cmpx_ex))
        signal = np.abs(signal)
        
        ########################### sinc flag ######################
        
        sinc_flag = 0
        sig = signal*np.sinc(267.513*0.1*sinc_flag*np.random.rand()*TE)
        
        ############### add noise ########################

        snr = self.snr
        sstd = sig[0]/snr
        noise_sig = sig+sstd*np.random.randn(len(TE))

        ########################### normalize & scaling ######################
        
        scale_factor, bias_factor = (0.9+0.3*np.random.randn(1),0.02*np.random.rand(1))
        scaled_sig = noise_sig * scale_factor + bias_factor
        
        batch_x = scaled_sig
        batch_y = MWF
        
        return batch_x, batch_y
    
########################### model train ##############################

def train(model, optimizer, criterion, args, training_generator, validation_generator):

    use_cuda = torch.cuda.is_available()
    
    for epoch in range(args.epoch):
        train_loss = 0.0
        val_loss = 0.0
        iteration = 0
        
        # Training
        for data, target in training_generator:

            model.train()

            if use_cuda:
                data = data.cuda().float()
                target = target.cuda().float()

            # 1. clear the gradients of all optimized variables
            optimizer.zero_grad()

            # 2. forward pass
            output = model(data)

            # 3. calculate the loss
            loss = criterion(output, target)

            # 4. backward pass
            loss.backward()

            # 5. parameter update
            optimizer.step()

            # update training loss
            train_loss += loss.item()
            iteration += 1
        
        # Validation
        for val_data, val_target in validation_generator:

            model.eval()

            if use_cuda:
                val_data = val_data.cuda().float()
                val_target = val_target.cuda().float()

            output = model(data)

            loss_val = criterion(output, target)

            # update training loss
            val_loss += loss_val.item()

        # calculate average loss over one epoch
        train_loss = train_loss/iteration
        val_loss = val_loss/iteration
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch+1, train_loss, val_loss))
        
    return model
