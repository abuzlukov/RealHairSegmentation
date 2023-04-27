# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 09:21:21 2022

@author: lucie
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import gc
import ImportData as data_path
from torch.optim import Adam
from matplotlib import pyplot as plt
from torchvision import transforms

# GPU_CUDA
gc.collect()
T.cuda.empty_cache()

class HairSegmentation(nn.Module):
    def __init__(self):
        super(HairSegmentation,self).__init__()
        # Convolution Layers
        '''
         Convolution torch.nn.Conv2d(
                  in_channels, 
                  out_channels,
                  kernel_size,
                  padding,
                  ...         )
        '''
        self.conv1=nn.Conv2d(1,32,3,padding='same')
        self.conv2=nn.Conv2d(32,64,3,padding='same')
        self.conv3=nn.Conv2d(64,128,3,padding='same')
        self.conv4=nn.Conv2d(128,256,3,padding='same')
        self.conv5=nn.Conv2d(256,512,3,padding='same')
        self.conv6=nn.Conv2d(512,1024,3,padding='same')
        self.conv7=nn.Conv2d(1024,512,3,padding='same')
        self.conv8=nn.Conv2d(512,256,3,padding='same')
        self.conv9=nn.Conv2d(256,128,3,padding='same')
        self.conv10=nn.Conv2d(128,64,3,padding='same')
        self.conv11=nn.Conv2d(64,32,3,padding='same')
        self.conv12=nn.Conv2d(32,1,3,padding='same')
    def forward(self,x):
            x=F.max_pool2d(F.relu(self.conv1(x)),(3,3))
            x=F.max_pool2d(F.relu(self.conv2(x)),(3,3))
            x=F.max_pool2d(F.relu(self.conv3(x)),(3,3))
            x=F.max_pool2d(F.relu(self.conv4(x)),(3,3))
            x=F.max_pool2d(F.relu(self.conv5(x)),(3,3))
            x=F.relu(self.conv5(x))
            x=F.max_pool2d(F.relu(self.conv6(x)),(3,3))


            x=F.max_pool2d(F.relu(self.conv7(x)),(3,3))
            x=T.cat((T.transpose(self.conv8(x)),self.conv2(x)),3)
            x=F.relu(self.conv9(x))
            x=T.cat((T.transpose(self.conv9),self.conv3(x)),3)
            x=F.relu(self.conv10(x))
            x=T.cat((T.transpose(self.conv10(x)),self.conv4(x)),3)
            x=F.relu(self.conv11(x))
            x=T.cat((T.transpose(self.conv11(x)),self.conv5(x)),3)
            x=F.relu(self.conv12(x))
            return x

HairSegmentation_net= HairSegmentation()

params=list(HairSegmentation_net.parameters())
print(HairSegmentation_net)
#if T.cuda.is_available():
#    HairSegmentation_net=HairSegmentation_net.cuda()
epochs=10
'''
Import Data: 
data_path.ImportData(DATA_PATH)
'''
data_train=data_path.ImportData('C:/Users/lucie/Downloads/HairSegmentation/HairDataset/Training_folder/')
data_validation=data_path.ImportData('C:/Users/lucie/Downloads/HairSegmentation/HairDataset/Validation_folder/')

train_loader=T.utils.data.DataLoader(data_train, batch_size=32, shuffle=False)
valid_loader=T.utils.data.DataLoader(data_validation, batch_size=32, shuffle=False)

is_gpu=T.cuda.is_available()

# Loss & Optim :

loss=nn.CrossEntropyLoss()
optimizer=Adam(params,lr=0.001,weight_decay=0.0001)
# Training : 

T_loss=[]
for i in range(epochs):
    loss_count=0.0
    for input_,ground_ in train_loader:
        if is_gpu:
            input_,ground_=input_.cuda(),ground_.cuda()
        optimizer.zero_grad()
        
        output=HairSegmentation_net(input_)
        l=loss(output,ground_)
        T_loss.append(l.item())
        l.backward()
        optimizer.step()
        print(f'Epoch: {i+1} / {epochs} \t\t\t Training Loss:{l}')
plt.plot(T_loss,'r')
plt.show()

# Validation: 
V_loss=[]
for i in range(epochs):
    loss_count=0.0
    for input_,ground_ in valid_loader:
        if is_gpu:
            input_,ground_=input_.cuda(),ground_.cuda()
        optimizer.zero_grad()
        
        output=HairSegmentation_net(input_)
        l=loss(output,ground_)
        V_loss.append(l.item())
        l.backward()
        optimizer.step()
        print(f'Epoch: {i+1} / {epochs} \t\t\t Training Loss:{l}')
plt.plot(V_loss,'r')
plt.show()
