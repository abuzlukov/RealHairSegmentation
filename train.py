import torch as T
import torch.nn as nn
import torch.nn.functional as F
import ImportData as data_path
from torchsummary import summary
from torch.optim import Adam
import matplotlib.pyplot as plt
import modelHair
from modelHair import HairSegmentation

from tqdm import tqdm

def train_valid_model(model,train_loader,valid_loader,loss,optimizer,epochs):

# Loss & Optim :
# Training : 

    T_loss=[]
    for i in range(epochs):
        loss_count=0.0
        for input_,ground_ in train_loader:
            optimizer.zero_grad()
            
            output=model(input_)
            ground_ = T.argmax(ground_, dim=1)
            l=loss(output.float(),ground_.float())
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
            optimizer.zero_grad()
            
            output=model(input_)
            ground_ = T.argmax(ground_, dim=1)
            l=loss(output.float,ground_.float())
            V_loss.append(l.item())
            l.backward()
            optimizer.step()
            print(f'Epoch: {i+1} / {epochs} \t\t\t Training Loss:{l}')
    plt.plot(V_loss,'r')
    plt.show()


def main(train_path,valid_path):
    HairSegmentation_net= HairSegmentation()
    params=list(HairSegmentation_net.parameters())
    
    epochs=10
    '''
    Import Data:
    data_path.ImportData(DATA_PATH)
    '''
    
    data_train=data_path.ImportData(root=train_path)
    data_validation=data_path.ImportData(root=valid_path)
    
    train_loader=T.utils.data.DataLoader(data_train, batch_size=32, shuffle=False)
    valid_loader=T.utils.data.DataLoader(data_validation, batch_size=32, shuffle=False)
    
    loss=nn.MSELoss()
    optimizer=Adam(params,lr=0.001,weight_decay=0.0001)
    
    train_valid_model(HairSegmentation_net,train_loader,valid_loader,loss,optimizer,epochs)
    
    
    
if __name__ =='__main__':
    data_train=data_path.ImportData(root='/home/anass/OPPPYTHON/RealHairSegmentation/Dataset_Hair/training/')
    data_validation=data_path.ImportData(root='/home/anass/OPPPYTHON/RealHairSegmentation/Dataset_Hair/validation/')
    main(data_train,data_validation)