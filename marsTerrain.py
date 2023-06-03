# Python script using PyTorch to perform Multi-Class Classification using different networks
# to test their performance on martian terrain. 
# Jorge Prado, 2023

#%% Import Packages
import torch
import torch.nn as nn
from torch import optim
import torchvision
import torchvision.transforms as TF
from torchvision import models
from torch.utils.data import DataLoader
from collections import OrderedDict 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd

#%% Setup Data variables
DatasetFolder = '/home/lpa3/Escritorio/prado/mars_terrain'
TrainData = DatasetFolder + '/train'
TestData = DatasetFolder + '/test'

# image transforms, could implement more transformations for better generalization 
transforms = TF.Compose([
    TF.Resize((224,224)),
    #TF.Grayscale(num_output_channels=1),# images already on grayscale
    TF.ToTensor(),
    #TF.Normalize((0.5, ), (0.5, ))  #to do, find normalization values for dataset
])

# Setup datasets & dataloaders 
batchSize = 8

# For the train/test splits we previously separate our data into a 70-30 % split
trainset = torchvision.datasets.ImageFolder(TrainData, transforms)
trainloader = DataLoader(trainset, batch_size=batchSize, shuffle=True)

testset = torchvision.datasets.ImageFolder(TestData, transforms)
testloader = DataLoader(testset, batch_size=batchSize, shuffle= True)


CLASSES = ['bedrock', 'big_rocks', 'gravel', 'sand']
NUM_CLASSES = len(CLASSES)

#%% define function to show image grid of data
def imshow(image_torch):
    image_torch = image_torch.numpy().transpose((1, 2, 0))
    plt.figure()
    plt.imshow(image_torch)

X_train, y_train = next(iter(trainloader))
img_grid = torchvision.utils.make_grid(X_train[:16, :, :, :], scale_each= True, nrow=4)
imshow(img_grid) 

#%%
# Load pretrained models

# ResNet50  https://arxiv.org/abs/1512.03385
resNetModel = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#resNetModel

# MobileNet
mobileNet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
#mobileNet

# AlexNet
AlexNet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT) 
#AlexNet

# DenseNet  https://arxiv.org/abs/1608.06993
denseNetModel = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
#denseNetModel

# Vision Transformer  https://arxiv.org/abs/2010.11929
ViTmodel = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
#ViTmodel

# Swin Transformer  https://arxiv.org/abs/2103.14030 
swinModel = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
#swinModel

# ConvNeXT  https://arxiv.org/abs/2201.03545
convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
#convnext

# MaxVIT  https://arxiv.org/abs/2204.01697
maxvit = models.maxvit_t(weights=models.MaxVit_T_Weights.DEFAULT)
#maxvit

# EfficientNet v2  https://arxiv.org/abs/2104.00298
effNet = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
#effNet

# %% freeze model parameters 
model = [resNetModel, mobileNet, AlexNet, denseNetModel, ViTmodel, swinModel,
          convnext, maxvit, effNet]
i=0
for allmodels in model:    
    i+=1
    for params in model[i-1].parameters(): 
        params.requires_grad = False 

# %% overwrite classifier of models to match out features
    #resNetModel.fc  in_f=2048, out= 1000
    #mobileNet.classifier[1] in_f=1280, out=1000, bias True
    #AlexNet.classifier[6] in_f=4096, out=1000, bias True
    #denseNetModel.classifier in_f = 1024, out= 1000
    #ViTmodel.heads in_f = 768, out= 1000
    #swinModel.head in_f = 768, out= 1000
    #convnext.classifier[2] in_f = 768, out= 1000, bias True
    #maxvit.classifier[5] in_f = 512, out = 1000, bias False
    #effNet.classifier[1] in_f = 1280, out = 1000, bias True

resNetModel.fc = nn.Sequential(OrderedDict([ 
    ('Linear',nn.Linear(2048,NUM_CLASSES, bias=True)),
])) 

mobileNet.classifier[1] = nn.Sequential(OrderedDict([
    ('Linear', nn.Linear(1280, NUM_CLASSES, bias=True))
]))

AlexNet.classifier[6] = nn.Sequential(OrderedDict([
    ('Linear', nn.Linear(4096, NUM_CLASSES, bias=True))
]))

denseNetModel.classifier = nn.Sequential(OrderedDict([
    ('Linear', nn.Linear(1024, NUM_CLASSES, bias=True))
]))

ViTmodel.heads = nn.Sequential(OrderedDict([
    ('Linear', nn.Linear(768, NUM_CLASSES, bias=True))
]))

swinModel.head = nn.Sequential(OrderedDict([
    ('Linear', nn.Linear(768, NUM_CLASSES, bias=True))
]))

convnext.classifier[2] = nn.Sequential(OrderedDict([
    ('Linear', nn.Linear(768, NUM_CLASSES, bias=True))

]))
maxvit.classifier[5] = nn.Sequential(OrderedDict([
    ('Linear', nn.Linear(512, NUM_CLASSES, bias=False))
]))

effNet.classifier[1] = nn.Sequential(OrderedDict([
    ('Linear', nn.Linear(1280, NUM_CLASSES, bias=True))
]))

# %% Optimizers, loss function and losses
# Learning rate
lr = 0.001

opt1 = optim.Adam(resNetModel.fc.parameters(), lr=lr)
opt2 = optim.Adam(mobileNet.classifier[1].parameters(), lr=lr)
opt3 = optim.Adam(AlexNet.classifier[6].parameters(), lr=lr)
opt4 = optim.Adam(denseNetModel.classifier.parameters(), lr=lr)
opt5 = optim.Adam(ViTmodel.heads.parameters(), lr=lr)
opt6 = optim.Adam(swinModel.head.parameters(), lr=lr)
opt7 = optim.Adam(convnext.classifier[2].parameters(), lr=lr)
opt8 = optim.Adam(maxvit.classifier[5].parameters(), lr=lr)
opt9 = optim.Adam(effNet.classifier[1].parameters(), lr=lr)

loss_function = nn.CrossEntropyLoss()

ResNet_acc=[]
resNet_losses=[]
resNet_l_mD=[]

MobileNet_acc=[]
mobileNet_losses=[]
mobileNet_l_mD=[]

AlexNet_acc=[]
AlexNet_losses=[]
AlexNet_l_mD=[]

denseNet_acc=[]
denseNet_losses=[]
denseNet_l_mD=[]

ViT_acc=[]
ViT_losses=[]
ViT_l_mD=[]

swin_acc=[]
swin_losses=[]
swin_l_mD=[]

convnext_acc=[]
convnext_losses=[]
convnext_l_mD=[]

maxvit_acc=[]
maxvit_losses=[]
maxvit_l_mD=[]

effNet_acc=[]
effNet_losses=[]
effNet_l_mD=[]

resNetModel.train()
mobileNet.train()
AlexNet.train()
denseNetModel.train()
ViTmodel.train()
swinModel.train()
convnext.train()
maxvit.train()
effNet.train()

#%% Train models
#torch.cuda.get_device_name(torch.cuda.current_device())
NUM_EPOCHS = 20

# train resnet
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        opt1.zero_grad()
        outputs = resNetModel(inputs)
        loss = loss_function(outputs,labels.long())
        loss.backward()
        opt1.step()
        resNet_l_mD.append(float(loss.data.detach().numpy()))
    resNet_losses.append(float(loss.data.detach().numpy()))
    #if (epoch % 10 == 0):
    print(f'ResNet, Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.12f}')

    # test model accuracy 
    y_test = []
    y_test_hat = []
    for i, data in enumerate(testloader, 0):
        inputs, y_test_temp = data
        with torch.no_grad():
            y_test_hat_temp = resNetModel(inputs).round()
        
        y_test.extend(y_test_temp.numpy())
        y_test_hat.extend(y_test_hat_temp.numpy())

    ResNet_acc.append(accuracy_score(y_test, np.argmax(y_test_hat, axis=1)))
    print(f'Accuracy: {ResNet_acc[epoch]*100:.2f} % after {epoch+1} epochs')

del loss

# train MobileNet
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        opt2.zero_grad()
        outputs = mobileNet(inputs)
        loss = loss_function(outputs,labels.long())
        loss.backward()
        opt2.step()
        mobileNet_l_mD.append(float(loss.data.detach().numpy()))
    mobileNet_losses.append(float(loss.data.detach().numpy()))
    #if (epoch % 10 == 0):
    print(f'MobileNet, Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.12f}')

    # test model accuracy 
    y_test = []
    y_test_hat = []
    for i, data in enumerate(testloader, 0):
        inputs, y_test_temp = data
        with torch.no_grad():
            y_test_hat_temp = mobileNet(inputs).round()
        
        y_test.extend(y_test_temp.numpy())
        y_test_hat.extend(y_test_hat_temp.numpy())

    MobileNet_acc.append(accuracy_score(y_test, np.argmax(y_test_hat, axis=1)))
    print(f'Accuracy: {MobileNet_acc[epoch]*100:.2f} % after {epoch+1} epochs')

del loss

# train AlexNet
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        opt3.zero_grad()
        outputs = AlexNet(inputs)
        loss = loss_function(outputs,labels.long())
        loss.backward()
        opt3.step()
        AlexNet_l_mD.append(float(loss.data.detach().numpy()))
    AlexNet_losses.append(float(loss.data.detach().numpy()))
    #if (epoch % 10 == 0):
    print(f'AlexNet, Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.12f}')
    
    # test model accuracy 
    y_test = []
    y_test_hat = []
    for i, data in enumerate(testloader, 0):
        inputs, y_test_temp = data
        with torch.no_grad():
            y_test_hat_temp = AlexNet(inputs).round()
        
        y_test.extend(y_test_temp.numpy())
        y_test_hat.extend(y_test_hat_temp.numpy())

    AlexNet_acc.append(accuracy_score(y_test, np.argmax(y_test_hat, axis=1)))
    print(f'Accuracy: {AlexNet_acc[epoch]*100:.2f} % after {epoch+1} epochs')
    
del loss

# train denseNet
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        opt4.zero_grad() 
        outputs = denseNetModel(inputs)
        loss = loss_function(outputs,labels)
        loss.backward()
        opt4.step()
        denseNet_l_mD.append(float(loss.data.detach().numpy()))
    denseNet_losses.append(float(loss.data.detach().numpy()))
    #if (epoch % 10 == 0):
    print(f'DenseNet, Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.12f}')

    # test model accuracy 
    y_test = []
    y_test_hat = []
    for i, data in enumerate(testloader, 0):
        inputs, y_test_temp = data
        with torch.no_grad():
            y_test_hat_temp = denseNetModel(inputs).round()
        
        y_test.extend(y_test_temp.numpy())
        y_test_hat.extend(y_test_hat_temp.numpy())

    denseNet_acc.append(accuracy_score(y_test, np.argmax(y_test_hat, axis=1)))
    print(f'Accuracy: {denseNet_acc[epoch]*100:.2f} % after {epoch+1} epochs')

del loss

# train ViT
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        opt5.zero_grad()
        outputs = ViTmodel(inputs)
        loss = loss_function(outputs,labels)
        loss.backward()
        opt5.step()
        ViT_l_mD.append(float(loss.data.detach().numpy()))
    ViT_losses.append(float(loss.data.detach().numpy()))
    #if (epoch % 10 == 0):
    print(f'ViT, Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.12f}')

    # test model accuracy 
    y_test = []
    y_test_hat = []
    for i, data in enumerate(testloader, 0):
        inputs, y_test_temp = data
        with torch.no_grad():
            y_test_hat_temp = ViTmodel(inputs).round()
        
        y_test.extend(y_test_temp.numpy())
        y_test_hat.extend(y_test_hat_temp.numpy())

    ViT_acc.append(accuracy_score(y_test, np.argmax(y_test_hat, axis=1)))
    print(f'Accuracy: {ViT_acc[epoch]*100:.2f} % after {epoch+1} epochs')

del loss

# train swin
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        opt6.zero_grad()
        outputs = swinModel(inputs)
        loss = loss_function(outputs,labels)
        loss.backward()
        opt6.step()
        swin_l_mD.append(float(loss.data.detach().numpy()))
    swin_losses.append(float(loss.data.detach().numpy()))
    #if (epoch % 10 == 0):
    print(f'Swin, Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.12f}')

    # test model accuracy 
    y_test = []
    y_test_hat = []
    for i, data in enumerate(testloader, 0):
        inputs, y_test_temp = data
        with torch.no_grad():
            y_test_hat_temp = swinModel(inputs).round()
        
        y_test.extend(y_test_temp.numpy())
        y_test_hat.extend(y_test_hat_temp.numpy())

    swin_acc.append(accuracy_score(y_test, np.argmax(y_test_hat, axis=1)))
    print(f'Accuracy: {swin_acc[epoch]*100:.2f} % after {epoch+1} epochs')

del loss

# train convnext
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        opt7.zero_grad()
        outputs = convnext(inputs)
        loss = loss_function(outputs,labels)
        loss.backward()
        opt7.step()
        convnext_l_mD.append(float(loss.data.detach().numpy()))
    convnext_losses.append(float(loss.data.detach().numpy()))
    #if (epoch % 10 == 0):
    print(f'ConvNeXt, Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.12f}')

    # test model accuracy 
    y_test = []
    y_test_hat = []
    for i, data in enumerate(testloader, 0):
        inputs, y_test_temp = data
        with torch.no_grad():
            y_test_hat_temp = convnext(inputs).round()
        
        y_test.extend(y_test_temp.numpy())
        y_test_hat.extend(y_test_hat_temp.numpy())

    convnext_acc.append(accuracy_score(y_test, np.argmax(y_test_hat, axis=1)))
    print(f'Accuracy: {convnext_acc[epoch]*100:.2f} % after {epoch+1} epochs')

del loss

# train maxvit
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        opt8.zero_grad()
        outputs = maxvit(inputs)
        loss = loss_function(outputs,labels)
        loss.backward()
        opt8.step()
        maxvit_l_mD.append(float(loss.data.detach().numpy()))
    maxvit_losses.append(float(loss.data.detach().numpy()))
    #if (epoch % 10 == 0):
    print(f'MaxVIT, Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.12f}')

    # test model accuracy 
    y_test = []
    y_test_hat = []
    for i, data in enumerate(testloader, 0):
        inputs, y_test_temp = data
        with torch.no_grad():
            y_test_hat_temp = maxvit(inputs).round()
        
        y_test.extend(y_test_temp.numpy())
        y_test_hat.extend(y_test_hat_temp.numpy())

    maxvit_acc.append(accuracy_score(y_test, np.argmax(y_test_hat, axis=1)))
    print(f'Accuracy: {maxvit_acc[epoch]*100:.2f} % after {epoch+1} epochs')

del loss

# train effnet
for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        opt9.zero_grad()
        outputs = effNet(inputs)
        loss = loss_function(outputs,labels)
        loss.backward()
        opt9.step()
        effNet_l_mD.append(float(loss.data.detach().numpy()))
    effNet_losses.append(float(loss.data.detach().numpy()))
    #if (epoch % 10 == 0):
    print(f'Efficient Net v2, Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.12f}')

    # test model accuracy 
    y_test = []
    y_test_hat = []
    for i, data in enumerate(testloader, 0):
        inputs, y_test_temp = data
        with torch.no_grad():
            y_test_hat_temp = effNet(inputs).round()
        
        y_test.extend(y_test_temp.numpy())
        y_test_hat.extend(y_test_hat_temp.numpy())

    effNet_acc.append(accuracy_score(y_test, np.argmax(y_test_hat, axis=1)))
    print(f'Accuracy: {effNet_acc[epoch]*100:.2f} % after {epoch+1} epochs')

del loss

#%%
# Create Dataframe to plot losses/epoch
losses_array = np.array([resNet_losses, mobileNet_losses, AlexNet_losses, denseNet_losses,
                        ViT_losses, swin_losses, convnext_losses, maxvit_losses, effNet_losses])
index_values = NUM_EPOCHS
column_values = model

loss_df = pd.DataFrame(data = losses_array,
                       index=index_values,
                       columns=column_values)
print(loss_df)