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
import pandas as pd

#%% Setup Data variables
DatasetFolder = 'C:/Users/jorge/Desktop/github/img_processing/mars_terrain'
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
batchSize = 16

trainset = torchvision.datasets.ImageFolder(TrainData, transforms)
trainloader = DataLoader(trainset, batch_size=batchSize, shuffle=True)

testset = torchvision.datasets.ImageFolder(TestData, transforms)
testloader = DataLoader(testset, batch_size=batchSize, shuffle= True)


CLASSES = ['bedrock', 'big_rocks', 'gravel', 'sand']
NUM_CLASSES = len(CLASSES)
