import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from utils import preprocess

def trainModel(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    
    dataPath = f'./dataset/{args.dataset}_data.csv'
    labelPath = f'./dataset/{args.dataset}_label.csv'
    
    trainData = pd.read_csv(dataPath)
    labelData = pd.read_csv(labelPath)
    
    trainData, trainLabel, testData, testLabel = preprocess(trainData, labelData, device)
    
    # TODO
    
    
def testModel():
    # TODO
    
    pass

if __name__ == '__main__':
    set_seed(42)
    praser = argparse.ArgumentParser(description='Train the Multi-Task Model')
    praser.add_argument('--lr', type=float, default=1e-3, help='training learning rate')
    praser.add_argument('--epoch', type=int, default=250, help='training epoch') 
    praser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    praser.add_argument('--alpha', type=float, default=0.5, help='loss function weight')
    praser.add_argument('--dataset', type=str, default='Al', help='the name of dataset')
    
    args = praser.parse_args() 
     
    trainModel(args)
    testModel()
    
    