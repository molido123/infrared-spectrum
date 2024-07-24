import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from utils import preprocess, set_seed

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


def main(args):
    """ 
    main 函数定义了 *args, 这意味着你可以传递任意数量的参数给 main 函数.
    这些参数会被打包成一个元组传递给 trainModel 函数
    """
    trainModel(args)
    testModel()

if __name__ == '__main__':
    set_seed(42)
    praser = argparse.ArgumentParser(description='Train the Multi-Task Model')
    praser.add_argument('--lr', type=float, default=1e-3, help='training learning rate')
    praser.add_argument('--epoch', type=int, default=250, help='training epoch') 
    praser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    praser.add_argument('--alpha', type=float, default=0.5, help='loss function weight')
    praser.add_argument('--dataset', type=str, default='Al', help='the name of dataset')
    
    args = praser.parse_args() 
    
    main(args)
    
    