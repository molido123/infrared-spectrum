
import math
import os
import random
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import torch

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def splitData(data, label, ratio):
    trainData, testData, trainLabel, testLabel = train_test_split(data, label, test_size=ratio, random_state=42)
    return trainData, testData, trainLabel, testLabel

def preprocess(trainData, trainLabel, device):
    # trainData = SNV(trainData)
    trainData, testData, trainLabel, testLabel = splitData(trainData, trainLabel, 0.2)
    
    # scaler = StandardScaler()
    # trainData = scaler.fit_transform(trainData)
    # testData = scaler.transform(testData)

    """ 
    在机器学习中，通常使用`fit`来计算训练数据的统计信息（例如均值和标准差），
    然后使用这些统计信息来对测试数据进行转换。因此，在给定的代码中，
    第一行代码`scaler.fit_transform(trainData)`用于计算并应用统计信息到训练数据，
    而第二行代码`scaler.transform(testData)`则是将相同的转换应用到测试数据上，
    但不再需要重新计算统计信息。
    
    此处师姐原代码应该错误，正确如上所述
    """
    
    trainData = torch.from_numpy(trainData).float().to(device)
    trainLabel = torch.from_numpy(trainLabel.values).float().to(device)
    testData = torch.from_numpy(testData).float().to(device)
    testLabel = torch.from_numpy(testLabel.values).float().to(device)
    
    # """save this scaler at local for prediction"""
    # with open('scaler.pkl', 'wb') as f:
    #     torch.save(scaler, f)

    # return trainData, trainLabel
    return trainData, testData, trainLabel, testLabel

def calcCorr(a, b):
    a_avg = sum(a) / len(a)
    b_avg = sum(b) / len(b)
    # 计算分子，协方差————按照协方差公式，本来要除以n的，由于在相关系数中上下同时约去了n，于是可以不除以n
    cov_ab = sum([(x - a_avg) * (y - b_avg) for x, y in zip(a, b)])
    # 计算分母，方差乘积————方差本来也要除以n，于是可以不除以n
    sq = math.sqrt(sum([(x - a_avg) ** 2 for x in a]) * sum([(x - b_avg) ** 2 for x in b]))
    
    corr_factor = cov_ab / sq
    return corr_factor

def SNV(data):
    data = np.array(data)
    data_std = np.std(data, axis=1)
    data_average = np.mean(data, axis=1)
    return [[((value - avg) / std) for value in row] for row, avg, std in zip(data, data_average, data_std)]


def Test_preprocess(trainData, device):
    trainData = SNV(trainData)

    scaler = StandardScaler()
    trainData = scaler.fit_transform(trainData)

    """ 
    在机器学习中，通常使用`fit`来计算训练数据的统计信息（例如均值和标准差），
    然后使用这些统计信息来对测试数据进行转换。因此，在给定的代码中，
    第一行代码`scaler.fit_transform(trainData)`用于计算并应用统计信息到训练数据，
    而第二行代码`scaler.transform(testData)`则是将相同的转换应用到测试数据上，
    但不再需要重新计算统计信息。

    此处师姐原代码应该错误，正确如上所述
    """

    trainData = torch.from_numpy(trainData).float().to(device)

    """save this scaler at local for prediction"""
    with open('scaler_test.pkl', 'wb') as f:
        torch.save(scaler, f)

    return trainData
