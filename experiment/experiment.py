import argparse
from sklearn.model_selection import KFold, train_test_split
import torch
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd

from utils.utils import calcCorr, preprocess, set_seed
from model import UnetRemake


def trainModel(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    
    dataPath = f'./dataset/{args.dataset}_data.csv'
    labelPath = f'./dataset/{args.dataset}_label.csv'
    
    trainData = pd.read_csv(dataPath)
    labelData = pd.read_csv(labelPath)

    trainData, trainLabel, testData, testLabel = preprocess(trainData, labelData, device)

    # TODO

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(trainData):
        trainData, trainLabel, testData, testLabel = trainData[train_index], trainLabel[train_index], trainData[
            test_index], trainLabel[test_index]
        X_train, X_test, y_train, y_test = train_test_split(trainData, trainLabel, test_size=0.2, random_state=42)

        trainDataset = Data.TensorDataset(X_train, y_train)
        testDataset = Data.TensorDataset(X_test, y_test)
        trainDataLoader = Data.DataLoader(dataset=trainDataset, batch_size=args.batch_size, shuffle=True)
        testDataLoader = Data.DataLoader(dataset=testDataset, batch_size=args.batch_size, shuffle=True)

        model = UnetRemake(tasks=args.tasks).to(device)
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.gamma)

        taskR = [0] * args.tasks
        for epoch in range(args.epoch):
            model.train()
            for i, (data, label) in enumerate(trainDataLoader):
                weights, output = model(data, label)
                loss = criterion(output, label)
                weightedLoss = torch.mul(weights, loss)  # 这里应该使用 mul 而不是 dot
                optimizer.zero_grad()
                weightedLoss.backward()
                # 这里使用权重损失来更新，是因为要考虑到不同任务的权重
                optimizer.step()

                for id in range(args.tasks):
                    taskR[id] = calcCorr(output[:, id], label[:, id])

                model.weights.grad.data = torch.zeros_like(model.weights.grad.data)

            scheduler.step()


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
    praser.add_argument('--tasks', type=int, default=3, help='the number of tasks')
    praser.add_argument('--lr', type=float, default=1e-3, help='training learning rate')
    praser.add_argument('--epoch', type=int, default=250, help='training epoch')
    praser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    praser.add_argument('--alpha', type=float, default=0.5, help='loss function weight')
    praser.add_argument('--gamma', type=float, default=0.1, help='scheduler gamma')
    praser.add_argument('--dataset', type=str, default='Al', help='the name of dataset')

    args = praser.parse_args()

    main(args)
