import argparse
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
import torch
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd

from model.UnetRemake import PLE
from utils.utils import calcCorr, preprocess, set_seed, Test_preprocess, SNV
from model import UnetRemake


def trainModel(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')
    
    dataPath = f'./dataset/{args.dataset}_data.xlsx'
    labelPath = f'./dataset/{args.dataset}_label.xlsx'
    
    trainData = pd.read_excel(dataPath)
    labelData = pd.read_excel(labelPath)

    # trainData, trainLabel, testData, testLabel = preprocess(trainData, labelData, device)
    # trainData, trainLabel = preprocess(trainData, labelData, device)
    trainData = SNV(trainData)
    trainData = np.array(trainData)
    labelData = np.array(labelData)

    # TODO

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 初始化存储交叉验证过程中损失、相关系数和RMSE的列表
    loss_test = []
    r_test = []
    rmse_test = []

    # 根据任务数量初始化各个元素
    task_names = ['AL', 'SI', 'FE']  # 假设有一个任务名称列表，可以根据需要扩展
    if len(task_names) < args.tasks:
        task_names.extend([f'Task_{i+1}' for i in range(len(task_names), args.tasks)])

    for _ in range(args.tasks):
        loss_test.append([])
        r_test.append([])
        rmse_test.append([])

    # 交叉验证的循环
    for train_index, test_index in kf.split(trainData):
        # 为当前fold准备训练和测试数据
        X_train, X_test = trainData[train_index], trainData[test_index]
        y_train, y_test = labelData[train_index], labelData[test_index]

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        # 将训练和测试数据转换为TensorDataset和DataLoader
        trainDataset = Data.TensorDataset(X_train, y_train)
        testDataset = Data.TensorDataset(X_test, y_test)
        trainDataLoader = Data.DataLoader(dataset=trainDataset, batch_size=args.batch_size, shuffle=True)
        testDataLoader = Data.DataLoader(dataset=testDataset, batch_size=args.batch_size, shuffle=True)

        # 初始化模型、损失函数、优化器和学习率调度器
        mmoe = PLE(num_task=args.tasks).to(device)
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(mmoe.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.gamma)

        # 训练循环
        initial_task_loss = None  # 初始化 initial_task_loss
        for epoch in range(args.epoch):
            mmoe.train()
            total_loss = 0
            R_task_train = [0] * args.tasks
            count = 0
            for idx, (b_x, b_y) in enumerate(trainDataLoader):
                b_x, b_y = b_x.to(device), b_y.to(device)
                b_x = torch.unsqueeze(b_x, dim=1)  # 增加一个通道维度

                # 模型前向计算
                predict = mmoe(b_x)

                # 计算每个任务的损失
                task_loss = []
                for j in range(args.tasks):
                    a = b_y[:, j]
                    a = torch.unsqueeze(a, dim=1)
                    task_loss.append(criterion(a, predict[j]))
                task_loss = torch.stack(task_loss)

                # 如果是第一个epoch的第一个batch，初始化 initial_task_loss
                if initial_task_loss is None:
                    initial_task_loss = task_loss.detach().cpu().numpy()

                # 计算加权损失
                weighted_task_loss = torch.mul(mmoe.weights, task_loss)
                loss = torch.sum(weighted_task_loss)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)

                if args.mode == 'grad_norm':
                    W = mmoe.get_last_shared_layer()
                    norms = []
                    for j in range(args.tasks):
                        gLgW = torch.autograd.grad(task_loss[j], W.parameters(), retain_graph=True)
                        norms.append(torch.norm(torch.mul(mmoe.weights[j], gLgW[0])))
                    norms = torch.stack(norms)

                    loss_ratio = task_loss.detach().cpu().numpy() / initial_task_loss
                    inverse_train_rate = loss_ratio / np.mean(loss_ratio)

                    mean_norm = np.mean(norms.detach().cpu().numpy())
                    constant_term = torch.tensor(mean_norm * (inverse_train_rate ** args.alpha), requires_grad=True).to(device)
                    grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
                    mmoe.weights.grad = torch.autograd.grad(grad_norm_loss, mmoe.weights)[0]

                optimizer.step()

                # 修正后的 total_loss 计算逻辑
                total_loss += torch.sum(task_loss).item() * b_x.size(0)
                for j in range(args.tasks):
                    R_task_train[j] += calcCorr(predict[j], b_y[:, j]).item() ** 2 * b_x.size(0)
                count += b_x.size(0)

                normalize_coeff = args.tasks / torch.sum(mmoe.weights.data, dim=0)
                mmoe.weights.data *= normalize_coeff

            scheduler.step()
            print(f'Epoch {epoch + 1} - Train Loss: {total_loss / count:.4f}')
            for j in range(args.tasks):
                print(f'{task_names[j]} R_train: {R_task_train[j] / count:.4f}')

        # 验证循环
        mmoe.eval()
        total_val_loss = 0
        R_task_val = [0] * args.tasks
        count_val = 0
        with torch.no_grad():
            for idx, (b_x, b_y) in enumerate(testDataLoader):
                b_x, b_y = b_x.to(device), b_y.to(device)
                b_x = torch.unsqueeze(b_x, dim=1)  # 增加一个通道维度

                # 模型前向计算
                predict = mmoe(b_x)

                # 计算每个任务的损失
                task_loss = []
                for j in range(args.tasks):
                    a = b_y[:, j]
                    a = torch.unsqueeze(a, dim=1)
                    task_loss.append(criterion(a, predict[j]))
                task_loss = torch.stack(task_loss)

                loss_val = torch.sum(task_loss)
                total_val_loss += torch.sum(task_loss).item() * b_x.size(0)
                for j in range(args.tasks):
                    R_task_val[j] += calcCorr(predict[j], b_y[:, j]).item() ** 2 * b_x.size(0)
                count_val += b_x.size(0)

        print(f'Epoch {epoch + 1} - Validation Loss: {total_val_loss / count_val:.4f}')
        for j in range(args.tasks):
            print(f'{task_names[j]} R_val: {R_task_val[j] / count_val:.4f}')

        # 在各个折叠中跟踪性能
        for j in range(args.tasks):
            loss_test[j].append(total_val_loss / count_val)
            r_test[j].append(R_task_val[j] / count_val)
            rmse_test[j].append(pow(mean_squared_error(predict[j].cpu().numpy(), b_y[:, j].cpu().numpy()), 0.5))

    # 最终的交叉验证结果
    for j in range(args.tasks):
        print(f'{task_names[j]} - Cross-validation results: Avg Loss: {np.mean(loss_test[j]):.4f}, Avg R: {np.mean(r_test[j]):.4f}, Avg RMSE: {np.mean(rmse_test[j]):.4f}')

    # --- 使用所有数据进行最终训练，并保存模型 ---
    print("Training final model with all data...")

    # 构建最终的 DataLoader 使用全部训练数据
    fullDataset = Data.TensorDataset(trainData, labelData)
    fullDataLoader = Data.DataLoader(dataset=fullDataset, batch_size=args.batch_size, shuffle=True)

    mmoe = PLE(num_task=args.tasks).to(device)
    optimizer = torch.optim.Adam(mmoe.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.gamma)

    # 训练模型（使用所有数据）
    for epoch in range(args.epoch):
        mmoe.train()
        total_loss = 0
        R_task_train = [0] * args.tasks
        count = 0
        for idx, (b_x, b_y) in enumerate(fullDataLoader):
            b_x, b_y = b_x.to(device), b_y.to(device)
            b_x = torch.unsqueeze(b_x, dim=1)  # 增加一个通道维度

            # 模型前向计算
            predict = mmoe(b_x)

            # 计算每个任务的损失
            task_loss = []
            for j in range(args.tasks):
                a = b_y[:, j]
                a = torch.unsqueeze(a, dim=1)
                task_loss.append(criterion(a, predict[j]))
            task_loss = torch.stack(task_loss)

            # 计算加权损失
            weighted_task_loss = torch.mul(mmoe.weights, task_loss)
            loss = torch.sum(weighted_task_loss)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # 修正后的 total_loss 计算逻辑
            total_loss += torch.sum(task_loss).item() * b_x.size(0)
            for j in range(args.tasks):
                R_task_train[j] += calcCorr(predict[j], b_y[:, j]).item() ** 2 * b_x.size(0)
            count += b_x.size(0)

        scheduler.step()
        print(f'Epoch {epoch + 1} - Final Train Loss: {total_loss / count:.4f}')
        for j in range(args.tasks):
            print(f'{task_names[j]} Final R_train: {R_task_train[j] / count:.4f}')

    # 保存模型到指定路径
    torch.save(mmoe.state_dict(), './checkpoint/model_final.pth')
    print("Final model saved as 'model_final.pth'.")
            


def testModel(input_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Testing on {device}')
    input_data=Test_preprocess(input_data,device)
    # 将数据转换为 TensorDataset 和 DataLoader
    testDataset = Data.TensorDataset(input_data)
    testDataLoader = Data.DataLoader(dataset=testDataset, batch_size=args.batch_size, shuffle=False)

    # 初始化模型，假设模型结构和训练时一致
    mmoe = PLE(num_task=args.tasks).to(device)

    # 加载训练好的模型权重
    checkpoint_path = './checkpoint/model_final.pth'
    mmoe.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 设置模型为评估模式
    mmoe.eval()

    predictions = []

    # 对测试数据进行前向传播，获取预测结果
    with torch.no_grad():
        for idx, (b_x,) in enumerate(testDataLoader):
            b_x = b_x.to(device)
            b_x = torch.unsqueeze(b_x, dim=1)  # 增加一个通道维度，如果输入需要

            # 模型前向计算
            predict = mmoe(b_x)

            # 预测结果转换为 numpy 格式并存储
            for j in range(args.tasks):
                predictions.append(predict[j].cpu().numpy())

    # 将结果转换为所需的格式输出，可以是 numpy 数组或者 DataFrame
    predictions = np.concatenate(predictions, axis=0)  # 将各批次的预测拼接
    result_df = pd.DataFrame(predictions, columns=[f'Task_{i + 1}_Prediction' for i in range(args.tasks)])

    # 返回预测结果
    return result_df


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
    praser.add_argument('--lr', type=float, default=1e-2, help='training learning rate')
    praser.add_argument('--epoch', type=int, default=250, help='training epoch')
    praser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    praser.add_argument('--alpha', type=float, default=0.5, help='loss function weight')
    praser.add_argument('--gamma', type=float, default=0.9, help='scheduler gamma')
    praser.add_argument('--dataset', type=str, default='all', help='the name of dataset')
    praser.add_argument('--mode', type=str, choices=('grad_norm', 'equal_weight'), default='grad_norm', help='set the grad_norm mode')

    args = praser.parse_args()

    main(args)
