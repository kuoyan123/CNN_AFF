import os
# os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import time
import torch.optim
import torch.utils.data
from ulit.pn_dataset import *
from ulit.init_seed import *
from ulit.acc import *
from model.CNN_1d import *
from model.CNN_AFF import *
import matplotlib.pyplot as plt
import csv
from torch.optim.lr_scheduler import StepLR



parser = argparse.ArgumentParser(description='PyTorch PN_Data Training')
parser.add_argument('--data', metavar='DIR',default=r'.\dataset',help='path to dataset')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,metavar='N',help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,metavar='W', help='weight decay (default: 1e-4)',dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,metavar='N', help='print frequency (default: 10)')
parser.add_argument('-gamma', default=0.1, type=float)
parser.add_argument('-stepsize', default=10, type=int)
parser.add_argument('-seed', default=123, type=int)
parser.add_argument('-use_model', default='CNN_AFF', type=str, help='CNN_1d，CNN_AFF')
# save
parser.add_argument('--save_model', default=True, type=bool)
parser.add_argument('--save_dir', default=r'.\result', type=str)
parser.add_argument('--save_acc_loss_dir', default=r'.\res ult', type=str)


def main():
    # 判断是否含有gpu，否则加载到cpu
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("use gpu")
    else:
        device = torch.device('cpu')
        print("use cpu")
    args = parser.parse_args()
    init_seed(args.seed)    # 初始化参数
    start_time = time.time()  # 记录程序开始时间
    # 构建模型
    if args.use_model == 'CNN_1d':
        model = CNN().to(device)
    elif args.use_model == 'CNN_AFF':
        model = CNN_AFF().to(device)

    args.save_dir = os.path.join(args.save_dir, args.use_model)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)          # 如果文件夹不存在，则创建文件夹
    args.save_acc_loss_dir = os.path.join(args.save_dir, 'train_test_result.csv')
    criterion = nn.CrossEntropyLoss().to(device)  # 交叉熵
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    lr_scheduler = StepLR(optimizer, gamma=args.gamma, step_size=args.stepsize)
    selected_channels = [0, 1]  # 0, 1电流信号,2振动信号
    traindir = os.path.join(args.data, 'train')
    testdir = os.path.join(args.data, 'test')
    train_dataset = CustomDataset(traindir, selected_channels)
    test_dataset = CustomDataset(testdir, selected_channels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.workers, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers, drop_last=False)
    # 训练测试模型
    train_acc_list, train_loss_list, test_acc_list, test_loss_list = [], [], [], []
    for epoch in range(args.start_epoch, args.epochs):
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, lr_scheduler, device)
        test_acc, test_loss = test(test_loader, model, criterion, epoch, device)
        train_acc_list.append(round(train_acc, 4))
        train_loss_list.append(round(train_loss, 4))
        test_acc_list.append(round(test_acc, 4))
        test_loss_list.append(round(test_loss, 4))
        if args.save_model:
            if epoch==0 or epoch==19:
                model_name = 'model' + '_' + str(epoch + 1) + '.pth'
                torch.save(model.state_dict(), os.path.join(args.save_dir, model_name))
    if args.save_model:
        with open(args.save_acc_loss_dir, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'train_acc', 'train_loss', 'test_acc','test_loss'])  # 写入表头
            for epoch in range(len(train_acc_list)):
                writer.writerow([epoch + 1, train_acc_list[epoch], train_loss_list[epoch], test_acc_list[epoch], test_loss_list[epoch]])
    plt.figure(figsize=(10, 4))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.subplot(1, 2, 1)
    # 绘制曲线
    plt.plot(train_acc_list, label='训练准确率', linestyle='-', color='blue')
    plt.plot(test_acc_list, label='测试准确率', linestyle='--', color='red')
    # 添加标签和标题
    plt.xlabel('迭代轮数')
    plt.ylabel('准确率')
    plt.title('训练和测试准确率')
    plt.legend(loc='lower right')  # 左下显示
    plt.subplot(1, 2, 2)
    # 绘制曲线
    plt.plot(train_loss_list, label='训练损失', linestyle='-', color='blue')
    plt.plot(test_loss_list, label='测试损失', linestyle='--', color='red')
    # 添加标签和标题
    plt.xlabel('迭代轮数')
    plt.ylabel('损失')
    plt.title('训练和测试损失')
    plt.legend(loc='upper right')  # 右下显示
    fig_path = os.path.join(args.save_dir, 'acc_loss.png')  # 保存图片
    plt.savefig(fig_path)
    plt.show()
    end_time = time.time()  # 记录程序结束时间
    total_time = end_time - start_time  # 计算程序运行总时间
    print(f"程序运行总时间: {total_time:.2f} 秒")


def train(train_loader, model, criterion, optimizer, epoch, lr_scheduler, device):
    losses = AverageMeter('Loss', ':.4f')
    train_acc = AverageMeter('Acc', ':.4f')
    # switch to train mode
    model.train()
    for i, (data, label) in enumerate(train_loader):
        model.zero_grad()
        optimizer.zero_grad()
        input = data.to(device)
        label = label.to(device)
        # compute output and loss
        output = model(input)
        loss = criterion(output, label.long())
        losses.update(loss.item(), label.size(0))
        # Compute accuracy
        _, predicted = torch.max(output, 1)
        accuracy = (predicted == label).sum().item() / label.size(0)
        train_acc.update(accuracy, label.size(0))
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    lr_scheduler.step()
    print(f'Epoch:[{epoch}] train_Acc:{train_acc.avg:.4f} train_Loss:{losses.avg:.4f}')
    return train_acc.avg,losses.avg


def test(train_loader, model, criterion, epoch, device):
    losses = AverageMeter('Loss', ':.4f')
    test_acc = AverageMeter('Acc', ':.4f')
    # switch to train mode
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            losses.update(loss.item(), label.size(0))
            # Compute accuracy
            _, predicted = torch.max(output, 1)
            accuracy = (predicted == label).sum().item() / label.size(0)
            test_acc.update(accuracy, label.size(0))
        print(f'Epoch:[{epoch}] test_Acc:{test_acc.avg:.4f} test_Loss:{losses.avg:.4f}')
    return test_acc.avg,losses.avg


if __name__ == '__main__':
    main()
