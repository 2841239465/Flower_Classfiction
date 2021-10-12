import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import functional as TF
from PIL import Image, ImageEnhance
from flower_classification.题目.task23.models import *
import random
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torchvision
torchvision.models.resnet18()
# Imagenet标准化
MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]

## 构建训练函数
def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    losses = []
    # 通过循环加载图像数据及对应标签进行学习过程
    for i, (itdata, itlabel) in enumerate(dataloader):
        itdata = itdata.to(device)
        itlabel = itlabel.to(device)
        optimizer.zero_grad()
        outputs = model(itdata)
        loss = loss_fn(outputs, itlabel)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().item())
    losses_res = sum(losses) / len(losses)
    print('train process loss {}'.format(losses_res))
    return losses_res

# 利用PyTorch相关接口，对模型进行验证，并输出测试结果
## 构建测试函数
def test(model, dataloader, loss_fn, device):
    model.eval()
    losses = []
    # 通过循环加载图像数据及对应标签进行测试过程
    with torch.no_grad():
        for i, (itdata, itlabel) in enumerate(dataloader):
            itdata = itdata.to(device)
            itlabel = itlabel.to(device)
            output = model(itdata)
            loss = loss_fn(output, itlabel)
            losses.append(loss.cpu().item())
    losses_res = sum(losses) / len(losses)
    print('test process loss {}'.format(losses_res))
    return losses_res

## 定义准确率计算函数
def accuracy(model, dataloader, device):
    model.eval()
    outputsprd = []
    outputslbl = []
    # 通过循环加载图像数据进行计算得到预测值,与标签一起计算准确率
    with torch.no_grad():
        for i, (itdata, itlabel) in enumerate(dataloader):
            itdata = itdata.to(device)
            itlabel = itlabel.to(device)
            output = model(itdata)
            outputsprd.append(output.detach().cpu().numpy())
            outputslbl.append(itlabel.detach().cpu().numpy())
    outputsprd = np.concatenate(outputsprd)
    outputslbl = np.concatenate(outputslbl)
    acc = np.sum(np.equal(np.argmax(outputsprd, axis=1), outputslbl))
    return acc / len(outputslbl)

# 数据增强
class RandomCrop(torch.nn.Module):
    def __init__(self, outputsize=(224,224)):
        super(RandomCrop, self).__init__()
        self.outputsize = outputsize
    def forward(self, img):
        """请在此处作答 实现图像按概率进行图像裁减"""
        '''0.05概率裁剪'''
        num = random.randint(1, 101)
        if num <= 5 :
            top = random.randint(0, 224)    # 确保裁剪框不会超出图片
            left = random.randint(0, 224)
            pilimg = TF.crop(img, top, left, 224, 224)    # top、lef表示裁剪框左上角坐标，32表示裁剪框的高度和宽度
            return pilimg    # 获取裁剪的图片
        else:
            return img

class Cutout(nn.Module):
    def __init__(self, n_holes=8, length=8):
        super().__init__()
        self.n_holes = n_holes
        self.length = length

    def forward(self, img):
        """请在此处作答 实现图像的模拟遮挡"""
        """思路：通过擦除随机位置，来实现遮挡效果，这里擦除区块大小设定为28*28的大小，可以自行修改"""
        i = random.randint(0, 224)
        j = random.randint(0, 224)
        tenimg = TF.to_tensor(img)
        finimg = TF.erase(tenimg, i, j, 28, 28, 1)    # 这个方法需要传入tensor,i、j是左上角坐标，28、28是擦除区域的高度和宽度,1是擦除值
        return TF.to_pil_image(finimg)

def main():
    # 批尺寸
    BATCH_SIZE = 8

    # 数据集路径
    TRAIN_DATA_DIR = '../task1/flowers/train'
    TEST_DATA_DIR = '../task1/flowers/test'

    ## 定义训练集的数据增强和预处理方法
    """请在此处作答 变化并统一图像尺寸方法"""
    """请在此处作答 请首先实现任务2中随机裁减 RandomCrop 函数功能，再使用该函数数据增强方法"""
    """请在此处作答 请首先实现任务2中模拟遮挡 Cutout 函数功能，再使用该函数数据增强方法"""
    """请在此处作答 可以使用其他的数据增强方法"""
    """请在此处作答 图像格式变化为Torch Tensor及归一化"""
    """请在此处作答 请首先实现任务2中标准化Normalize函数功能，再使用Imagenet标准化操作"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        #RandomCrop(),
        Cutout(),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(p=0.05),    # 概率水平反转图像
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    ## 定义测试集的预处理方法
    """请在此处作答 变化并统一图像尺寸方法"""
    """请在此处作答 图像格式变化为Torch Tensor及归一化，可以使用Imagenet标准化操作"""
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    # 加载训练集将预处理方法作为参数输入
    trdataset = ImageFolder(TRAIN_DATA_DIR, transform=train_transform)

    # 加载测集并将预处理及增强的方法作为参数输入
    tsdataset = ImageFolder(TEST_DATA_DIR, transform=test_transform)

    ## 从Torch Dataset变化到DataLoader
    # 构建训练DataLoader
    traindataloader = torch.utils.data.DataLoader(trdataset, BATCH_SIZE, shuffle=True)
    # 构建测试DataLoader
    testdataloader = torch.utils.data.DataLoader(tsdataset, BATCH_SIZE, shuffle=True)

    """请在此处作答 请首先实现models.py脚本中模型的构建 再进行调用"""
    """不允寻使用Resnet18模型"""
    model = resnet20(5)

    # 初始化损失函数
    loss_function = nn.CrossEntropyLoss()
    LEARNING_RATE = 0.01
    # 初始化优化函数
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE)

    lr_scheduler = StepLR(optimizer, step_size=6, gamma=0.1)    # 学习率衰减策略

    ## 定义训练周期
    # 指定训练周期
    EPOCHS = 20
    train_losses = []
    test_losses = []
    test_accs = []
    # 指定模型训练设备
    model.to(device)

    import time
    since = time.time()

    ## 完成指定周期的训练并将损失函数值保存
    for ep in range(EPOCHS):
        print("The number of epoch {}".format(ep))
        # 进行训练，返回损失值
        """请在此处作答 调用训练函数并将参数传入，并返回损失值"""
        train_loss = train(model, traindataloader, loss_function, optimizer, device)
        # 进行测试，返回损失值
        """请在此处作答 调用测试函数并将参数传入，并返回损失值"""
        test_loss = test(model, testdataloader, loss_function, device)
        """请在此处作答 调用准确率计算函数并将参数传入，并返回准确率"""
        countacc = accuracy(model, testdataloader, device)
        lr_scheduler.step()    # 学习率衰减
        # 将训练损失值记录
        train_losses.append(train_loss)
        # 将测试损失值记录
        test_losses.append(test_loss)
        test_accs.append(countacc)

    print("Coustoming time {}".format(time.time() - since))

    ## 绘制损失曲线
    plt.title('Training and Validation Loss')
    """请在此处作答 使用matplotlib函数api绘制训练和测试返回的损失值"""
    trloss, =plt.plot(train_losses, label='Train_Loss')
    tsloss, =plt.plot(test_losses, label='Test_loss')
    plt.legend(handles=[trloss, tsloss])
    plt.show()

    ## 绘制准确率曲线
    plt.title('Test Accuracy')
    """请在此处作答 使用matplotlib函数api绘制测试返回的准确率值"""
    plt.plot(test_accs, label='Tes_Acc')
    plt.show()

    ## 计算测试集的准确率
    acc = accuracy(model, testdataloader, device)
    """请在此处作答 调用准确率计算函数并将参数传入"""
    print("The accuracy {}".format(acc))

    ## 利用PyTorch相关接口，完成模型的保存，请保存到个人空间/space/pytorch下
    ## 如果/space下没有pytorch文件夹，请先创建
    """请在此处作答"""
    if not os.path.exists('../task1/space/pytorch'):
        os.mkdir('../task1/space/pytorch')
    else:
        os.remove('../task1/space/pytorch')
        os.mkdir('../task1/space/pytorch')
    torch.save(model.state_dict(), '../task1/space/pytorch/model.pth')

if __name__ == '__main__':
    main()
