from torchvision.models import resnet18
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from torch import nn
from PIL import Image
import cv2

# 图像ImageNet标准化
MEAN_RGB=[0.485, 0.456, 0.406]
STED_RGB=[0.229, 0.224, 0.225]

# 指定类别名称
label_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic']

# 指定硬件设备
device = torch.device('cpu') # 指定cpu

def load_image(image_path):
    """请在此处作答 调用Opencv api加载图像"""
    img = cv2.imread(image_path)
    #npimg = np.array(img)
    return img

# 利用OpenCV和Numpy相关接口函数完成待测试图像的预处理
def preprocess(img, mean=MEAN_RGB, std=STED_RGB):
    assert isinstance(img, np.ndarray)
    # 图像尺寸变化
    img_rs = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
    # 图像通道变换BGR转换为RGB
    im_rgb = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB)
    # 满足torch tensor的维度要求
    img_rs_arr_chw = img_rs.transpose(2,1,0)    # C、H、W
    # 数据类型变换为float并归一化
    img_rs_tensor = torch.Tensor(img_rs_arr_chw).to(torch.float32) / 255.
    #标准化处理
    img_norm_t = transforms.functional.normalize(img_rs_tensor, mean, std)
    #满足模型计算要求
    img_norm_t_b = img_norm_t.reshape(1, 3, 224, 224)
    return img_norm_t_b

## 加载中的预先训练模型或者赛题中指定的模型
"""请在此处作答 使用torchvision.models中的resnet18作为模型"""
model = resnet18()
"""请在此处作答 并对模型resnet18进行修改，以模型满足任务需要"""
model.fc = nn.Linear(512, 5)    # 修改全连接层
model.relu = nn.Tanh()    # 修改激活函数，实现二分类
# 注意这是一个二分类问题
# 加载预训练模型权重
model.load_state_dict(torch.load('./resnet18-flowers-model.pth', map_location=device))

def infer(image_path, model=model, device=device, label_names=label_names):
    img = load_image(image_path)
    # 完成图像的预处理过程
    img_t = preprocess(img)
    # 指定模型运行设备
    """请在此处作答 将模型和img_t分别指定到对应的硬件设备上"""
    model.to(device)
    img_t = img_t.to(device)
    # 计算得到模型输出结果
    """请在此处作答 得到模型计算结果"""
    model.eval()
    output = model(img_t)
    result = output.detach().cpu().numpy()
    label_index = np.argmax(result)
    # 请注意这是一个二分类问题
    label = label_names[label_index]
    print("分类结果为： {}".format(label))
    return label

if __name__ == '__main__':
    infer('images/daisy1.jpg')
    infer('images/daisy2.jpg')
    infer('images/dandelion1.jpg')
    infer('images/dandelion2.jpg')
    infer('images/rose1.jpg')
    infer('images/rose2.jpg')
    infer('images/sunflower1.jpg')
    infer('images/sunflower2.jpg')
    infer('images/tulip1.jpg')
    infer('images/tulip2.jpg')
