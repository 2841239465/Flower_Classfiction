import os
from os.path import join, exists
import numpy as np
from PIL import Image
import shutil
import matplotlib.pyplot as plt

DATA_ROOT = './images'
#移动后的数据集images路径
SRC_ROOT = './space/images'
#将数据集images拷贝至个人空间路径SRC_ROOT，仅运行一次
if not exists("./space/images"):
    shutil.copytree(DATA_ROOT, SRC_ROOT)

#下面根据各函数功能完成任务

def find_unlabeled():
    """找到没有标注的所有文件名以及观察其类别。"""
    #没有命名的文件名
    for root, _, files in os.walk(SRC_ROOT):
        for fname in files:
            import re
            # 通过正则表达式匹配字符串
            temp = re.match(r"\w+_\d{4}", fname)
            # match函数 匹配失败返回None
            if temp == None:
                fp = open(join(SRC_ROOT, fname), 'rb')
                img = Image.open(fp)
                # 观察类别
                print(img.format, img.size, img.mode)
                print("错误的文件名是：" + fname)
                # img.show()

def label_img(fname, label):
    """
    给未标注的图像名加上标签。
    ------
    unlabeled：没有标注的文件名
    label：对应的标注信息
    """
    #根据观察到的标签，按照现有格式重命名文件
    # 如果错误文件存在则改名，避免重复操作
    if exists(join(SRC_ROOT, fname)):
        os.rename(join(SRC_ROOT, fname), join(SRC_ROOT, label+'_'+fname))
        print("更改成功")
    else:
        print("更名完毕，没有错误文件")


def regulate_filename():
    """规范文件命名方式"""
    for _, _, files in os.walk(SRC_ROOT):
        for fname in files:
            bname, ext = fname.split('.')    # 获取基础名和扩展名
            cls, imgid = bname.split('_')    # 获取类名和图像号
            dstname = '_'.join(["hn1", imgid, cls]) + "." + ext
            # 避免重复操作
            if exists(join(SRC_ROOT, fname)):
                os.rename(join(SRC_ROOT, fname), join(SRC_ROOT, dstname))


def convert_file():
    """转换文件格式"""
    for _, _, files in os.walk(SRC_ROOT):
        for fname in files:
            if not fname.endswith('jpg'):
                srcpath = join(SRC_ROOT, fname)
                dstname = fname.split('.')[0] + '.' + 'jpg'
                img = Image.open(srcpath)
                img.save(join(SRC_ROOT, dstname))
                os.remove(srcpath)

def check_pixel():
    """清除噪声图像"""
    for root, _, files in os.walk(SRC_ROOT):
        for fname in files:
            img = Image.open(join(SRC_ROOT, fname))    # 获取图片对象
            if (Image.Image.getpixel(img, (0,0))) == (0,0,0):    # 删除像素为0的图片
                os.remove(join(SRC_ROOT, fname))
            if (Image.Image.getpixel(img, (0, 0))) == (255, 255, 255):    # 删除像素为255的图片
                os.remove(join(SRC_ROOT, fname))
            if img.size < (32, 32):
                os.remove(join(SRC_ROOT, fname))

DST_ROOT = './space/garbage'

def restruct_folder():
    """重构文件目录"""
    #为了避免重复运行造成文件夹信息混乱
    if exists(DST_ROOT):
        shutil.rmtree(DST_ROOT)

    for _, _, files in os.walk(SRC_ROOT):
        for fname in files:
            temp = fname.replace('.', '_')    # 换成_方便字符串分割
            clsname = temp.split('_')[-2]    # 获取类名
            srcpath = join(SRC_ROOT, fname)    # 设定原文件路径
            dstdir = join(DST_ROOT, clsname)    # 设定目标文件夹，名称为类名

            if not exists(dstdir):    # 生成目标文件夹，判断防止冗余
                os.makedirs(dstdir)

            shutil.copyfile(srcpath, join(dstdir, fname))    # 复制文件

FINAL_ROOT = './space/garbage-final'

def split_dataset():
    """划分训练测试集"""
    import random
    #为了避免重复运行造成文件夹信息混乱
    if exists(FINAL_ROOT):
        shutil.rmtree(FINAL_ROOT)

    random.seed(1)  # 随机种子
    # 1.确定原图像数据集路径
    dataset_dir = DST_ROOT
    # 2.确定数据集划分后保存的路径
    split_dir = join('./flowers')
    train_dir = join(split_dir, "train")
    test_dir = join(split_dir, "test")
    # 3.确定将数据集划分为训练集，测试集的比例
    train_pct = 0.8
    test_pct = 0.2
    # 4.划分
    for root, dirs, files in os.walk(DST_ROOT):
        for sub_dir in dirs:  # 遍历0，1，2，3，4，5文件夹
            imgs = os.listdir(os.path.join(root, sub_dir))  # 展示目标文件夹下所有的文件名
            imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))  # 取到所有以.jpg结尾的文件，如果改了图片格式，这里需要修改
            random.shuffle(imgs)  # 乱序图片路径
            img_count = len(imgs)  # 计算图片数量
            train_point = int(img_count * train_pct)
            test_point = int(img_count * test_pct)

            for i in range(img_count):
                if i < train_point:  # 保存train_point的图片到训练集
                    out_dir = os.path.join(train_dir, sub_dir)
                else:  # 保存test_point结束的图片到测试集
                    out_dir = os.path.join(test_dir, sub_dir)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)      # 创建文件夹
                target_path = os.path.join(out_dir, imgs[i])  # 指定目标保存路径
                src_path = os.path.join(dataset_dir, sub_dir, imgs[i])  # 指定目标原图像路径
                shutil.copy(src_path, target_path)  # 复制图片

            print('Class:{}, train:{}, test:{}'.format(sub_dir, train_point, img_count - train_point))

if __name__ == '__main__':

    #find_unlabeled()
    #补充缺失的真实标签：可能有多个，需要多次调用label_img函数，修改成功后无需再次调用
    label_img('0048.jpg', "rose")
    label_img('0080.jpg', "sunflower")
    regulate_filename()
    convert_file()
    check_pixel()
    restruct_folder()
    split_dataset()
