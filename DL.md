# 深度学习笔记

跟着李沐《动手学深度学习》(d2l.ai)，配合代码实践。

## 1.torch基础

**辅助函数**

> ```dir()```函数能让我们知道包和包内的函数
> ```help()```函数能让我们知道每个包内的函数如何使用

### 1.1 reshape与view的区别


| 操作               | 创建新的Tensor对象 | 共享Storage                      | 修改b会不会影响a               | 限制                    |
| -------------------- | -------------------- | ---------------------------------- | -------------------------------- | ------------------------- |
| b = a              | 同一个对象，不创建 | 共享                             | 有影响                         | 无                      |
| b = a.view(...)    | 创建新的Tensor对象 | 共享                             | 有影响                         | a必须contiguous内存连续 |
| b = a.reshape(...) | 创建新的Tensor对象 | 如果内存不连续就创建新的内存空间 | 不一定，看是否创建新的内存空间 | 无                      |
| b = a.clone()      | 创建新的Tensor对象 | 不共享                           | 不影响                         | 无                      |

> `view()`只能作用域内存连续的对象，而`reshape()`能作用于所有的对象，如果内存连续就直接指向该内存，如果不连续就另外创建内存。

### 1.2 数据读取

**OS包读取文件路径**

- **```os.listdir(dir_path)```将dir_path中的文件名以字符串列表的形式返回。**

```python
#导入包
from PIL import Image
import os
#路径
img_path = r'data'
#把该目录的所有文件以列表的形式返回
img_path_list = os.listdir(img_path)
#输出相关信息
print(img_path_list)
print(img_path_list[0])
print(type(img_path_list))
#输出如下：
#['cifar-10-batches-py', 'cnn_demo_image.png', 'Kaggle_First_Project']
#cifar-10-batches-py
#<class 'list'>
```

- **```os.path.join(img_dir, img_path)```将img_dir和img_path拼接起来。**

```python
import os
img_dir = r'data'
img_path = r'cnn_demo_image.png'
path = os.path.join(img_dir, img_path)
print(path)
#输出 data/cnn_demo_image.png
```

### 1.3Dataset类如何重定义

**自定义Dataset类必须重写的三个函数**

- ```__init__()```初始化函数。
- ```__getitem__()```索引函数
- ```__len__()```获取长度函数

```python
from PIL import Image
import os
from torch.utils.data import Dataset
#创建该类
class imgdataset(Dataset):
    #初始化
    def __init__(self, root_path, label_path):
        #获得根目录
        self.root_path = root_path
        self.label_path = label_path
        #获得路径
        self.path = os.path.join(root_path, label_path)
        #获得该路径下的文件名列表 [str, str, str, ...]
        self.img_path = os.listdir(self.path)
      

    def __getitem__(self, idx):
        #根据输入的idx找到对应的文件名
        img_name = self.img_path[idx]
        #将文件名填入路径中，找到该文件的准确路径
        img_path = os.path.join(self.path, img_name)
        #打开图片
        img = Image.open(img_path)
        #返回label
        label = self.label_path
        #__getitem__通常返回特征和标签，在这里是img和label
        return img, label

    def __len__(self):
        return len(self.img_path)
  

def main():
    #实例化一个dataset
    ant_dataset = imgdataset(r'hymenoptera_data\train', r'ants')
    #获取第一个dataset
    img, label = ant_dataset[0]
    img.show()
    print(label)

if __name__ == '__main__':
    main()
```

**冷知识**

```python
#dataset之间可以进行相加拼接
train_dataset = ants_dataset + bees_dataset
```

### 1.4TensorBoard的使用（可视乎loss函数）

**TensorBoard网页使用指南**


| 需求                 | 操作                                                       |
| ---------------------- | ------------------------------------------------------------ |
| 曲线太抖看不清       | 调大 Smoothing 滑块                                        |
| 想对比两次训练       | 把日志分别存`logs/exp1`、`logs/exp2`，TensorBoard 自动叠图 |
| 想确认模型结构对不对 | 看一眼 GRAPHS 面板，双击展开                               |
| 想下载图片           | 右下角有个下载按钮                                         |
| 想定位某个具体的数值 | 鼠标悬停在曲线上                                           |

**导入TensorBoard包**

```python
from torch.utils.tensorboard import SummaryWriter
```

**SummaryWrite类**

> 类似于可视化包，用于观察训练时loss下降情况。

**add_scalar()**

```python
#导入包
from torch.utils.tensorboard import SummaryWriter
#实例化SummaryWriter对象
write = SummaryWriter('logs')
#绘制图像
for i in range(100):
    write.add_scalar('y = x', i, i)
#关闭write
write.close()
```

> 运行完之后会生成一个logs文件夹，接下来需要在目录中运行如下指令

```bash
tensorboard --logdir=logs
```

> 然后会输出一个端口路径，打开即可查看可视化图像。

### 1.5Transforms类（Image预处理工具包）

**Transforms类通常内置于Dataset类中做一个内置工具，当取数据的时候，就会把数据放入Transforms中加工然后再拿出来。**
**其中```__getitem__()```函数是用来取Dataset中数据的函数，这个函数中往往需要用Transforms来对数据进行处理。**

```python
class MyDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform  # 存起来

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]            # 原始图片
        label = self.labels[idx]

        if self.transform:              # 有 transform 就用
            img = self.transform(img)

        return img, label
```

 **Transforms通常用于对图像进行预处理和数据增强**

- **统一输入尺寸**：将不同大小的图片缩放、裁剪到模型所需的固定尺寸（如``` Resize```、```CenterCrop```）。
- **转为张量**：将 PIL 图像或 NumPy 数组转换为 PyTorch / TensorFlow 张量，并调整数据范围（如 ```ToTensor``` 将 [0,255] 转为 [0,1]）。
- **数据增强**：在训练阶段通过随机变换（如随机翻转、旋转、色彩抖动、仿射变换等）生成多样化的样本，提升模型泛化能力，减少过拟合。
- **格式转换**：如灰度化（Grayscale）、转换为 PIL 图像等。

```python
from torchvision import transforms
import os
from PIL import Image
#构建文件路径
root_path = r'hymenoptera_data\train'
label_path = r'ants'
img_name = r'0013035.jpg'
img_path = os.path.join(root_path, label_path, img_name)
print(img_path)
#导入图像文件
img = Image.open(img_path)
print(img)
#实例化工具对象
t = transforms.ToTensor()
#进行转换
img_tensor = t(img)
print(img_tensor.shape)
print(img_tensor)
```

**Compose函数**

> ```Compose()```可以将参数中的操作一次性执行完，也就是他可以组合多个工具，形成一个超级工具类。前面的```ToTensor()```只能算是一个转化为Tensor的工具，**而Compose可以组合多个工具。**

```python
from torchvision import transforms
# 定义一个由 Compose 包装的预处理流程
transform = transforms.Compose([
    transforms.Resize(256),          # 步骤1: 将图像最短边缩放到256像素
    transforms.CenterCrop(224),      # 步骤2: 从图像中心裁剪出224x224的区域
    transforms.ToTensor(),           # 步骤3: 将PIL图像或NumPy数组转换为PyTorch张量
    transforms.Normalize(            # 步骤4: 用均值和标准差标准化张量
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# 假设有一张图片 'img'
# transformed_img = transform(img)  # 一次性完成所有步骤
```

**Transforms常用的工具类**

- ```ToTensor()```将Image对象转化为Tensor。
- ```ToPILImage()```讲Tensor或者Ndarray对象转化为Image对象。
- ```Normalize()```对一个Tensor的image进行归一化。

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),           # 将 PIL 图像转为 [0,1] 的张量
    transforms.Normalize(            # 标准化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
#mean表示每个通道的均值，std表示每个通道对应的方差。
#mean[0]和std[0]表示第0个通道的均值和方差，然后这个通道的所有值对这个均值和方差进行归一化。
# 假设 img 是一个 PIL Image 或 np.array
tensor_img = transform(img)   # 输出张量，每个通道满足近似 N(0,1) 分布
```

- ```Resize()```把Image对象重塑为指定大小。

### 1.6DataLoader类（数据迭代器）

**类比**

> Dataset是一副牌，Dataset中的每一个样本是一张牌，那么DataLoader就是发牌器。
>
> **dataset参数**：告诉发牌器，你的牌是哪一副。
>
> **batch_size参数**：告诉发牌器，一次发多少张。
>
> **shuffle参数**：告诉发牌器，是否洗牌（打乱数据顺序）
>
> **drop_last参数**：如果有100张牌，每次发3张，必定会余1张牌，那么当drop_last=True时会抛弃（drop）最后一张牌，当drop_last=False时不抛弃最后一张牌。
>
> **num_workers参数**：工作线程数，如果工作量太大可以通过增加num_workers来减少工作时间。
>
> **pin_memory = True参数**：把数据锁在CPU固定内存内，往GPU拷贝的时候更快，配合```.to(m_device, non_blocking=True)```（锁定存储数据的内存页，不会被替换到磁盘中），效果更佳。
>
> **persistent_workers = True参数**：（避免每个epoch都重开进程）
>
> **prefetch_factor = 4参数**：预获取更多batch的数据，减少GPU等待时间，默认2。

```python
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)
```

```python
from torch.utils.data import Dataset, DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
test_data = torchvision.datasets.CIFAR10(r'./data', train = False, transform = torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset = test_data, batch_size = 4, drop_last = False)
write = SummaryWriter('logs')
step = 0
for img, label in dataloader:
  
    write.add_images('test_data', img_tensor = img, global_step = step)
    step += 1
write.close()
```

### 1.7 三者关系（Transform、Dataset、DataLoader）（究极重点）

#### 1.7.1 一句话关系

```
原始数据 ──(Transform)──► Dataset ──(DataLoader)──► 模型
```

- **Transform**：嵌入到Dataset中的转化工具，对Dataset中要取出的数据进行转化。
- **Dataset**：数据的"仓库"，负责存取和索引
- **DataLoader**：仓库的"物流"，负责分批、打乱、并行运输

---

#### 1.7.2 各自职责

##### 1. Transform —— 加工工人


| 类别     | 例子                        | 作用                                                 |
| ---------- | ----------------------------- | ------------------------------------------------------ |
| 类型转换 | `ToTensor()`                | PIL Image → Tensor ([0,255] uint8 → [0,1] float32) |
| 归一化   | `Normalize(mean, std)`      | 把像素值拉到均值为0、标准差为1的分布                 |
| 数据增强 | `RandomHorizontalFlip()`    | 随机翻转，同一张图每次不同，防止过拟合               |
| 数据增强 | `RandomCrop(32, padding=4)` | 随机裁剪                                             |
| 数据增强 | `ColorJitter(...)`          | 随机颜色扰动                                         |

> **关键：** Transform 在 Dataset 创建时存起来，在 `__getitem__` 被调用时才执行。

##### 2. Dataset —— 数据仓库

```text
┌─────────────────────────────────┐
│           Dataset               │
│                                 │
│  __init__:  存路径/标签/transform │
│  __len__:   返回数据总量          │
│  __getitem__: 取第 i 条，调 transform│
│                                 │
│  dataset[i] 返回 (tensor, label) │
└─────────────────────────────────┘
```

- **Dataset 本身不是数据**，是数据的**容器 + 索引规则**
- Transform 只是它的一个工具，存在 `self.transform` 里

##### 3. DataLoader —— 物流系统

```text
┌──────────────────────────────────────────┐
│               DataLoader                 │
│                                          │
│  从 Dataset 里取数据：                     │
│  ① shuffle（打乱顺序）                     │
│  ② 按 batch_size 分组                     │
│  ③ 把每组样本拼成一个大 tensor             │
│  ④ num_workers 并行加载（可选）            │
│  ⑤ drop_last 丢弃最后凑不齐的零头（可选）   │
│                                          │
│  输出：(batch_tensor, batch_labels)       │
└──────────────────────────────────────────┘
```

---

#### 1.7.3 数据流转全景图

```text
磁盘上的图片文件 (PIL Image)
        │
        │  dataset[i] 被调用时
        ▼
   ┌─────────────┐
   │  Transform   │   ToTensor() / Normalize() / RandomFlip()
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │   Dataset    │   存图片路径+标签，__getitem__ 里调 transform
   │   (仓库)     │   dataset[i] → (tensor, label)
   └──────┬──────┘
          │
          │  DataLoader 包在外面，循环取 batch
          ▼
   ┌─────────────┐
   │  DataLoader  │   shuffle → batch → 拼接 → 并行加载
   │   (物流)     │   for batch in dataloader → 喂给模型
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │    模型      │
   └─────────────┘
```

---

#### 1.7.4 代码对照

```python
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

# ========== 1. Transform：定义加工规则 ==========
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),           # 数据增强
    transforms.RandomCrop(32, padding=4),        # 数据增强
    transforms.ToTensor(),                       # 类型转换：PIL → Tensor
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))  # 归一化
])

test_transform = transforms.Compose([
    transforms.ToTensor(),                       # 测试集不需要增强！
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# ========== 2. Dataset：数据仓库，绑定 transform ==========
train_dataset = torchvision.datasets.CIFAR10(
    root='data',
    train=True,
    transform=train_transform    # transform 存进 dataset，用的时候才执行
)

test_dataset = torchvision.datasets.CIFAR10(
    root='data',
    train=False,
    transform=test_transform
)

# ========== 3. DataLoader：物流，包住 dataset ==========
train_loader = DataLoader(
    train_dataset,        # ← 包的是 Dataset
    batch_size=64,        # ← 每批 64 张
    shuffle=True,         # ← 训练打乱
    num_workers=2         # ← 2个进程并行加载
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,        # ← 测试不用打乱
    num_workers=2
)

# ========== 4. 训练循环 ==========
for epoch in range(10):
    for imgs, labels in train_loader:   # 每次取一个 batch
        # imgs.shape = (64, 3, 32, 32)
        # labels.shape = (64,)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        ...
```

---

#### 1.7.5 自定义 Dataset 示例

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images        # 原始数据（路径或 PIL Image）
        self.labels = labels
        self.transform = transform  # 把 transform 存起来

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]      # 取原始图
        label = self.labels[idx]

        if self.transform:          # transform 在这里才真正执行
            img = self.transform(img)

        return img, label           # 返回 (处理后的tensor, 标签)
```

---

#### 1.7.6 常见误区


| 误区                                 | 正解                                                                   |
| -------------------------------------- | ------------------------------------------------------------------------ |
| "Dataset 里面存的是 Tensor"          | 不一定。存的是原始数据（路径/PIL），取的时候才经 transform 转成 Tensor |
| "transform 把 Dataset 变成了 Tensor" | transform 作用在**每一张图上**，不改变 Dataset 本身的类型              |
| "DataLoader 只是分批"                | 还做了 shuffle、并行加载、自动拼接、drop_last 等                       |
| "测试集也要做数据增强"               | ❌ 测试集只需要`ToTensor()` + `Normalize()`，不要随机变换              |

### 1.8 神经网络的搭建

#### 1.8.0 官方文档访问

[中文文档](https://docs.pytorch.ac.cn/docs/2.12/index.html)

> 按住CTRL左键点击。

#### 1.8.1 nn.Module类

**所有模型的父类**

**搭建模型时要重写一下函数：**

- ```__init__()```：初始化函数。
- ```forward()```：前向传播函数。

#### 1.8.2 卷积层API使用

**常用Conv2d，二维卷积，对应图片。也有一维和三维的，三维对应视频或者其他的有第三个维度的数据类型。**

**常用参数**

- in_channels：输入通道数。
- out_channels：输出通道数。
- kernel_size：卷积核大小，可以填入标量也可以填入向量。例如标量参数3表示3×3、向量参数(3, 5)表示3×5的卷积核。
- stride：表示卷积核的步幅。
- padding：表示周围填充的数量。

#### 1.8.3 池化层API使用

**常用参数**

- kernel_size：池化核大小。
- stride：移动步幅。
- padding：周围填充大小。

#### 1.8.4 其他常用层API使用

**BN层**

> 用于对数据做归一化。

```python
#num_feature表示表示特征数，在卷积神经网络中表示通道数。
nn.BatchNorm2d(num_feature, ...)
```

**Recurrent Layers（处理自然语言的网络层）**

具体查官方文档

**Transformer Layers（自注意力）**

具体查看官方文档

**Linear Layers（线性层）（常用）**

具体查看官方文档

**Dropout Layers（丢弃层）（常用的正则化方式）**

具体查看官方文档

**Sequential**

```python
#构建一个简单的网络
model = nn.Sequential(
	nn.Conv2d(), 
	nn.ReLU(), 
	nn.Conv2d(), 
	nn.Dropout(), 
	nn.ReLU()
)
```

具体查看官方文档

**Flatten层**

**作用**

> **连接卷积部分与全连接部分**，将多维张量转化为1维向量。

例如：输入特征图：```(4, 4, 32)```，输出```(4*4*32)=512```的一维向量。

#### 1.8.5 优化器（optim）

```python
#优化器实例化。
#model.parameters()，必须把模型的参数放进去，这样优化器才知道这个模型的参数有哪些，才能进行参数更新。
optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9)
optimizer = optim.Adam([var1, var2], lr = lr)
#zero_grad()函数用来对梯度清理，防止前一轮学习的梯度影响当前的计算。
optimizer.zero_grad()
#利用损失函数的backward()前向传播函数进行梯度计算。
loss.backward()
#step函数，用损失函数计算出的梯度对参数进行更新。
optimizer.step()
```

#### 1.8.6 模型的保存和模型的加载

```python

#模型保存，输入模型的名字和保存路径即可，路径要以pth格式保存。
#方式1，保存模型的结构和参数。
torch.save(model_name, path)
#方式2，仅仅保存模型的参数（官方推荐，因为这样可以减少内存占用）
torch.save(model_name.state_dict(), path)
#导入模型
#如果path中保存了网络结构，那么model可以直接使用
#如果path中仅仅保存了网络参数，那么model还不能直接使用，model只是个参数字典。
#方式1，导入含有网络结构的pth文件
model = torch.load(path)
#方式2，导入仅含有网络参数的pth文件
#先创建网络结构
model = torchvision.models.resnet18(pretrained = False)
#导入网络参数
model.load_state_dict(torch.load(path))
```

#### 1.8.7数据增强API（Transforms类）

**`torchvision.transforms.v2` 常用数据增强 API 速查笔记**

---

**`Compose`**组合多个变换

- `transforms` (list): 变换列表

**`RandomHorizontalFlip`**随机水平翻转

- `p` (float): 翻转概率，默认 0.5

**`RandomVerticalFlip`**随机垂直翻转

- `p` (float): 翻转概率，默认 0.5

**`RandomRotation`**随机旋转

- `degrees` (float/tuple): 旋转角度范围，如 30 表示 [-30,30]
- `expand` (bool): 是否扩大画布以适应旋转，默认 False
- `fill` (int/tuple): 填充像素值，默认 0
- `antialias` (bool): 是否抗锯齿，推荐 True

**`RandomCrop`**随机裁剪

- `size` (int/tuple): 裁剪后尺寸
- `padding` (int/tuple): 裁剪前填充大小，默认 None
- `pad_if_needed` (bool): 若图像小于目标尺寸是否填充，默认 False

**`RandomResizedCrop`**随机裁剪后缩放到固定尺寸

- `size` (int/tuple): 输出尺寸
- `scale` (tuple): 裁剪面积相对于原图的比例范围，默认 (0.08, 1.0)
- `ratio` (tuple): 宽高比范围，默认 (3/4, 4/3)
- `antialias` (bool): 是否抗锯齿，推荐 True

**`ColorJitter`**随机调整颜色属性

- `brightness` (float/tuple): 亮度调整因子，0 为全黑，1 为原图；若为单数 a 表示 [max(0,1-a), 1+a]
- `contrast` (float/tuple): 对比度，同上
- `saturation` (float/tuple): 饱和度，同上
- `hue` (float/tuple): 色相偏移范围，最大不超过 0.5

**`RandomGrayscale`**随机转灰度图

- `p` (float): 转换概率，默认 0.1

**`RandAugment`**自动搜索增强策略（RandAugment）

- `num_ops` (int): 每张图像应用的操作数量，默认 2
- `magnitude` (int): 增强强度，常用 1~10，默认 9
- `num_magnitude_bins` (int): 强度分级数，默认 31

**`AutoAugment`**自动搜索策略（AutoAugment）

- `policy` (AutoAugmentPolicy): 预设策略，可选 `AutoAugmentPolicy.IMAGENET`、`CIFAR10`、`SVHN`

**`AugMix`**混合多种增强图像

- `severity` (int): 增强链的强度，默认 3
- `mixture_width` (int): 混合链的条数，默认 3
- `alpha` (float): 混合权重的分布参数，默认 1.0

**`ToDtype`**转换张量数据类型（推荐替代旧版 `ToTensor`）

- `dtype` (torch.dtype): 目标类型，如 `torch.float32`
- `scale` (bool): 如果输入是 uint8 且目标为 float，是否自动除以 255，默认 False

**`Normalize`**按均值和标准差归一化

- `mean` (sequence): 各通道均值
- `std` (sequence): 各通道标准差
- `inplace` (bool): 是否原地操作，默认 False

**`Resize`**缩放图像

- `size` (int/tuple): 目标尺寸
- `antialias` (bool): 是否抗锯齿，推荐 True

**`CenterCrop`**中心裁剪

- `size` (int/tuple): 裁剪尺寸

**`Pad`**填充图像边界

- `padding` (int/tuple): 填充大小
- `fill` (int/tuple): 填充像素值
- `padding_mode` (str): 填充模式，支持 `constant`, `edge`, `reflect`, `symmetric`

**`GaussianBlur`**高斯模糊

- `kernel_size` (int/tuple): 高斯核大小，必须为奇数
- `sigma` (float/tuple): 高斯标准差，若为单数则表示范围 [0, sigma]

**`RandomApply`**以一定概率应用一组变换

- `transforms` (list): 变换列表
- `p` (float): 应用概率

---

**使用示例**（标准流水线）

```python
from torchvision.transforms import v2

train_transform = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

#### 1.8.7 GPU训练API

只有三个东西需要送进GPU

1. 网络模型
2. 数据
3. 损失函数

```python
model.cuda()
loss.cuda()
feature.cuda()
label.cuda()
#严谨的写法
if torch.cuda.is_available():
	#送入cuda中
else:
	#不送入cuda，在CPU中运行
```

### 1.9 网络设计思路

---

#### 1.9.0 总论：网络设计的核心思维

**不要把网络设计当成"调参数"，要当成"设计信息管道"。**

每一个神经网络本质上做的是一件事：**把原始数据逐步变换成目标形式**。设计网络时，你问自己的第一个问题不应该是"用几层"，而是：

> **"输入数据的结构是什么？输出需要什么形式？中间需要经历怎样的信息变换？"**

这句话听起来抽象，但它是所有网络设计的总纲。下面逐一拆解。

**三大设计原则（适用所有架构）**


| 原则             | 含义                                   | 反面教材                                    |
| ------------------ | ---------------------------------------- | --------------------------------------------- |
| **结构匹配数据** | 网络的归纳偏置要匹配数据的结构特性     | 用 MLP 处理图像（无视空间局部性，参数爆炸） |
| **容量匹配任务** | 网络的参数量要匹配任务的复杂度和数据量 | 1000 条数据用 ResNet-152（严重过拟合）      |
| **梯度流通畅**   | 反向传播时梯度能从输出顺畅传到输入     | 深层 Sigmoid 网络没有残差连接（梯度消失）   |

**拿到一个新问题时，按这个流程走：**

```
1. 分析数据结构：输入是什么类型？（图像/表格/文本/图/...）
   → 这决定了你选哪一类基础架构

2. 分析任务类型：输出是什么？（分类/回归/生成/检测/...）
   → 这决定了输出层的设计

3. 评估数据规模：训练样本有多少？
   → 这决定了网络可以有多深、多宽

4. 设计管道：输入 → 中间表示 → 输出，每一层的 shape 变化是什么？
   → 在纸上画出来，确认每一层的输入输出维度

5. 用一个 batch 的假数据做 dry-run，验证 shape 正确
```

---

#### 1.9.1 图像网络设计思路（CNN）

**适用场景**：图像分类、目标检测、图像分割、图像生成

**数据的结构特性**：

- **空间局部性**：相邻像素相关性强，远处的像素相关性弱
- **平移不变性**：一只猫在图片左边还是右边，都是猫
- **层次化特征**：边缘 → 纹理 → 局部形状 → 全局语义

**核心设计模式：空间压缩 + 通道扩张**

```
原始图像 (H × W × 3)
    ↓  空间大，语义浅
  浅层卷积（小通道数，捕捉边缘/纹理）
    ↓  空间缩小，语义加深
  中层卷积（通道翻倍，组合成局部形状）
    ↓  空间再缩小，语义更深
  深层卷积（通道再翻倍，组合成全局语义）
    ↓  空间最小，语义最抽象
  全局池化 → 分类输出
```

**五条具体设计规则**

**规则 1：通道数的翻倍节奏**

```
第一层：输入 3(RGB) → 输出 16 或 32
之后每降一次空间分辨率：通道翻倍
32 → 64 → 128 → 256 → 512
```

为什么翻倍？当你用 stride=2 把空间砍半（H/2 × W/2），信息量减少到 1/4。通道翻倍（×2）只能补偿一半的信息容量损失——这是故意设计的有损压缩，迫使网络保留最重要的语义信息，丢弃噪声。

**规则 2：卷积核大小选择**


| 核大小      | 何时用            | 原因                                                              |
| ------------- | ------------------- | ------------------------------------------------------------------- |
| 3×3        | **默认选择**      | 最小能捕捉空间关系的核，两个 3×3 的感受野 = 一个 5×5，参数更少  |
| 1×1        | 调整通道数 / 降维 | 不改变空间，只做通道间的线性组合（NiN、ResNet bottleneck 的核心） |
| 5×5 / 7×7 | 第一层偶尔用      | 输入图大时快速扩大感受野，但现代网络也多用 3×3 堆叠替代          |

几乎永远不需要 2×2、4×4、6×6 的卷积核。3×3 是过去十年被验证的最优默认值。

**规则 3：降空间分辨率的方式和时机**

两种方式，效果近似，选一种保持一致：

- `Conv2d(..., stride=2)` — 可学习的降采样
- `MaxPool2d(2, 2)` — 固定降采样

降采样的时机：不要在第一个卷积层立刻降，先让网络在原始分辨率上提取一些特征：

```
Conv(保持空间) → Conv(保持空间) → 降采样(空间减半+通道翻倍) → 重复
```

**规则 4：用"块"思维搭网络**

不要一个 `nn.Sequential` 从头写到尾。把反复出现的模式封装成块：

```python
# 基础卷积块
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    )

# 降采样块
def downsample_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    )
```

**规则 5：用全局平均池化代替全连接**

```python
# 旧做法（VGG/AlexNet）：Flatten → 巨大的全连接层 → 参数爆炸
# 新做法（ResNet/现代CNN）：
nn.AdaptiveAvgPool2d(1)   # H×W×C → 1×1×C
nn.Flatten()               # → C 维向量
nn.Linear(C, num_classes)  # 轻量分类头
```

好处：参数少、不容易过拟合、允许任意输入尺寸。

**规则 6：降采样次数怎么定？**


| 输入尺寸            | 降采样次数 | 最终特征图     | 适用任务          |
| --------------------- | :----------: | ---------------- | ------------------- |
| 32×32 (CIFAR-10)   |    3 次    | 4×4           | 小图分类          |
| 224×224 (ImageNet) |    5 次    | 7×7           | 标准分类          |
| 512×512            |   5-6 次   | 8×8 或 16×16 | 高分辨率分类/检测 |

降采样到特征图大约 4×4 ~ 8×8 时接入全局池化。

**实战示例：为新任务快速搭一个 CNN**

CIFAR-10（32×32，10 类）的设计推导：

```
输入 32×32×3
  ↓ 规则3：先不降采样，在原始分辨率提取特征
  Conv(3→32, 3×3, pad=1) → 32×32×32
  Conv(32→32, 3×3, pad=1) → 32×32×32
  ↓ 规则1+3：降空间 + 翻通道
  stride=2, 32→64 → 16×16×64
  Conv(64→64, 3×3, pad=1) → 16×16×64
  ↓ 再降 + 再翻
  stride=2, 64→128 → 8×8×128
  Conv(128→128, 3×3, pad=1) → 8×8×128
  ↓ 再降 + 再翻
  stride=2, 128→256 → 4×4×256
  ↓ 规则5
  AdaptiveAvgPool2d(1) → 1×1×256 → Flatten → 256
  ↓
  Linear(256, 10)
```

管道：32→16→8→4（三次降采样），通道 3→32→64→128→256。

**现有经典架构的位置（你的知识地图）**

你已经在第 8 章学过的网络，每个解决了一个核心问题：


| 网络      | 解决了什么问题                | 设计创新                 |
| ----------- | ------------------------------- | -------------------------- |
| LeNet     | 奠定 CNN 基本范式             | 卷积→池化→全连接       |
| AlexNet   | 证明深度 CNN 能处理大规模图像 | ReLU、Dropout、GPU       |
| VGG       | 证明深度有用，架构可以很规整  | 全部 3×3、VGG 块        |
| NiN       | 去掉昂贵的大全连接层          | 1×1 卷积 + 全局池化     |
| GoogLeNet | 同一层用不同感受野并行        | Inception 多分支         |
| ResNet    | 让 100+ 层网络能训练          | 残差连接（梯度高速公路） |
| BN        | 让深层网络稳定训练            | 归一化 + 可学习缩放平移  |

---

#### 1.9.2 表格数据网络设计思路（MLP）

**适用场景**：房价预测、用户分类、信用评分、推荐系统的特征交叉——任何"Excel 表格"形式的数据

**数据的结构特性**：

- **无空间/时间结构**：列之间没有固定的拓扑关系（不像图像的相邻像素或文本的先后顺序）
- **特征独立但语义相关**：年龄和收入虽然不"相邻"，但逻辑上有关系
- **混合数据类型**：数值型（年龄 25）+ 类别型（城市=北京）+ 缺失值

**核心设计模式：逐步压缩维度**

```
输入 (N个特征，可能几百到几千维)
    ↓  宽→窄，每一层压缩一些
  隐藏层1：宽（让网络有足够容量发现特征间的组合关系）
    ↓
  隐藏层2：中（压缩冗余信息）
    ↓
  隐藏层3：窄（提炼最关键信息）
    ↓
  输出层
```

**四条设计规则**

**规则 1：层数和宽度的选择**


| 数据量       | 网络规模               | 例子                 |
| -------------- | ------------------------ | ---------------------- |
| < 1000 条    | 2-3 层，每层 64-128    | 小规模回归/分类      |
| 1000-10万 条 | 3-5 层，每层 128-512   | 房价预测、信用评分   |
| > 10万 条    | 5-10 层，每层 256-1024 | 推荐系统、大规模 CTR |

**表格数据第一铁律**：数据量小的时候，网络一定要窄要浅。表格数据比图像容易过拟合得多——因为表格特征已经是被人工提取过的"高层语义"，网络需要做的只是发现特征之间的组合关系，不需要像 CNN 那样从像素重建语义。

**规则 2：形状——漏斗型 vs 长方体型**

```
漏斗型（推荐，大多数情况）：
in_features → 256 → 128 → 64 → 1

长方体型（数据量大且特征维度高时）：
in_features → 256 → 256 → 256 → 1

膨胀型（几乎不用，除非做自编码器的编码部分）：
不要用
```

默认使用漏斗型。每一层压缩 30%-50% 的神经元数量。

**规则 3：激活函数选择**


| 激活函数   | 适用场景               | 原因                          |
| ------------ | ------------------------ | ------------------------------- |
| ReLU       | 默认选择               | 简单高效，大多数表格任务够用  |
| Leaky ReLU | 担心神经元死亡         | 负半轴有小斜率                |
| GELU       | 大规模数据 / 深层网络  | 比 ReLU 更平滑，现代 MLP 常用 |
| Tanh       | 输入已归一化到 [-1, 1] | 对称性好，但容易饱和          |

**规则 4：表格数据特有的预处理（比网络设计更重要）**

> **表格数据的性能差距，80% 来自预处理，20% 来自网络设计。**


| 步骤         | 操作                                          | 原因                                |
| -------------- | ----------------------------------------------- | ------------------------------------- |
| 缺失值处理   | 数值型填中位数/均值，类别型加一列`is_missing` | 神经网络不能处理 NaN                |
| 数值型归一化 | StandardScaler 或 MinMaxScaler                | 不同量纲的特征会让梯度不稳定        |
| 类别型编码   | 低基数（<50类）用 one-hot，高基数用 embedding | one-hot 维度爆炸时 embedding 更高效 |
| 标签异常值   | 房价预测等任务，标签取 log1p                  | 标签跨度大时防止梯度爆炸            |
| 特征工程     | 根据业务知识构造交叉特征                      | 让网络的"发现关系"工作变简单        |

**实战示例：Kaggle 房价预测的网络设计**

你之前做的房价预测就是一个典型的表格网络：

```python
class Model(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.layout1 = nn.Linear(in_features, 200)   # 宽→中
        self.layout2 = nn.Linear(200, 100)            # 中→窄
        self.layout3 = nn.Linear(100, 1)              # 窄→输出
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.tanh(self.layout1(x))
        x = self.dropout(x)
        x = torch.tanh(self.layout2(x))
        x = self.dropout(x)
        x = self.layout3(x)  # 最后一层不用激活
        return x
```

设计逻辑：

- 输入特征约 300 维（one-hot 后）
- 漏斗形 200→100→1
- Dropout(0.2) 防止过拟合（数据只有 1168 条）
- 最后一层不用激活函数（回归任务，输出无界）
- Tanh 配合 xavier 初始化

**表格网络 vs 树模型的权衡**


|                           | 神经网络 (MLP)               | 树模型 (XGBoost/LightGBM)  |
| --------------------------- | ------------------------------ | ---------------------------- |
| 数据量小 (< 1万)          | 容易过拟合，需要仔细调参     | **通常更优**，天然抗过拟合 |
| 数据量大 (> 10万)         | **通常更优**，能学到复杂交互 | 训练慢，可能欠拟合         |
| 特征含义模糊（embedding） | **天然支持**，可以端到端学习 | 不擅长处理稠密 embedding   |
| 需要在线更新              | **支持**，SGD 增量学习       | 不支持增量更新             |
| 可解释性                  | 差                           | **好**，特征重要性一目了然 |

---

#### 1.9.3 语言/序列网络设计思路（RNN/LSTM/Transformer）

**适用场景**：文本分类、情感分析、机器翻译、文本生成、时间序列预测

**数据的结构特性**：

- **时序依赖性**：第 t 个词的语义取决于前面的词（"我不喜欢"中的"不"翻转了"喜欢"的语义）
- **长距离依赖**：一段话开头的主语可能影响 100 个词之后的谓语
- **变长输入**：每条文本长度不同，不像图像有固定尺寸
- **离散性**：词是离散符号，不是连续数值，需要先变成向量（Embedding）

---

**Part A：RNN / LSTM 设计思路**

**核心设计模式：循环体 + 时间展开**

RNN 的本质是一个**在时间轴上反复使用的同一个网络块**。不是串联很多层不同的网络，而是**一层网络反复调用 T 次**（T = 序列长度）。

```
输入序列: w1  w2  w3  ...  wT
          ↓   ↓   ↓        ↓
Embedding: e1  e2  e3  ...  eT
          ↓   ↓   ↓        ↓
RNN Cell → RNN Cell → RNN Cell → ... → RNN Cell
  ↓h0       ↓h1       ↓h2            ↓hT
           （同一个 Cell，参数共享）

输出：取最后一个隐藏状态 hT → Linear → 分类
     或者：取所有 h1...hT → 每个时刻都输出（序列标注）
```

**RNN 的致命问题：梯度消失/爆炸**

RNN 在时间轴上展开等同于一个 T 层的极深网络。反向传播穿过 T 个时间步（BPTT），连乘 T 次权重矩阵，梯度指数级衰减或爆炸。

**LSTM：用"门"机制解决长距离依赖**

LSTM 引入三个门（遗忘门、输入门、输出门）+ 一个细胞状态 C：

```
遗忘门：决定丢掉哪些旧信息
输入门：决定写入哪些新信息
输出门：决定输出哪些信息
细胞状态 C：一条贯穿时间的高速公路（类似 ResNet 的恒等映射）
```

LSTM 的设计核心是**细胞状态 C 的更新是加法不是乘法**：

```
C_t = f_t * C_{t-1}  +  i_t * C_tilde_t
      ↑ 遗忘旧信息      ↑ 加入新信息
      乘法门控          加法写入
```

加法操作让梯度可以不衰减地穿过时间步——这和 ResNet 的 x+F(x) 是同一个思路。

**RNN/LSTM 设计规则**


| 设计要素            | 建议                                                                               |
| --------------------- | ------------------------------------------------------------------------------------ |
| Embedding 维度      | 50-300（小任务），300-768（大任务）。嵌入维度 = 词表大小的 4 次方根 是一个经验起点 |
| 隐藏层维度          | 128-512（小/中任务），512-2048（大任务）                                           |
| RNN 层数            | 1-3 层。>3 层 RNN 很难训练，需要考虑 LSTM/GRU                                      |
| 双向(Bidirectional) | 分类任务用双向，生成任务用单向（因果约束）                                         |
| 取隐藏状态的方式    | 分类取最后时刻/最大池化/平均池化；序列标注取所有时刻                               |
| 防止过拟合          | Embedding 后加 Dropout，RNN 层之间加 Dropout                                       |

**LSTM vs GRU 的选择**


|        | LSTM                       | GRU                    |
| -------- | ---------------------------- | ------------------------ |
| 参数量 | 4 组门（更多）             | 3 组门（少 25%）       |
| 性能   | 大任务略优                 | 小任务与 LSTM 持平     |
| 何时用 | 数据充足、需要最强建模能力 | 数据量一般、想更快训练 |

**实战示例：IMDB 情感分类（你马上要做的任务）**

```
输入：一段电影评论 (变长文本)
  ↓
Tokenization + 词表映射 → [seq_len] 的整数索引
  ↓
Embedding(vocab_size, 128) → [seq_len, 128]
  ↓
LSTM(128, 256, num_layers=2, bidirectional=True, dropout=0.3)
  → 取出最后一个时间步的隐藏状态（双向拼接后 = 512 维）
  ↓
Dropout(0.5)
  ↓
Linear(512, 1) → Sigmoid → 正面/负面
```

设计逻辑：

- Embedding 128 维（IMDB 词表约 2-5 万，128 维够用）
- LSTM 256 隐藏 + 2 层（2 万条训练数据，不需要太深）
- 双向（分类任务不需要因果约束）
- 高 Dropout（文本任务比图像更容易过拟合）
- 取最后隐藏状态（整个句子 → 一个情感标签）

---

**Part B：Transformer 设计思路（现代 NLP 的绝对主力）**

> Transformer 彻底取代了 RNN/LSTM 成为 NLP 的主流架构。李沐 d2l 第 11 章会讲，暑假第四阶段你也要手写一个。这里先建立设计直觉。

**Transformer 的核心创新：抛弃时间顺序，用"注意力"一步到位**

RNN 最大的痛点是：词 w1 和词 w100 之间的信息要穿过 99 个时间步。Transformer 说：**让每个词直接看到所有其他词**——这就是 Self-Attention。

```
RNN 的信息流：   w1 → w2 → w3 → ... → w100  （串行，99 步）
Transformer：    w1 ↔ w2 ↔ w3 ↔ ... ↔ w100  （并行，1 步全局可见）
```

不再有时间步，不再有循环，所有位置同时处理。训练时可以并行（比 RNN 快得多）。

**Transformer 的完整管道**

```
输入文本: ["我", "爱", "深度", "学习"]
  ↓
Embedding + Positional Encoding → [4, d_model]
（位置编码注入位置信息——因为没有了 RNN 的天然时序，需要手动加）
  ↓
× N 层 Transformer Block:
  ├── Multi-Head Self-Attention（核心：让每个词看到所有其他词并加权）
  ├── Add & Norm（残差连接 + LayerNorm）
  ├── Feed-Forward Network（对每个位置独立做非线性变换）
  └── Add & Norm
  ↓
取 [CLS] token 或平均池化 → Linear → 分类输出
```

**Transformer 设计规则**


| 设计要素                | 小模型（GPT-2 Small） | 中模型（BERT-base） | 大模型（LLaMA-7B） |
| ------------------------- | ----------------------- | --------------------- | -------------------- |
| d_model（隐藏维度）     | 768                   | 768                 | 4096               |
| num_heads（注意力头数） | 12                    | 12                  | 32                 |
| num_layers（层数）      | 12                    | 12                  | 32                 |
| d_ff（FFN 中间维度）    | 3072                  | 3072                | 11008              |
| 参数量                  | 117M                  | 110M                | 7B                 |

**设计规律**：

- d_model 必须能被 num_heads 整除（每个头 d_model/num_heads 维）
- d_ff 通常是 d_model 的 4 倍（经验比例，不要随意改）
- num_layers 决定深度——层数越多，能建模的抽象层次越高
- 小数据用浅层（4-6 层），大数据用深层（12+ 层）

**对于你自己的学习/实验级 Transformer**

```
d_model = 256 或 512
num_heads = 8
num_layers = 4 或 6
d_ff = d_model × 4 = 1024 或 2048
```

这个配置参数在 5M-20M 之间，能在单张 GPU 上轻松训练，足够学习。

**RNN vs LSTM vs Transformer：三种序列架构的演变逻辑**


| 架构        | 核心机制                    | 解决了什么                 | 新问题                      |
| ------------- | ----------------------------- | ---------------------------- | ----------------------------- |
| RNN         | 循环：h_t = f(h_{t-1}, x_t) | 让网络有了"记忆"           | 长序列梯度消失              |
| LSTM        | 门控 + 细胞状态             | C_t 用加法更新，梯度不衰减 | 还是串行，速度慢            |
| Transformer | Self-Attention              | 并行 + 全局依赖 + 可解释   | 计算量 O(n^2)，长序列吃显存 |

**一句话串联**：RNN 给了网络记忆 → LSTM 让记忆能穿越很远的时间 → Transformer 说"为什么还要穿越时间？让所有词一秒钟互通！"

---

#### 1.9.4 其他重要架构（知识地图）

除了图像、表格、语言三大主流架构，还有几个你应当知道存在的方向。现阶段不需要深入，但要知道它们分别解决什么问题。

**1. 图神经网络（GNN / Graph Neural Network）**

**解决的问题**：数据不是规则的网格（图像）或序列（文本），而是一个图——社交网络、分子结构、知识图谱、推荐系统的用户-物品关系。

**数据结构**：节点 + 边，节点之间的连接关系不规则。

**核心思想**：每个节点通过聚合它的邻居节点的信息来更新自己的表示（Message Passing）。

```
节点 A ← 聚合所有邻居 {B, C, D} 的特征 → 更新 A 的表示
```

**典型应用**：药物发现（分子结构 → 药性预测）、社交推荐、蛋白质结构预测。

**何时需要 GNN**：当你的数据天然是一个**图结构**（对象之间有明确的关系连接），而不是独立的行（表格）或网格（图像）。

---

**2. 自编码器（AutoEncoder, AE / VAE）**

**解决的问题**：不是分类/回归，而是**无监督学习**——数据压缩、降噪、异常检测、生成。

**核心设计模式**：

```
输入 x → 编码器(Encoder) → 低维潜在表示 z（瓶颈） → 解码器(Decoder) → 重建 x'
训练目标：让 x' 尽可能接近 x
```

设计逻辑：

- 编码器：逐步压缩维度（漏斗），迫使网络提取最关键的信息
- 瓶颈（bottleneck）：维度远小于输入，这是"压缩"的关键
- 解码器：逐步扩展维度（倒漏斗），从压缩表示重建原始数据

**关键变体**：

- **VAE（变分自编码器）**：瓶颈不是固定向量，而是一个概率分布的参数（μ, σ），可以生成新样本
- **去噪自编码器**：输入加噪声，训练重建原始无噪声数据，学习更鲁棒的特征

---

**3. U-Net（编码器-解码器 + 跳跃连接）**

**解决的问题**：图像分割、图像到图像的翻译（图像去噪、超分辨率）。

**核心设计**：

```
输入图像
  ↓ 编码器（下采样，类似 CNN，提取语义）
  逐层压缩空间 + 增加通道
  ↓ ... ↓ ... ↓ ... ↓
  Bottleneck（最小空间，最高语义）
  ↓ 解码器（上采样，恢复空间）
  逐层恢复空间 + 减少通道
  ↓ ... ↓ ... ↓ ... ↓
  跳跃连接：编码器每层的输出直接拼接到解码器对应层
  ↓
输出（与输入相同空间尺寸）
```

**为什么需要跳跃连接**：编码器下采样丢失了精确的空间位置。解码器做分割时，既需要高层语义（"这是肝脏"），也需要低层空间细节（"边界精确到像素"）。跳跃连接把编码器的低层特征直接传给解码器——类似 ResNet，但跨的层数更多。

---

**4. 生成对抗网络（GAN）**

**解决的问题**：生成逼真的图像、超分辨率、风格迁移。

**核心思想**：两个网络互相对抗——生成器 G 造假图，判别器 D 区分真假。G 和 D 互相博弈，最终 G 生成的图可以以假乱真。

```
随机噪声 z → 生成器 G → 假图
                         ↓
              真假图混在一起 → 判别器 D → 真/假
真实图像 → ------------------↑
```

**一般不自己设计 GAN**：GAN 的训练极不稳定，需要大量调参技巧。知道这个方向存在即可，需要时直接用 StyleGAN 等成熟框架。

---

**5. 扩散模型（Diffusion Models）**

**解决的问题**：图像生成（Stable Diffusion、DALL-E、Midjourney 的基础）。

**核心思想**：先学怎么加噪声（把图一步步毁掉），再学怎么去噪声（把纯噪声一步步恢复成图）。生成时，从纯噪声出发，一步步去噪，最终生成一张真实图片。

这是当前（2024-2025）图像生成领域的最强范式，但数学要求很高（随机微分方程、马尔可夫链），现阶段只需知道它存在。

---

**"我应该学哪个"决策树**

拿不准的时候，按这个走：

```
你的数据长什么样？
├── 图像（规则网格像素）
│   ├── 分类 → CNN (ResNet/VGG)
│   ├── 分割 → U-Net
│   └── 生成 → Diffusion Model / GAN
├── 表格（Excel 行列）
│   ├── 数据量 < 1万 → 先试 XGBoost，再试 MLP
│   └── 数据量 > 10万 → MLP / TabTransformer
├── 文本/序列
│   ├── 序列 → LSTM/GRU（小规模）/ Transformer（大规模）
│   └── 分类/生成 → Transformer 系列
├── 图（社交网络、分子）
│   └── GNN
└── 无标签数据（降维、压缩、异常检测）
    └── AutoEncoder / VAE
```

---

#### 1.9.5 从"会调 API"到"会设计网络"的检查清单

当你拿到一个新问题时，按顺序回答下面 7 个问题。回答完，网络的大致结构就有了：

1. **输入是什么类型？**（图像/表格/文本/其他）
2. **输出是什么类型？**（分类→Softmax+类别数 / 回归→无激活+输出维度=1 / 分割→逐像素分类 / 生成→输出=输入维度）
3. **训练数据有多少条？**（决定网络的深度和宽度上限）
4. **输入张量的形状？**（决定第一层的 in_features / in_channels）
5. **信息管道怎么走？**（画出来：每一步 shape 怎么变）
6. **哪里可能梯度断流？**（深层无残差？Sigmoid 饱和？RNN 长序列？→ 加残差/LSTM/BN）
7. **哪里可能过拟合？**（数据少网络大？→ 加 Dropout / 减小网络 / 数据增强）

做完这 7 步，你的网络设计就不再是"凭感觉填参数"，而是有逻辑、可以解释、可以 debug 的工程设计。

### 1.10 完整模型训练套路

#### 1.10.0 卷积层设计公式

给定输入高度 $H_{\text{in}}$、宽度 $W_{\text{in}}$，以及卷积层参数：

- **卷积核大小 $K$**（假设正方形，即 $K \times K$，若矩形则分别指定 $K_h, K_w$）
- **步长 $S$**
- **填充 $P$**（每边填充的行/列数）
- **膨胀率 $D$**（dilation，默认1）

则输出尺寸为：

$$
H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2P - D \times (K - 1) - 1}{S} + 1 \right\rfloor

$$

$$
W_{\text{out}} = \left\lfloor \frac{W_{\text{in}} + 2P - D \times (K - 1) - 1}{S} + 1 \right\rfloor

$$

- **简化版**（当 $D=1$，即无膨胀时）：

$$
H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2P - K}{S} + 1 \right\rfloor

$$

- **向下取整 $\lfloor \cdot \rfloor$** 表示如果除不尽，多余部分被丢弃（PyTorch默认行为）。

#### 1.10.1测试方式

**```argmax(dim)```**将dim维度中的所有值取出，然后返回最大值的下标。

```python

import torch
#初始化output
output = torch.tensor([[0.6, 0.3, 0.1], 
                       [0.1, 0.1, 0.8]])
#dim表示维度，将dim维度中的所有值，返回最大值的下标。
pre = output.argmax(dim = 1)
label = torch.tensor([0, 0])
print(pre)
print(pre == label)
print((pre == label).sum())
print((pre == label).sum().item())
#输出如下：
# tensor([0, 2])
# tensor([ True, False])
# tensor(1)
# 1
```

## 2.线性回归基础

### 2.1线性回归数学原理

线性回归的标签y和特征值x的关系：

$$
y = w\_1 x\_1 + w\_2 x\_2 + \cdots + w\_n x\_n + b

$$

而线性回归模型所要做的就是**找到最佳的权重w和偏置b**

$$
X = \begin{bmatrix}1 & x_{11} & x_{12} & \cdots & x_{1d} \\
				   1 & x_{21} & x_{22} & \cdots & x_{2d} \\
				   \vdots &\vdots  &\vdots  &\ddots &\vdots \\
				   1 & x_{n1} & x_{n2} & \cdots & x_{nd}
	\end{bmatrix}

\\
\\
\theta = \begin{bmatrix}
		b\\w_1\\w_2\\\vdots\\w_d
		 \end{bmatrix}
\\
\\
X\theta = Y_{pre} = \begin{bmatrix}
		y_1\\y_2\\\vdots\\y_d
		 \end{bmatrix}
\\
\\
损失函数：
loss(\theta) = \frac{1}{2} \begin{Vmatrix} X\theta - Y_{true} \end{Vmatrix}^2 \\= \frac{1}{2} \begin{Vmatrix} Y_{pre} - Y_{true} \end{Vmatrix}^2\\ = \frac{1}{2} (X\theta - Y_{true})^T (X\theta - Y_{true})
\\
= \frac{1}{2}[(X\theta)^T - Y_{true}^T](X\theta - Y_{true})\\ = \frac{1}{2}[(X\theta)^TX\theta - (X\theta)^T Y_{true} - Y_{true}^TX\theta + Y_{true}^TY_{true}]
\\=\frac{1}{2}[\theta^TX^TX\theta - 2Y_{true}^TX\theta + Y_{true}^TY_{true}]\\其中(X\theta)^T Y_{true} 和 Y_{true}^TX\theta都是标量，所以他们的转置等于自己。\\
根据矩阵求导法则:\frac{\partial (X^TAX)}{\partial X} = (A+A^T)X\\\frac{\partial(a^TX)}{\partial X} = a
\\
所以:\frac{\partial loss}{\partial\theta} = \nabla_{\theta} loss(\theta) = \frac{1}{2}[2X^TX\theta - 2X^TY_{true}] = X^TX\theta - X^TY_{true}
\\
为了求出loss的最小值，我们就必须找到梯度为0的w和b，也就是\theta
\\
令 \nabla_{\theta} loss(\theta) = 0 = X^TX\theta - X^TY_{true} => \theta = (X^TX)^{-1}X^TY_{true}
\\
所以最优解\theta = (X^TX)^{-1}X^TY
\\只有线性回归才有通解，并且只有当X^TX可逆时，才能直接求出\theta

$$

### 2.2 学习率对loss的影响

梯度下降需要消耗大量的算力，所以学习率太大和太小都会浪费算力。

![图片](img/学习率对loss的影响直观表现.png)

### 2.3手写线性回归

**Python语法扩展（yield）**

- `return`在函数结束时返回一个结果。
- `yield` 函数运行到`yield`时，返回一个值，然后函数挂起，知道下一次用`next()`调用或者`for`迭代。

```python
def count_up_to(n):
    i = 0
    while i < n:
        yield i   # 每调用一次 next()，就返回当前的 i，并暂停
        i += 1

# 使用生成器
gen = count_up_to(3)
print(next(gen))  # 输出 0
print(next(gen))  # 输出 1
print(next(gen))  # 输出 2
# print(next(gen))  # 触发 StopIteration

# 更常见的用法：直接 for 循环
for num in count_up_to(3):
    print(num)    # 打印 0 1 2
```

**手写线性回归**

```python
import random
import torch

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#创建随机数据
def create_data(w, b, num_examples):
  
    #生成符合正态分布的x，0是均值，1是方差，（num_examples, len(w)）行列。
    x = torch.normal(0, 1, (num_examples, len(w)))
  
    y = x@w + b
    #加入噪音
    y += torch.normal(0, 0.01, y.shape)

    return x, y.reshape((-1, 1))

#批量抽取数据进行梯度下降
def data_iter(batch_size, feature, label):

    num_example = len(feature)
    #生成0到num_example的索引列表
    index = list(range(num_example))
    #用shuffle函数将index打乱，以达到随机抽样的效果
    random.shuffle(index)
    #i是每次抽样的第一个下标
    for i in range(0, num_example, batch_size):
        batch_index = index[i: min(i + batch_size, num_example)]
        #返回feature和label
        yield feature[batch_index], label[batch_index]

#定义模型
def linear_model(x, w, b):
    return x @ w + b

def squared_loss(y_pre, y_true):
    #用平方误差，为了防止两个y的维度不同，我们进行reshape调整
    return 0.5 * (y_pre - y_true.reshape(y_pre.shape))**2

def sgd(params, lr, batch_size):

    with torch.no_grad():
        for param in params:
            #梯度下降
            param -= lr * param.grad / batch_size
            #计算完一轮之后要将grad清零
            param.grad.zero_()



if __name__ == '__main__':

    #创建数据
    example = 1000
    w_true = torch.tensor([2, -3.4])
    b_true = 4.2

    feature, label = create_data(w_true, b_true, example)
    #数据创建成功
    #=========================================================

    #初始化权重和偏置
    #requires_grad = True表示该参数需要进行梯度下降
    w = torch.normal(0, 0.01, size = (2, 1), requires_grad = True)
    b = torch.zeros(1, requires_grad = True)
    print(w, b)

    #=========================================================
    #开始训练
    lr = 0.03
    num_epochs = 3
    batch_size = 10
    net = linear_model
    loss = squared_loss
    #第一层循环，对全部数据扫一遍，一共扫三遍
    for epoch in range(num_epochs):
        #每次拿出batch_size的x和y
        for x, y in data_iter(batch_size, feature, label):
            #计算小批量损失
            l = loss(net(x, w, b), y)
            #计算得到的l是一个[batch_size, 1]的向量
            #我们需要进行求和才是每个样本预测值和真实值的差距
            #用backward()计算梯度
            l.sum().backward()
            #计算完梯度之后才能访问grad这个属性
            if epoch == 0:
                print("grad->", w.grad)
            sgd([w, b], lr, batch_size)
            if epoch == 0:
                print("w->", w)
        #出来这个for循环之后表示已经扫完一遍数据了
        #表示一下内容不需要计算梯度
        with torch.no_grad():
            train_l = loss(net(feature, w, b), label)
            print(f'epoch: {epoch + 1}, loss: {train_l.sum()/example}')

"""
第一次数据：
epoch: 1, loss: 0.040368854999542236
epoch: 2, loss: 0.00015070709923747927
epoch: 3, loss: 5.0547547289170325e-05
tensor([[ 1.9995],
        [-3.3997]], requires_grad=True) tensor([4.2006], requires_grad=True)
第二次数据：
epoch: 1, loss: 0.043382592499256134
epoch: 2, loss: 0.00017751296400092542
epoch: 3, loss: 5.148643322172575e-05
tensor([[ 2.0008],
        [-3.3987]], requires_grad=True) tensor([4.1997], requires_grad=True)
第三次数据：
epoch: 1, loss: 0.04251382499933243
epoch: 2, loss: 0.00017745311197359115
epoch: 3, loss: 5.071829218650237e-05
tensor([[ 1.9993],
        [-3.3996]], requires_grad=True) tensor([4.1995], requires_grad=True)
"""
```

### 2.4 学习率lr和epoch的关系

该模型对应的数据最合适的lr和epoch应该是lr = 0.3、epoch = 3。如图：

<img src="img/lr=0.3_epochs=3.png" alt="图片" style="zoom:50%;" />

我们对参数lr进行调整，将其调整为lr = 0.0003、epoch = 3。如图：

<img src="img/lr=0.0003_epochs=3.png" alt="图片" style="zoom:50%;" />

> 由于学习率lr过低，epoch也低，导致学习不充分，也就是欠拟合，loss一直降不下来。

我们将参数epoch提升到100。如图：

<img src="img/lr=0.0003_epochs=100.png" alt="图片" style="zoom:50%;" />

> 此时虽然学习率低，但是epoch上来了，也就是扫描了100遍该训练数据，也能强行把loss降下来。

通常工程中的参数很难找到理想的，一般情况下都是这种情况：

<img src="img/lr=0.03_epochs=3.png" alt="图片" style="zoom:50%;" />

> loss有升有降，但loss最终能下降到可接受范围。

学习率过小会导致欠拟合，学习率过大则会造成梯度爆炸：

<img src="img/lr=3_epochs=3.png" alt="图片" style="zoom:50%;" />

> 此时loss已经是nan了，梯度爆炸了，学习率过大，导致权重变化过快。

**总结：**

1. 当学习率过小的时候，loss降不下来，提升学习次数（epochs），有可能可以降下来，但是推荐调整参数。
2. 当学习率过大的时候，会出现梯度爆炸，loss=nan，请立刻调整学习率。

### 2.5 用torch实现线性回归

```python
class model(nn.Module):
    #定义单层神经网络

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #一层线性回归网络层
        self.layout1 = nn.Linear(in_features = 2, out_features = 1)
  
    def forward(self, x):

        x = self.layout1(x)
        #经过线性计算之后直接返回值
        return x
if __name__ == '__main__':
    plt.figure(figsize = (10, 5))
    #初始化样本数量
    example = 1000
    #初始化真实的w和b
    w_true = torch.tensor([[2.], [-3.4]]) # 2*1
    b_true = torch.tensor(1.0)
    #随机生成样本
    x = torch.normal(0, 3, (example, 2))# example*2
    #y_true的值
    y_true =  x @ w_true + b_true + torch.normal(0, 0.01, (example, 1))
    #把feature和label组合成dataset！！！！！
    data = TensorDataset(x, y_true)#Dataset可以把feature和label组合起来，格式类似于DataFrame但是在深度学习中比DataFrame更加方便
    #初始化参数
    lr = 0.003
    epochs = 3
    batch_size = 10
    net = model()
    #输出初始化的w和b
    print(net.layout1.weight, ' ', net.layout1.bias)
    #定义损失函数
    loss = nn.MSELoss()
    #定义优化器，用来优化参数的，梯度下降
    opt = optim.SGD(net.parameters(), lr = lr)
    loss_num = []
    for epoch in range(epochs):
        data_loader = DataLoader(data, batch_size, shuffle = True)
        loss_sum = 0
        for x, y_true in data_loader:
            #用神经网络进行预测
            y_pre = net(x)
            #计算损失
            l = loss(y_pre, y_true)
            # print(f'loss = {l}')
            loss_num.append(l.item())
            loss_sum += l
            #清除之前的梯度
            opt.zero_grad()
            #进行反向传播，计算梯度
            l.backward()
            #更具bachward计算出的梯度更新参数
            opt.step()  
        l = loss(net(x), y_true)
        print(f'epoch: {epoch + 1} ==> loss: {loss_sum / 100}')
    print(net.layout1.weight, ' ', net.layout1.bias)
    sns.lineplot(x = range(300), y = loss_num)
    plt.show()
```

### 2.5 激活函数

> 激活函数为神经网络引入**非线性**。如果没有激活函数，多层线性变换的复合仍然是线性变换——网络再深也等价于一层。激活函数让网络能表达"满足条件就激活，否则不激活"这种复杂模式。


| 激活函数         | 公式                                                                                | 特点                                                                                                                                                                   | 缺点                                                                                                                                    |
| :----------------- | :------------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------- |
| **Sigmoid**      | 
$$
\sigma(x) = \frac{1}{1 + e^{-x}}

$$

                                         | 输出 (0, 1)，平滑可导，适合二分类输出层                                                                                                                                | ① 梯度消失：两端导数趋近 0，深层网络梯度回传困难 ② 输出非零均值（0.5 中心），导致下一层输入全为正，收敛慢 ③ exp 运算昂贵             |
| **Tanh**         | 
$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}

$$

                             | 输出 (-1, 1)，零均值中心，比 Sigmoid 收敛更快                                                                                                                          | ① 仍有梯度消失问题（两端导数为 0）② exp 运算昂贵 ③ 深层网络中梯度比 Sigmoid 大但依然会衰减                                           |
| **ReLU**         | 
$$
f(x) = \max(0, x)

$$

                                                        | ① x>0 时导数为 1，**彻底解决正半轴梯度消失** ② 计算极简（一个比较操作）③ 稀疏激活：负值直接为 0，网络更稀疏、泛化更好                                               | ①**Dead ReLU**：输出恒为 0 的神经元梯度为 0，权重永不更新。常见原因：学习率过大 / 偏置初始化过负 ② 输出非零均值 ③ 负半轴信息完全丢弃 |
| **Leaky ReLU**   | 
$$
f(x) = \max(\alpha x, x)

$$

，α 取 0.01                                     | ① 负半轴保留微小梯度（αx），解决 Dead ReLU ② 保留 ReLU 的正半轴优势                                                                                                 | ① α 是超参数，需手动设定 ② 负半轴梯度恒为 α，不如可学习的灵活 ③ 实际效果不一定优于 ReLU                                            |
| **PReLU**        | 
$$
f(x) = \max(\alpha x, x)

$$

，α **可学习**                                  | α 通过反向传播学习，每个通道可不同，比 Leaky ReLU 更灵活                                                                                                              | ① 增加少量参数 ② 训练不稳定时 α 可能学到奇怪的值                                                                                     |
| **ELU**          | 
$$
f(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}

$$

 | ① 负半轴输出平滑（指数衰减而非线性截断）② 输出均值接近 0，加速收敛                                                                                                   | ① exp 运算比 ReLU 慢 ② α 是超参数                                                                                                    |
| **SiLU / Swish** | 
$$
f(x) = x \cdot \sigma(x) = \frac{x}{1+e^{-x}}

$$

                            | ①**非单调、平滑**：x<0 时不会像 ReLU 硬截断 ② 自门控（self-gated）：用 Sigmoid 控制信息通过量 ③ 深层网络中常优于 ReLU                                               | ① 计算比 ReLU 复杂（含 Sigmoid）② 不是所有任务都优于 ReLU                                                                             |
| **GELU**         | 
$$
f(x) = x \cdot \Phi(x)

$$

，Φ 是标准正态 CDF                                | ① 平滑的"随机正则化"：以概率形式决定激活，而非硬阈值 ②**Transformer 默认激活函数**（BERT, GPT 等）③ 近似公式：
$$
0.5x(1+\tanh(\sqrt{2/\pi}(x+0.044715x^3)))

$$

 | ① 计算比 ReLU 复杂得多 ② 推理时可优化但训练时开销大                                                                                   |
| **Softmax**      | 
$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}

$$

             | ① 输出 (0, 1) 且**所有输出之和为 1**，可解释为概率分布 ② 指数运算放大差异，利于分类决策 ③ 与交叉熵损失天然配合                                                      | ① 仅用于输出层（中间层不用）② 对异常值敏感（指数放大）③ 配合交叉熵时，梯度 = y_pred - y_true，极易计算                               |

**核心对比速记：**


| 问题                                | 解决方案                                 |
| :------------------------------------ | :----------------------------------------- |
| 梯度消失（Sigmoid/Tanh 两端饱和）   | ReLU（正半轴导数=1）                     |
| Dead ReLU（负值被永久杀死）         | Leaky ReLU / PReLU / ELU                 |
| 输出非零均值（偏移下一层输入）      | Tanh（零中心）/ ELU / BN 层              |
| 需要平滑激活（概率解释 + 深层表现） | GELU（Transformer）/ SiLU（ConvNeXt 等） |
| 多分类输出                          | Softmax                                  |

## 3.Softmax回归

### 3.0 Softmax激活函数公式

> 将输出值进行指数运算可以**扩大不同输出之间的差异**，有利于分类。

$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} \quad \text{for } i = 1, 2, \ldots, K

$$

$$
\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}

$$

$$
\text{Softmax}(z_i) = \frac{\exp(z_i / T)}{\sum_{j=1}^{K} \exp(z_j / T)}

$$

### 3.1Softmax数学原理

二分类交叉熵公式：

$$
\text{BCE}(y, \hat{y}) = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]

$$

多分类交叉熵公式：

$$
\text{CE}(y, \hat{y}) = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)

$$

- $ y_i  \in \begin{Bmatrix} 0, 1 \end{Bmatrix}$：one-hut真是标签，只有第i维为1，其余都为0，表示该样本分类为i类。
- $\hat{y}$：表示模型预测样本为第i类的概率大小，会被归一化。所有的$\hat{y}$相加等于1。
- 公式：只有 $y_i$ 这一项不为零，所以公式的数值等于 $-y_i \log(\hat{y}_i)$ ，此时 $\hat{y}$ 越大，损失值越小，所以模型会尽可能地让 $\hat{y}$ 变大，从而提升了区分度。

<img src="img/Softmax数学原理.png" alt="图片" style="zoom: 33%;" />

**容易误解：** 交叉熵损失通常不能用于梯度下降，因为它不具有参数，但是它可以指导前面的全连接层（隐藏层）进行梯度下降。

### 3.2 损失函数

3个常用损失函数：

1. $l(y, y') = \frac{1}{2}(y - y')^2$：均方损失，当权重距离真实权重很远的时候，梯度会比较大，可以加速学习，但同时也不稳定。
2. $l(y, y') = |y - y'|$：绝对损失，不管权重距离真实权重有多远，梯度都是一个常数，学习速度可能没那么快，但是很稳定。当权重距离真实权重很近的时候，因为原点不可导，容易导致稳定性变差。
3. $l(y, y') = \begin{cases}
   |y - y'| - \frac{1}{2} & \text{if } |y - y'| > 1 \\
   \frac{1}{2}(y - y')^2 & \text{otherwise}
   \end{cases}$：Robust Loss，结合了两者的优点，是梯度下降整体变稳定。

<img src="img/RobustLoss.png" alt="图片" style="zoom:33%;" />

**RobustLoss比较常用**

### 3.3 前向传播和反向传播

前向传播 (Forward Propagation)
第一层：

* **线性变换与激活**：

  $$
  z_1 = w_1 x + b_1 = 3 \times 2 + 1 = 7

  $$

  $$
  a_1 = f(z_1) = 7 \times 2 + 1 = 15

  $$

第二层（假设全连接层）：

* **线性变换**：
  $$
  z_2 = W_2 a_1 + b_2 = 3 \times 15 + 1 = 46

  $$
* **激活**：
  $$
  a_2 = f(z_2) = 3 \times 46 + 1 = 139

  $$

**损失计算**：

$$
L = (y - a_2)^2 = 2435

$$

---

反向传播 (Backward Propagation)

方向流：`首先输出层 -> dL/d... -> 更新参数`

输出层到损失函数的梯度推导
设损失为均方误差，则有：

$$
\hat{y} = a_2

$$

$$
\frac{dL}{da_2} = 2(y - a_2)

$$

$$
\text{代入数值： } 2 \times (80 - 139) = -118

$$

全连接层链式法则 (Chain Rule)
笔记右下角的推导逻辑展示了链式法则在各层的应用：

**输出层函数导数（需根据激活函数选择）：**

* **如果是 Sigmoid 函数**：
  $$
  \frac{da_2}{dz_2} = a_2(1-a_2)

  $$
* **如果是线性函数（如 \(y=a_2\)）**：
  $$
  \frac{da_2}{dz_2} = 1 \quad (\text{即当 } y=a_2 \text{ 时})

  $$

第二层权重梯度计算
根据链式法则：

$$
\frac{dL}{dw_2} = \frac{dL}{dz_2} \cdot a_1

$$

图中具体的数值逻辑（假设 \(\frac{dL}{dz_2} = 26\)）：

$$
grad = \frac{dL}{dw_2} = \frac{dL}{dz_2} \cdot \frac{dz_2}{dw_2} = \frac{dL}{dz_2} \cdot a_1

$$

$$
\text{代入数值： } 26 \times 15 = 390

$$

 第一层权重梯度计算
继续利用链式法则向上一层传播：

$$
\frac{dL}{dw_1} = \frac{dL}{dz_1} \cdot x

$$

图中具体的数值逻辑（假设 \(\frac{dL}{dz_1} = 52\)）：

$$
\text{代入数值： } 52 \times 2 = 104

$$

---

 参数更新
图中底部注明了计算出的梯度用于**更新参数**，完成向后传播 → 向前传播的闭环迭代：

* \( \frac{dL}{dw_1} \rightarrow \text$4{用于更新参数} \)
* \( \frac{dL}{dw_2} \rightarrow \text{用于更新参数} \)

**易错提醒：**

> 每一个神经元都有一组参数（权重），如果某一层的输入3且输出2，那么这一层的W矩阵的形状是3×2。
> 通常都是把反向传播完全算完才进行梯度更新，不是边反向传播边更新梯度。

**具体例子：**

**三层神经网络的前向与反向传播（手算版）**

> 费曼说：别背公式，动手拧旋钮。下面我们用一个具体的三层网络（无激活函数、无偏置），手算前向传播和反向传播的每一步。所有数字都是小整数，方便你拿笔跟着算。

**网络结构**

- **输入层**：2个特征
  $x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$
- **第一隐藏层**：3个神经元 → 权重矩阵 $W^{(1)}$ 大小 **3×2**
- **第二隐藏层**：2个神经元 → 权重矩阵 $W^{(2)}$ 大小 **2×3**
- **输出层**：1个神经元 → 权重矩阵 $W^{(3)}$ 大小 **1×2**（一个行向量）

**激活函数**：无（线性传播，先学会走路）
**损失函数**：平方误差的一半
$L = \frac{1}{2}(y_{pred} - y_{true})^2$

**具体数字（全是小整数）**

- 输入：
  $x = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$
- 第一层权重 $W^{(1)}$（3行2列）：
  $W^{(1)} = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \\ 0.5 & 0.6 \end{bmatrix}$
- 第二层权重 $W^{(2)}$（2行3列）：
  $W^{(2)} = \begin{bmatrix} 0.7 & 0.8 & 0.9 \\ 1.0 & 1.1 & 1.2 \end{bmatrix}$
- 输出层权重 $W^{(3)}$（1行2列）：
  $W^{(3)} = \begin{bmatrix} 1.3 & 1.4 \end{bmatrix}$
- 真实值（标量）：
  $y_{true} = 5$
- 学习率：
  $\eta = 0.1$

---

**第一步：前向传播（算出当前预测）**

**第一层输出 $h^{(1)} = W^{(1)} x$**

$ \begin{aligned} h^{(1)}_1 &= 0.1\times1 + 0.2\times2 = 0.1 + 0.4 = 0.5 \\ h^{(1)}_2 &= 0.3\times1 + 0.4\times2 = 0.3 + 0.8 = 1.1 \\ h^{(1)}_3 &= 0.5\times1 + 0.6\times2 = 0.5 + 1.2 = 1.7 \end{aligned} $

$h^{(1)} = \begin{bmatrix} 0.5 \\ 1.1 \\ 1.7 \end{bmatrix}$

**第二层输出 $h^{(2)} = W^{(2)} h^{(1)}$**

$ \begin{aligned} h^{(2)}_1 &= 0.7\times0.5 + 0.8\times1.1 + 0.9\times1.7 \\ &= 0.35 + 0.88 + 1.53 = 2.76 \\ h^{(2)}_2 &= 1.0\times0.5 + 1.1\times1.1 + 1.2\times1.7 \\ &= 0.5 + 1.21 + 2.04 = 3.75 \end{aligned} $

$h^{(2)} = \begin{bmatrix} 2.76 \\ 3.75 \end{bmatrix}$

**输出层 $y_{pred} = W^{(3)} h^{(2)}$**

$y_{pred} = 1.3\times2.76 + 1.4\times3.75 = 3.588 + 5.25 = 8.838$

**损失**

$L = \frac{1}{2}(y_{pred} - y_{true})^2 = \frac{1}{2}(8.838 - 5)^2 = \frac{1}{2}(3.838)^2 = \frac{1}{2}\times14.730244 = 7.365122$

预测 8.838，真实 5，误差很大。现在反向传播，看看每个旋钮要拧多少。

---

**第二步：反向传播（从输出往输入算梯度）**

**输出层 delta**

$\delta^{(3)} = \frac{\partial L}{\partial y_{pred}} = y_{pred} - y_{true} = 8.838 - 5 = 3.838$
（这是一个标量）

**对 $W^{(3)}$ 的梯度**

$\frac{\partial L}{\partial W^{(3)}} = \delta^{(3)} \cdot (h^{(2)})^T = 3.838 \times \begin{bmatrix} 2.76 & 3.75 \end{bmatrix}$

$3.838\times2.76 \approx 10.593 \quad,\quad 3.838\times3.75 = 14.3925$

$\nabla_{W^{(3)}} L \approx \begin{bmatrix} 10.593 & 14.3925 \end{bmatrix}$

---

**第二隐藏层的 delta（传播到 $h^{(2)}$）**

$\delta^{(2)} = \left( \frac{\partial L}{\partial h^{(2)}} \right)^T = (W^{(3)})^T \cdot \delta^{(3)}$

因为 $y_{pred} = W^{(3)} h^{(2)}$，所以 $\frac{\partial y_{pred}}{\partial h^{(2)}} = (W^{(3)})^T$。

$W^{(3)} = \begin{bmatrix} 1.3 & 1.4 \end{bmatrix} \quad\Rightarrow\quad (W^{(3)})^T = \begin{bmatrix} 1.3 \\ 1.4 \end{bmatrix}$

$\delta^{(2)} = \begin{bmatrix} 1.3 \\ 1.4 \end{bmatrix} \times 3.838 = \begin{bmatrix} 1.3\times3.838 \\ 1.4\times3.838 \end{bmatrix} = \begin{bmatrix} 4.9894 \\ 5.3732 \end{bmatrix}$

**对 $W^{(2)}$ 的梯度**

$\frac{\partial L}{\partial W^{(2)}} = \delta^{(2)} \cdot (h^{(1)})^T$

$\delta^{(2)}$ 是 $2\times1$，$h^{(1)}$ 是 $3\times1$，外积得 $2\times3$ 矩阵：

$\nabla_{W^{(2)}} L = \begin{bmatrix} 4.9894 \\ 5.3732 \end{bmatrix} \begin{bmatrix} 0.5 & 1.1 & 1.7 \end{bmatrix} = \begin{bmatrix} 4.9894\times0.5 & 4.9894\times1.1 & 4.9894\times1.7 \\ 5.3732\times0.5 & 5.3732\times1.1 & 5.3732\times1.7 \end{bmatrix}$

计算近似值：

- 第一行：$2.4947,\; 5.48834,\; 8.48198$
- 第二行：$2.6866,\; 5.91052,\; 9.13444$

---

**第一隐藏层的 delta（传播到 $h^{(1)}$）**

$\delta^{(1)} = (W^{(2)})^T \cdot \delta^{(2)}$

$W^{(2)}$ 是 $2\times3$，转置是 $3\times2$：

$(W^{(2)})^T = \begin{bmatrix} 0.7 & 1.0 \\ 0.8 & 1.1 \\ 0.9 & 1.2 \end{bmatrix}$

$\delta^{(1)} = \begin{bmatrix} 0.7 & 1.0 \\ 0.8 & 1.1 \\ 0.9 & 1.2 \end{bmatrix} \begin{bmatrix} 4.9894 \\ 5.3732 \end{bmatrix} = \begin{bmatrix} 0.7\times4.9894 + 1.0\times5.3732 \\ 0.8\times4.9894 + 1.1\times5.3732 \\ 0.9\times4.9894 + 1.2\times5.3732 \end{bmatrix}$

计算：

- 第一项：$3.49258 + 5.3732 = 8.86578$
- 第二项：$3.99152 + 5.91052 = 9.90204$
- 第三项：$4.49046 + 6.44784 = 10.9383$

$\delta^{(1)} \approx \begin{bmatrix} 8.86578 \\ 9.90204 \\ 10.9383 \end{bmatrix}$

**对 $W^{(1)}$ 的梯度**

$\frac{\partial L}{\partial W^{(1)}} = \delta^{(1)} \cdot x^T$

$x = [1; 2]$，$x^T = [1, 2]$。$\delta^{(1)}$ 是 $3\times1$，结果 $3\times2$：

$\nabla_{W^{(1)}} L = \begin{bmatrix} 8.86578 \\ 9.90204 \\ 10.9383 \end{bmatrix} \begin{bmatrix} 1 & 2 \end{bmatrix} = \begin{bmatrix} 8.86578\times1 & 8.86578\times2 \\ 9.90204\times1 & 9.90204\times2 \\ 10.9383\times1 & 10.9383\times2 \end{bmatrix} = \begin{bmatrix} 8.86578 & 17.73156 \\ 9.90204 & 19.80408 \\ 10.9383 & 21.8766 \end{bmatrix}$

---

**第三步：更新权重（往梯度反方向拧）**

学习率 $\eta = 0.1$，新权重 = 旧权重 $-\; \eta \times \nabla L$

**更新 $W^{(3)}$**

$W^{(3)}_{\text{new}} = \begin{bmatrix} 1.3 & 1.4 \end{bmatrix} - 0.1\times\begin{bmatrix} 10.593 & 14.3925 \end{bmatrix} = \begin{bmatrix} 1.3 - 1.0593 & 1.4 - 1.43925 \end{bmatrix} = \begin{bmatrix} 0.2407 & -0.03925 \end{bmatrix}$

**更新 $W^{(2)}$**

$W^{(2)}_{\text{new}} = \begin{bmatrix} 0.7 & 0.8 & 0.9 \\ 1.0 & 1.1 & 1.2 \end{bmatrix} - 0.1\times\begin{bmatrix} 2.4947 & 5.48834 & 8.48198 \\ 2.6866 & 5.91052 & 9.13444 \end{bmatrix}$

$= \begin{bmatrix} 0.7-0.24947 & 0.8-0.548834 & 0.9-0.848198 \\ 1.0-0.26866 & 1.1-0.591052 & 1.2-0.913444 \end{bmatrix} = \begin{bmatrix} 0.45053 & 0.251166 & 0.051802 \\ 0.73134 & 0.508948 & 0.286556 \end{bmatrix}$

**更新 $W^{(1)}$**

$W^{(1)}_{\text{new}} = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \\ 0.5 & 0.6 \end{bmatrix} - 0.1\times\begin{bmatrix} 8.86578 & 17.73156 \\ 9.90204 & 19.80408 \\ 10.9383 & 21.8766 \end{bmatrix}$

$= \begin{bmatrix} 0.1-0.886578 & 0.2-1.773156 \\ 0.3-0.990204 & 0.4-1.980408 \\ 0.5-1.09383 & 0.6-2.18766 \end{bmatrix} = \begin{bmatrix} -0.786578 & -1.573156 \\ -0.690204 & -1.580408 \\ -0.59383 & -1.58766 \end{bmatrix}$

---

**第四步：验证（一次前向传播看损失是否下降）**

用新权重重新计算一遍（只算到输出）：

**第一层** $h^{(1)}_{\text{new}} = W^{(1)}_{\text{new}} x$：

$ \begin{aligned} h^{(1)}_{\text{new},1} &= (-0.786578)\times1 + (-1.573156)\times2 = -0.786578 - 3.146312 = -3.93289 \\ h^{(1)}_{\text{new},2} &= (-0.690204)\times1 + (-1.580408)\times2 = -0.690204 - 3.160816 = -3.85102 \\ h^{(1)}_{\text{new},3} &= (-0.59383)\times1 + (-1.58766)\times2 = -0.59383 - 3.17532 = -3.76915 \end{aligned} $

**第二层** $h^{(2)}_{\text{new}} = W^{(2)}_{\text{new}} h^{(1)}_{\text{new}}$（取近似值）：

$ \begin{aligned} h^{(2)}_{\text{new},1} &= 0.45053\times(-3.933) + 0.251166\times(-3.851) + 0.051802\times(-3.769) \\ &\approx -1.771 + (-0.967) + (-0.195) = -2.933 \\ h^{(2)}_{\text{new},2} &= 0.73134\times(-3.933) + 0.508948\times(-3.851) + 0.286556\times(-3.769) \\ &\approx -2.876 + (-1.960) + (-1.080) = -5.916 \end{aligned} $

**输出层** $y_{pred,\text{new}} = W^{(3)}_{\text{new}} h^{(2)}_{\text{new}}$：

$y_{pred,\text{new}} = 0.2407\times(-2.933) + (-0.03925)\times(-5.916) \approx -0.706 + 0.232 = -0.474$

**新损失**：

$L_{\text{new}} = \frac{1}{2}(-0.474 - 5)^2 = \frac{1}{2}(-5.474)^2 = \frac{1}{2}\times29.96 \approx 14.98$

> 😲 损失反而从 7.37 升到了 14.98！
> **费曼点评**：我们步子迈得太大了——学习率 0.1 对于这么大的梯度（数量级 10~20）来说太大，导致权重更新过猛，甚至改变了符号。算法本身没错，只是超参数没选好。把学习率改成 0.01 再试一次，你就会看到损失稳步下降。这就是为什么深度学习训练要小心调整学习率。

---

**核心规则（费曼三句话记住反向传播）**

1. **前向**：输出 = 权重 × 输入（矩阵乘法）
2. **反向**：
   - 当前层的 delta = 后一层的权重矩阵的转置 × 后一层的 delta
   - 当前层权重的梯度 = 当前层的 delta × 前一层的输出的转置
3. **更新**：新权重 = 旧权重 − 学习率 × 梯度

只要记住这三句，就算有一百层也能算。

---

> 现在，你可以自己取一组新数字，比如把学习率改成 0.01，重新执行第三步和第四步，你会看到损失真的变小了。数学永远不会骗人，只是手算太累而已。

## 4.正则化方法

1. 损失函数添加正则项（范数惩罚），通常使用L1和L2正则项。

- $\min \ell(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2$
- $\min \ell(\mathbf{w}, b) + \lambda \|\mathbf{w}\|_1$

2. Dropout正则化。

- 放弃全连接层的某些神经元，以达到正则化的效果。（放弃掉的神经元梯度为0，权重无法更新，和范数惩罚效果差不多）
- 没被放弃的神经元要进行相应的扩大，为了使整体的期望不变。

$$
x'_i = \begin{cases} 
0 & \text{with probability } p \\
\frac{x_i}{1-p} & \text{otherwise}
\end{cases}

$$

> 被放弃的直接变成0，没被放弃的，除1-p。整体数值权重不变。

3. BN层。
   **详情在8.9章节**。

## 5.数据稳定性

### 5.0 梯度下降直观感受

<img src="img/梯度下降.png" alt="图片" style="zoom:33%;" />

### 5.1引发梯度爆炸和梯度消失的原因

$$
\frac{\partial \ell}{\partial \mathbf{W}^t} = \frac{\partial \ell}{\partial \mathbf{h}^d} \frac{\partial \mathbf{h}^d}{\partial \mathbf{h}^{d-1}} \cdots \frac{\partial \mathbf{h}^{t+1}}{\partial \mathbf{h}^t} \frac{\partial \mathbf{h}^t}{\partial \mathbf{W}^t}

$$

> 反向传播时，要用链式法则，也就是后面的权重的梯度会不断地连乘前面的权重，如果前面的权重大部分大于1的话，由于网络层数可能很深，当这么多层的权重乘在一起容易使后面的梯度变得异常大甚至无法用数据容器装载（溢出）。如果前面的权重都小于1并且比较小，则容易导致梯度消失，也就是梯度很小，几乎起不到学习的效果。

例如：

我们设有一个三层的网络结构：

- 第一层：$h_1 = w_1x$
- 第二层：$h_2 = w_2x$
- 第三层：$\hat{y} = w_3 \cdot h_2$

**前向传播过程省略**

反向传播：

链式法则：

$$
\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h_2} \cdot \frac{\partial h_2}{\partial h_1} \cdot \frac{\partial h_1}{\partial w_1}

$$

- $\frac{\partial L}{\partial \hat{y}} = \hat{y} - y = 24$
- $\frac{\partial \hat{y}}{\partial h_2} = w_3 = 4$
- $\frac{\partial h_2}{\partial h_1} = w_2 = 3$
- $\frac{\partial h_1}{\partial w_1} = x = 1$

这些都是链式法则要连乘的内容，这些是后面这些层的权重，而且权重都>1。

**第一层的梯度会变成：**

$$
\frac{\partial L}{\partial w_1} = 24 \times 4 \times 3 \times 1 = 288

$$

> 变成288了，这里只是3层，如果是100层的话，梯度很容易爆炸，当然，如果权重都特别小，也会造成梯度消失。

### 5.2激活函数对反向传播的影响


| **激活函数**              | **对梯度影响**                                                       | **主要原因**                                                                                   |
| --------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| RuLU                      | 如果权重初始化不到位，容易引发**梯度爆炸**，但是缓解了**梯度消失**。 | 正半轴导数为1，梯度完全由权重乘积决定；权重>1时连乘导致爆炸。                                  |
| Leaky ReLU / PReLU        | 比ReLU更稳定，轻微缓解爆炸风险。                                     | 负半轴有微小斜率（如0.01），避免神经元死亡，但正半轴仍为1，爆炸风险仍存                        |
| Sigmoid                   | 容易造成**梯度消失**，几乎不会**梯度爆炸**。                         | 导数最大值仅0.25，且大部分区域导数接近0；连乘后梯度指数级衰减。                                |
| Tanh                      | 容易**梯度消失**，极少爆炸。                                         | 导数最大值1（在0处），但两端饱和趋近0；连乘后梯度仍会消失（除非权重非常大且输入一直在0附近）。 |
| Softmax（通常用于输出层） | 本身不直接引起消失/爆炸，但配合交叉熵时梯度稳定                      | 梯度形式为$p_i - y_i$，范围在[-1,1]，不受深层连乘影响。                                        |

**知识扩展：激活函数和参数初始化函数的搭配**


| 函数                               | 适用场景             |
| ------------------------------------ | ---------------------- |
| xavier_uniform_ / xavier_normal_   | 配合 sigmoid/tanh    |
| kaiming_uniform_ / kaiming_normal_ | 配合 ReLU            |
| normal_                            | 简单的正态分布初始化 |

### 5.3梯度消失和梯度爆炸所带来的问题

**梯度消失带来的问题：**

- 梯度值会变成0，因为机器存储数据的精度有限，所以梯度太小了会变成0

> 对16位浮点数尤其严重

- 梯度太小对训练没有进展
- 梯度太小仅仅对距离输出层进的全连接层训练有效果，因为越往前，梯度变得越小。

**梯度爆炸带来的问题：**

- 值会超出值域

> 对于16位浮点数尤其严重

- 对学习率异常敏感

> 如果学习率太大->大参数值->更大的梯度
> 如果学习率太小->学习没有进展

### 5.4**如何让训练变得更加稳定**

1. 将乘法变成加法，避免连乘

> ResNet和LSTM

2. 归一化

> 梯度归一化，梯度裁剪

3. 选择合理的激活函数和合理的参数初始化

**补充知识ont-hot编码：**

> 补充知识：one-hut热编码，用于将数据中的离散值变成bool值。例如：
> 原始列：颜色 = 红色、蓝色、绿色
> one-hot之后：
> 颜色_红	颜色_绿	颜色_蓝
> 0 		        1		    0           --------->表示绿色
> 1	             0		    0	       --------->表示红色
> 0	             0            1           --------->表示蓝色

> 要将DataFrame转化为Tensor，不能直接转化，要先把DataFrame转化为Numpy数组，因为DataFrame有行id和列名等Tensor所没有的东西，Tensor只表示数组，没有列名和行号之类的，而Numpy数组就是只有数值。用DataFrame.values可以转化为Numpy数组。

增加维度之后对算力要求有没有提升？


| 情况                | 影响         | 建议                                                    |
| --------------------- | -------------- | --------------------------------------------------------- |
| 离散值很小（<50）   | 基本可以忽略 | 直接用one-hot                                           |
| 离散值多（50~几百） | 有点吃力     | 可以考虑用embedding，把一个高维向量压缩成稠密的低维向量 |
| 离散值极多（>几千） | 显著拖慢     | 必须用embedding或做特征工程                             |

```python
train_data = pd.get_dummies(train_data, dummy_na = True)
test_data = pd.get_dummies(test_data, dummy_na = True)
#dummy_na = True表示将空值区分开
#因为 dummy_na=True 给每个原始列都生成了一个 _nan 列，如果原始数据中该特征没有缺失值，那个 _nan 列就全是 False，转成 float 后自然全是 0.0。
```

**补充Torch小知识：**

```python
# 将特征和标签按样本对齐，打包在一起
dataset = TensorDataset(X, y)
# 你可以在此设定批大小(batch_size)、是否打乱(shuffle)等
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

#激活函数的调用时从torch中调用，而不是torch.nn中调用。
```

> TensorDataset(X, y)将tensorX和y组合起来，组合成类似于DataFrame但是又比DataFrame更适合深度学习的数据结构，其中X是特征，y是标签。
> DataLoader用来加载数据，按批次送入网络中学习。

## 6.Kaggle房价预测实例

### 6.1补充StandardScaler归一化知识

**StandardScaler归一化：**

> StandardScaler中，是将每一列的数据进行归一化，写成正态分布的形式，每一个值对这一列的平均值差了几个标准差。

**造成数据泄露的原因：**

> 如果错误的将测试数据进行fit_transform之后，这个transform对象就会存储测试集的均值和方差，如果再对训练数据进行transform的话，训练数据对应的列会使用测试数据对应的列的均值和方差进行归一化，也就是说训练数据中可以间接的反映出测试数据的均值和方差，这会造成数据泄露。

**容易误解的地方：**

> 如果把测试集的标签和特征拆开也会造成数据泄露，因为即使没有标签，测试集的特征也已经泄露到训练集中，会导致模型评分虚高。

**标签是否要进行StandardScaler归一化：**

需要使用归一化处理：

1. 神经网络中依赖梯度下降来拟合曲线，其中要用到反向传播和链式法则，如果y_true特别大（房价预测等），那么对应的y_pre与y_true的值相差**可能**就会特别大，这个差值过大的话，就容易导致梯度爆炸
2. 标签的跨度极大，也需要用归一化处理。

不需要使用归一化处理：

1. 树模型，树在分裂节点只关系**特征或标签的相对大小和排序，完全不关心绝对数值**。
2. 分类任务，分类任务的标签是离散型，没必要归一化。

**知识扩展：**

```python
net.eval()
#让网络进入评估模式，此时不进行参数更新
net.train()
#让网络进行训练模式，此时进行参数更新
```

### 6.2 源代码

```python
import torch
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split

# 模型定义
# =============================================================================

class Model(nn.Module):

    def __init__(self, in_features):
        #调用父类的初始化函数
        super().__init__()
        #第一层线性层
        self.layout1 = nn.Linear(in_features, 200)
        #第一层参数初始化
        nn.init.xavier_normal_(self.layout1.weight)
        nn.init.zeros_(self.layout1.bias)
        #第二层线性层
        self.layout2 = nn.Linear(200, 100)
        #第二层参数初始化
        nn.init.xavier_normal_(self.layout2.weight)
        nn.init.zeros_(self.layout2.bias)
        #第三层线性层
        self.layout3 = nn.Linear(100, 1)
        #第三层参数初始化
        nn.init.xavier_normal_(self.layout3.weight)
        nn.init.zeros_(self.layout3.bias)
        #Dropout防止过拟合
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        #前向传播，用tanh激活函数，并且每一层都是用dropout
        x = torch.tanh(self.layout1(x))
        x = self.dropout(x)
        x = torch.tanh(self.layout2(x))
        x = self.dropout(x)
        x = self.layout3(x)
        return x


# 数据处理
# =============================================================================

#读取表格数据DataFrame
train_data = pd.read_csv(r'data\train.csv')
test_data = pd.read_csv(r'data\test.csv')

# 提取特征和标签
train_label = train_data.iloc[:, -1].values.reshape(-1, 1)
train_feature = train_data.iloc[:, 1:-2]
test_feature = test_data.iloc[:, 1:-1]

# one-hot 编码（train + test 合并，保证列一致）
#concat将两个DataFrame合并成一个
all_feature = pd.concat([train_feature, test_feature], axis=0)
#进行one-hot编码
all_feature = pd.get_dummies(all_feature, dummy_na=True)

#编码完之后再拆开
n_train = train_feature.shape[0]
train_feature = all_feature.iloc[:n_train, :]
test_feature = all_feature.iloc[n_train:, :]

# ===== 先切分，再 fit scaler（防止数据泄露）=====
X_train, X_val, y_train, y_val = train_test_split(
    train_feature, train_label,
    test_size=0.2, random_state=42
)

# 特征标准化：只在训练集上 fit
feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(X_train)
X_val   = feature_scaler.transform(X_val)
X_test  = feature_scaler.transform(test_feature)

# label：log1p → 标准化（只在训练集上 fit）
y_train = np.log1p(y_train)
y_val   = np.log1p(y_val)

label_scaler = StandardScaler()
y_train = label_scaler.fit_transform(y_train)
y_val   = label_scaler.transform(y_val)

# 清理 NaN（零方差 one-hot 列导致）
X_train = np.nan_to_num(X_train)
X_val   = np.nan_to_num(X_val)
X_test  = np.nan_to_num(X_test)

# 转 tensor
#要转成tensor才能送入dataset中转化问dataset数据结构，然后才能送入dataloader中
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val   = torch.tensor(X_val,   dtype=torch.float32)
y_val   = torch.tensor(y_val,   dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)

# 模型训练
# =============================================================================

batch_size = 100
epochs = 500

#初始化模型
net = Model(in_features=X_train.shape[1])

#转化为dataset，送入dataloader，方便读取batch_size条数据
dataset_train = TensorDataset(X_train, y_train)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

#L2损失函数
loss_fn = nn.MSELoss()
#用向量法来稳定参数，缓解梯度爆炸
optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-4)

#开始训练
for epoch in range(epochs):
    # ---- 训练 ----
    #将net网络切换为训练模式
    net.train()
    loss_sum = 0
    for x, y_true in dataloader_train:
        y_pre = net(x)
        l = loss_fn(y_pre, y_true)

        optimizer.zero_grad()
        l.backward()
        #对梯度进行裁剪
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        loss_sum += l.item()

    train_loss = loss_sum / len(dataloader_train)

    # ---- 验证 ----
    #将net网络切换为评估模式
    net.eval()
    with torch.no_grad():
        val_pred = net(X_val)
        val_loss = loss_fn(val_pred, y_val).item()

    print(f"epoch {epoch+1:3d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

# 验证集 RMSLE
# =============================================================================

net.eval()
with torch.no_grad():
    val_pred = net(X_val).numpy()

# 逆变换：标准化空间 → log 空间
y_pred_log = label_scaler.inverse_transform(val_pred)
y_true_log = label_scaler.inverse_transform(y_val.numpy())

# RMSLE = log 空间里的 RMSE
rmsle = np.sqrt(np.mean((y_pred_log - y_true_log) ** 2))
print(f"\n验证集 RMSLE: {rmsle:.5f}")

# 生成提交文件
# =============================================================================

net.eval()
with torch.no_grad():
    test_pred = net(X_test).numpy()

# 逆变换：标准化空间 → log 空间 → 原始价格
test_pred_log = label_scaler.inverse_transform(test_pred)   # log 空间
test_pred_price = np.expm1(test_pred_log)                    # 原始价格

# 组装提交文件（两列：Id, SalePrice）
submission = pd.DataFrame({
    'Id': test_data['Id'],
    'SalePrice': test_pred_price.flatten()
})
submission.to_csv('submission.csv', index=False)

print(f"提交文件已保存: submission.csv ({len(submission)} 条)")
print('end')
```

**RMSLE评估公式：**

$$
\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( \log(p_i + 1) - \log(a_i + 1) \right)^2 }

$$

其中 $p_i$ 是预测价格，$a_i$ 是真实价格，$n$ 是样本数。

### 6.3遇到的问题和解决方式

1. 'int' object is not callable


| 问题                   | 原因                                                                                               | 解决                                         |
| ------------------------ | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| TensorDataset 报这个错 | 传入了 numpy 数组而非 torch.Tensor，numpy 的 .size 是 int 属性，PyTorch 内部把它当 .size(0) 调用了 | torch.tensor(data, dtype=torch.float32) 转换 |

2. 梯度爆炸，loss = NaN


| 问题               | 原因                                                                              | 解决                                                 |
| -------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------ |
| 训练中 loss 变 NaN | SGD + 深层 ReLU 无梯度裁剪，某个 batch 大误差 → 大梯度 → 权重暴涨 → 正反馈爆炸 | ① SGD 换 Adam ② 加 clip_grad_norm_ ③ ReLU 换 tanh |

3. 特征中含有NaN


   | 问题                  | 原因                                                                           | 解决                                 |
   | ----------------------- | -------------------------------------------------------------------------------- | -------------------------------------- |
   | train_feature has NaN | one-hot 列的稀有类别在训练集全为 0，方差 = 0，StandardScaler 做 (x-0)/0 → NaN | np.nan_to_num() 清掉，默认用0代替NaN |
4. 初始化和激活函数不匹配


   | 问题                   | 原因                                                                        | 解决                |
   | ------------------------ | ----------------------------------------------------------------------------- | --------------------- |
   | tanh 用了 Kaiming init | Kaiming 的 gain=√2 是给 ReLU 的，tanh 用这个会导致输入方差偏大、神经元饱和 | 换成 xavier_normal_ |
5. 数据泄露问题


   | 问题                        | 原因                                                 | 解决                                                                    |
   | ----------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------- |
   | StandardScaler 在切分前 fit | scaler 学到了验证集和测试集的均值/方差，验证分数虚高 | 先 train_test_split，再 fit_transform（训练集），transform（验证/测试） |
6. 严重过拟合


   | 问题                             | 原因                                           | 解决                                     |
   | ---------------------------------- | ------------------------------------------------ | ------------------------------------------ |
   | train_loss=0.0002, val_loss=0.19 | 1168 条数据 vs 8 万参数，三层 200→100→1 太大 | 加 Dropout(0.2) + Adam weight_decay=1e-4 |
7. 欠拟合假象


   | 问题                               | 原因                                                                   | 解决                               |
   | ------------------------------------ | ------------------------------------------------------------------------ | ------------------------------------ |
   | train=0.10, val=0.15，以为是欠拟合 | 其实是过拟合已缓解但模型到瓶颈，增加 epoch 只让 train 继续跌、val 不动 | 认清"数据承载力天花板"，不做无用功 |

## 7.在GPU中创建神经网络

**在GPU中创建张量和神经网络**

```python

#在GPU创建张量
#默认创建在CPU
x = torch.tensor([1, 2, 3])
print(x.device)#输出GPU
#在GPU中创建数据
y = torch.tensor([1, 2, 3], device = 'cuda')#表示在GPU创建y
y = torch.tensor([1, 2, 3], device = 'cuda:0')#表示在第一张GPU创建y，通常个人计算机只有1张GPU

#在GPU创建网络
#方法1：
device = torch.device("cuda")
layer = nn.Linear(10, 20, device=device)   # 参数直接在GPU上
#方法2：
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        #在CPU创建的线性层
        self.fc = nn.Linear(10, 20)
        #用to函数挪到GPU
        self.to("cuda")   # 将整个模型（包括刚创建的Linear）移到GPU
```

## 8.卷积神经网络

### 8.1卷积数学原理理解

**视频讲解：**

> https://www.bilibili.com/video/BV1VV411478E?t=1599.0

**数学公式：**

$$
(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) \, d\tau

$$

卷积处理的问题：

> 在一个系统中
>
> 1. 系统的输入不稳定。$f(x)$
> 2. 系统的输出稳定。$g(x)$
> 3. 需要求系统的剩余量$(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) \, d\tau$

**在卷积神经网络中的直观理解：**

> 1. 卷积核规定了周围像素点对该像素点会有什么样的影响。
> 2. 一个像素点会如何试探周围的像素点（过滤器）。

也就是将一个卷积核大小的像素块按照卷积核的具体数值进行压缩提取特征，以达到提取局部特征的效果，然后再送给神经网络进行判断。

### 8.2 卷积层

**卷积核：**

- 卷积核就像是**一组不同形状的筛子。**
- 训练过程就是不断调整卷积核，也就是不断调整筛子的大小，让某种**特征**能通过。
- 区分不同图片（猫 vs 狗）时，不是单个筛子直接说出答案，而是所有筛子过滤后的结果（特征图）**组合**起来，再由最后的分类器（全连接层）判断。

**填充与步幅：**

1. 填充：

> 在输入矩阵的周围n圈填充0

作用：

> 1. 控制输出特征图的空间尺寸。
> 2. 保留图像边缘和角落的特征（这些地方被扫描的次数更少）。
> 3. 保持特征图中心与边缘的平衡（中间部分往往容易被多次扫描，填充之后可以缓解“中心偏向”）。
> 4. 配合步长精确控制下采样，可以配和步长来精准控制输出的矩阵大小。

2. 步幅：

> 每次扫描移动的步数

**卷积核大小与卷积层输入输出的关系：**

<img src="img/卷积核大小与该卷积层输入输出的关系.png" alt="图片" style="zoom:33%;" />

> **输入通道数（层数）要和卷积核的通道数（层数）相同，然后卷积核的个数，决定了该卷积层的输出通道数（层数）。**

**卷积层对相对位置非常敏感**

> 卷积层对于位置信息非常敏感。
> 例如：让CNN区分数字6和数字9，因为CNN只能提取局部特征，所以在CNN中，这两个数字就是一个圈圈加一个勾勾。如果CNN对位置信息不敏感的话，也就无法区分圈圈和勾勾的相对位置，自然无法分辨6和9。所以对于CNN这种提取局部信息来判断图片的模型，必须对每个特征的相对位置敏感。

**不同卷积层所提取的特征之间的区别**

- **底层卷积：**它会去匹配一些非常简单的模式，比如明暗交界线（脸和背景的边界）、45度方向的线条（眉毛的角度）、小圆点（可能是瞳孔）。这些特征在猫脸、车轮上也可能出现，所以非常通用。
- **中层卷积：**在底层的基础上组合，开始匹配局部部件：一个弧形加一个圆点（眼睛）、两条短线加一个空隙（嘴巴）、一个半圆（耳朵）。这些已经和人脸相关了。
- **高层卷积：**感受野覆盖了几乎整张脸。它匹配的是完整的脸型、各个部件之间的全局配置（两只眼睛在鼻子上面，嘴巴在下面）。甚至能区分“正面脸”还是“侧脸”。这一层的一个卷积核可能只对“闭着的左眼”有强烈响应，另一个只对“微笑的嘴”有响应。

**数学原因**：

1. **感受野随层数指数增长**
   假设每层都是3×3卷积，步长1，那么：
   - 第1层输出上的1个像素，看到输入3×3区域。
   - 第2层输出上的1个像素，看到上一层3×3区域，每个又对应输入3×3，所以实际看到输入5×5区域。
   - 第3层看到7×7，依此类推。

因此，**高层卷积的一个神经元，其感受野覆盖了原始图像的很大一部分。**它能“看到”全局布局。

**Conv2d和Conv3d**

> Conv2d就是处理一张图片，Conv3d就是比图片多一个维度，这个维度可以是时间（视频）也可以是空间（核磁共振3D图，就是多个剖面图）。

### 8.3 池化层

**池化层的作用：**

> 卷积层对位置信息比较敏感，所以能区分6和9。但是为了防止**过拟合**，我们需要引入池化层。

> 卷积层用来区分6和9，而池化层用来区分不同人写的9。也就是说卷积层用来拟合，池化层用来泛化也就是防止过拟合。可以增加鲁棒性。

> 池化层也具有填充和步幅，池化层没有可学习参数。

**池化层核心思想：**

> **在一个小局部窗口内，只保留最强烈的激活信号，而忽略它具体在这个窗口的哪个位置。**
> **局部窗口内取最大/平均，主动丢弃精确位置信息，换来对小位移和形变的容忍。**

**池化层分类：**

1. 最大池化层：

> 在池化层照射的区域内选取最大的数值代替该区域。

<img src="img/最大池化层.png" alt="图片" style="zoom:33%;" />

2. 平均池化层：

> 在池化层照射的区域内选取平均值来代替该区域。

直观对比：
<img src="img/池化层两种类别直观对比.png" alt="图片" style="zoom:33%;" />

### 8.4 LeNet

<img src="img/LeNet.png" alt="图片" style="zoom:33%;" />

> 其中池化层是平均池化层。

**卷积神经网络设计思想**

> 将一个图片的长宽信息不断地进行压缩，然后将信息压缩进不同的通道中（层数），最后将通道数高且长宽低的数据整理成矩阵或者向量，传入多层感知机中进行收尾学习。
>
> 整体趋势是通道增加而长宽减少。

**卷积神经网络设计思想提炼：**

> **通过交替的卷积和池化（或步长卷积），逐步压缩空间尺寸、扩张通道数，从而将低级空间特征抽象为高级语义特征；最后根据任务决定——分类任务常用全连接层或全局平均池化输出类别，而检测/分割任务则保留空间维度进行逐像素预测。**

### 8.5 AlexNet

**容易误解的数学知识：**

> **凸问题**通常存在这种特性：局部最小值 = 全局最小值
> **凹问题**通常存在这种特性：局部最大值 = 全局最大值
> 真正让AI领域头疼的是非凸非凹的问题，也就是**非凸问题**，这类问题通常局部最优解 ！= 全局最优解，所以容易让AI学习陷入到鞍点和局部最小值点卡住。鞍点问题能通过判断奇异值的正负性来判断是不是鞍点，局部最小值往往还没有很好的解决办法，但是工程领域中也认为局部最优解也可以接受，因为他能满住需求，同时还可以防止过拟合。

**AlesNet与LeNet对比：**


| 对比维度                 | LeNet-5 (1998)                                                                                                                                                                         | AlexNet (2012)                                                                                                |
| :------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| **提出者 / 团队**        | Yann LeCun 等                                                                                                                                                                          | Alex Krizhevsky、Ilya Sutskever、Geoffrey Hinton                                                              |
| **主要应用场景**         | 手写数字识别（MNIST）                                                                                                                                                                  | 大规模图像分类（ImageNet 1000类）                                                                             |
| **输入图像尺寸**         | 32×32×1（灰度图）                                                                                                                                                                    | 227×227×3（RGB彩色图）                                                                                      |
| **网络深度（可训练层）** | **7层**（2个卷积层 + 3个全连接层 + 2个子采样层，通常不计入池化）<br>更常见描述：2卷积+3全连接，共5层？实际经典结构：C1-S2-C3-S4-C5-F6-输出，即2个卷积+2个池化+3个全连接（含输出）共7层 | **8层**（5个卷积层 + 3个全连接层）                                                                            |
| **卷积层数**             | 2层（C1、C3）                                                                                                                                                                          | 5层（Conv1 ~ Conv5）                                                                                          |
| **全连接层数**           | 3层（C5、F6、输出层）                                                                                                                                                                  | 3层（FC6、FC7、FC8）                                                                                          |
| **池化方式**             | **平均池化**（子采样，2×2，步长2，非重叠）                                                                                                                                            | **最大池化**（3×3，步长2，**重叠**）                                                                         |
| **激活函数**             | **Tanh** 或 **Sigmoid**                                                                                                                                                                | **ReLU**（大幅缓解梯度消失，加速训练）                                                                        |
| **局部响应归一化 (LRN)** | 无                                                                                                                                                                                     | 有（在Conv1、Conv2后，增强泛化）                                                                              |
| **Dropout 正则化**       | 无                                                                                                                                                                                     | **有**（仅在FC6、FC7后，比例50%，防过拟合）                                                                   |
| **数据增强**             | 无（或简单的平移/缩放）                                                                                                                                                                | **丰富**（随机裁剪、水平翻转、PCA颜色扰动）                                                                   |
| **参数量**               | 约**6万** (60k)                                                                                                                                                                        | 约**6000万** (60M)                                                                                            |
| **神经元数量**           | 约 20万（主要是子采样层降低）                                                                                                                                                          | 约 65万                                                                                                       |
| **GPU 并行设计**         | 单CPU/GPU即可                                                                                                                                                                          | **双GPU分路**（由于显存限制，模型结构分为上下两路）                                                           |
| **主要创新点**           | • 奠定CNN基本框架（卷积-池化-全连接）<br>• 梯度下降与反向传播成功应用于CNN                                                                                                           | • 首次将深度CNN用于大规模图像识别<br>• 引入ReLU、Dropout、重叠池化、LRN<br>• 充分运用GPU并行计算与数据增强 |
| **历史意义**             | 现代CNN的**鼻祖**，验证了端到端学习的有效性                                                                                                                                            | **深度学习复兴的里程碑**，开启ImageNet时代                                                                    |

### 8.6 VGG

**创新点：**

**全部使用 3×3 小卷积核**

- 两个 3×3 卷积堆叠等效于一个 5×5 感受野；三个堆叠等效于 7×7。
- **优点**：更少的参数（3×3×C×C×2 vs 5×5×C×C）、更多的非线性（多次ReLU），加深网络的同时控制参数量。

**网络深度大幅提升：**

- VGG16 有 **13 个卷积层** + **3 个全连接层** = 16 个可训练层。相比 AlexNet 的 8 层，深度翻倍，证明了深度的有效性。

**统一的架构设计：**

- 所有卷积层配置高度一致，结构简洁、易于理解和复现。（VGG块，封装）

**参数量巨大（约 1.38 亿）：**

- 虽然卷积层参数较少，但 全连接层占绝大部分参数（约 1.2 亿），导致模型臃肿、推理慢。

**训练技巧：**

- 继承 AlexNet 的 ReLU、Dropout、数据增强等。
- 采用多尺度训练（将图像缩放到不同尺寸再裁剪）提升泛化。

### 8.7 NiN

**NiN块：**

- 1个卷积层后面跟2个全连接层（1×1的卷积层可以看作全连接层），本质上这两个全连接层是两个1×1的卷积层。

**NiN结构：**

- 无全连接层
- 交替使用NiN块和步幅为2的最大池化层
  - 逐步减少高宽和增大通道数
- 最后使用全局平均池化层得到输出
  - 其输入通道是类别数

### 8.8 GoogleLeNet

**Inception块**

<img src="img/GoogLeNet.png" alt="图片" style="zoom:33%;" />

> 对输入分别复制输入4个层中。

### 8.9 批量归一化（BN）

**BN是什么：**

> 把每一层输入的数据“拉回”标准形态（均值0、方差1），再让网络自己微调成最合适的形态，从而让训练更稳、更快、更不挑初始化。

**通俗说法：**

> BN层先将每一层的数据归一化，让他们符合标准状态（均值=0，方差=1），然后再让网络自己搜索γ和β，γ用于用来拉伸和压缩差异β用来平移，通过γ和β来使得数据最终符合神经网络的需求，**因为并不是所有的输入都要标准状态，我们需要适当的用γ和β来调整。**

**为什么需要BN层：**

**在深度学习中，因为链式法则导致深层网络和浅层网络的梯度不一样甚至差异巨大，会导致梯度变小。这也就是在前向传播的时候，因为梯度大，深层网络很快就收敛了，但是浅层网络因为梯度小而收敛慢，但是数据是从浅层向上传播的，如果浅层没有收敛的话，深层的收敛也就没有意义，类似于木桶效应。**

**易错误解：**

> 通常都是梯度消失导致的深层先收敛问题。也存在梯度变大导致浅层先收敛，但是这种情况很小，当出现这种情况的时候，因为深层的梯度不为零，传到浅层的时候，梯度也不可能为0，所以还会继续更新。

**BN层的作用：**

1. 没有BN层：

- **内部协变量偏移**：每层的输入分布随前层参数变化而变化，导致后层需要不断适应新的分布。
- **梯度不稳定**：浅层参数的微小变化，经过多层放大/缩小后，可能造成深层梯度爆炸或消失。为了安全，学习率必须设得很小。
- **木桶效应：**即使深层梯度大、收敛快，但浅层梯度小、更新慢，整个网络的实际收敛速度由最慢的浅层决定。
  结果就是：**需要很多轮迭代（epochs）**，总计算量大。

2. 有BN层：

- **切断前层参数变化对后层输入分布的剧烈影响**：后层看到的分布总是稳定的（均值方差固定），不需要花精力去适应分布变化。
- **梯度流更平滑**：每层输入被限制在合理的范围（不会太大或太小），反向传播时梯度可以更稳定地传递到浅层。
- **浅层的梯度不再被深层“压制”**：因为每层的输出都被标准化，梯度的尺度被规范化，浅层也能获得有效的更新信号。
  **浅层收敛变快**，木桶效应被缓解。网络不再需要小心翼翼地用很小的学习率，可以大胆地用大学习率，迭代次数大幅减少。
  此时学习率可以挑大，迭代次数可以减少，总体提升计算效率。

**当BN层可能起副作用的情况：**

> 因为BN层的参数γ和β也需要进行**梯度下降**来更新，所以BN层也会提升计算量。
> 那么有些情况会导致BN提升的计算量甚至大于它减少的计算量。


| 场景                                  | 分析                                                                                                                                                                       |
| :-------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **极小的网络**（如 LeNet-5 在 MNIST） | 原网络迭代已经很快，BN 的额外计算可能使总时间变长。很多小模型上不加 BN 反而更快。                                                                                          |
| **batch size 非常小**（如 ≤4）       | 均值和方差估计不准，BN 效果差，甚至拖慢收敛。此时 LayerNorm 或 GroupNorm 更合适。                                                                                          |
| **推理阶段**                          | BN 在推理时没有参数更新，但仍有$\gamma/\beta$ 和固定的 $\mu/\sigma$，计算量增加但不会减少迭代数。不过推理通常不需要收敛加速，BN 的收益来自提高精度或稳定性，而非节省时间。 |

**BN的正则化：**

> BN的正则效果更像是意外，因为BN最初想法是用来加快收敛减少计算量的

**正则化效果产生的原因：**

> 引入**“幽灵误差”**。
> 这个幽灵误差是因为每次BN计算样本的均值和方差都是根据**当前batch的全部数据**来计算的，但是实际上最科学的均值和方差应该是从**所有batch的数据**中计算得来，也就是从**全部数据**中获得。这其中的误差就叫做幽灵误差。
> batch越大，也就导致幽灵误差越小，从而导致正则化效果差。有的研究会在大的batch中，拆成更小的batch来计算，因为**小的batch正则化效果强**。

**BN 的正则化，本质上是因为训练时使用了不完美的、随 batch 变化的统计量来近似全局统计量，这种近似误差带来了有益的正则化噪声。**

**BN具体作用：**

> 特征和通道差不多，卷积就是不断地减少高宽提升通道，每一个通道存储着一个像素点对应的各种特征。

1. 作用与线性层时，对输出的值的每一个特征值进行归一化。
2. 作用与卷积层时，对输出的值的同一个通道的图内的所有值进行归一化。
   **例如：**
   **一个通道只有2个样本**
   通道1的样本1
   [1.0, 2.0]
   [3.0, 4.0]

```
通带1的样本2
       [5.0, 6.0]
       [7.0, 8.0]
```


对[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]这8个值进行归一化。

**BN的应用：**

> **通常用于很深的神经网络，防止链式反应导致梯度变小。**

**注意：**使用了BN层的卷积层可以不使用偏置bias。

### 8.10 ResNet（残差网络）（很常用）

**ResNet直觉：**、

> 学习复杂的东西不会一次性学会，而是先把简单的学通透，然后再一点点补充残差。

举个例子：

> 你学骑自行车。普通网络是：你必须从零学会所有动作——平衡、踩踏板、转弯……如果让你学高级技巧（比如脱把），你可能连平衡都忘了。
> ResNet呢？它说：“你先把普通骑行学透（这部分是x），然后只需要额外学一点点新东西（F(x)就是‘脱把’的那个小调整）。最后输出 = 普通骑行 + 这个小调整。”这样，你永远不会忘记基础，只需要学残差——就是“还差的那一点点”。

**额外新学的信息就是残差**

**为什么叫“残差”：**

> 因为神经网络想学的是从输入到输出的完美映射H(x)。但H(x) = x + 多余部分。如果直接学H(x)很难，那就去学F(x) = H(x) - x —— 也就是剩下没被x解释的部分。当网络很深时，它至少可以轻松让F(x) = 0，这样输出就等于x，啥也不变——这叫恒等映射。所以即便中间有几层没用，网络也不会变差，反而能安心地多堆层数，只在一两个地方学真正的残差。

**易错知识**

> g(x) = f(x) + x，有没有一种可能，就是f(x)学坏了，导致离目标值变远了

**这是完全有可能的，但是ResNet有一个设计可以防止这种情况发生。它让“什么都不做”变得超级容易**

**学坏之后，梯度下降会自动把f(x)往0方向推**

**ResNet可以缓解梯度消失，从而可以大幅加深网络深度**

ResNet的一个残差块是 $g(x) = f(x) + x$。反向传播时，从输出 $g$ 回到输入 $x$ 的梯度分为两条路：

1. **经过 $f(x)$ 的路**：这跟普通网络一样，要经过权重、激活函数，梯度可能会变小。
2. **恒等映射的路**：就是那个 $+x$，它的导数**正好是1**，不乘以任何小于1的数。

所以总的梯度 $\partial g / \partial x = \partial f / \partial x + 1$。

就算 $\partial f / \partial x$ 很小（甚至为零），那个 **+1** 依然存在。梯度就像水管里的水，除了那条容易堵塞的小管子（$f(x)$ 路径），旁边还有一根**粗直的备用管道**（恒等映射），水永远可以通畅地流回去。

> 无论梯度有多小，都有一个1可以兜着。

**实际使用时：**要先将g(x)+x然后在将其送入激活函数中一起激活。

### 8.11 数据增广（图片增广）（数据增强）

**数据增广：通过对已有数据进行变换，让数据具有更多的多样性，从而提升模型的泛化能力。**

> 如果是语言数据，我可以加入背景噪音来达到数据增强的效果。
> 如果是图像数据，我可以通过改变图像颜色和形状达到数据增强的效果。

通常只有在训练的时候才会进行数据增强，测试的时候不会，可以将数据增强看成正则化。

**图像中的数据增强**

- **翻转：**上下翻转、左右翻转。
- **切割：**从原图中切割一部分下来，然后拉伸到对应的尺寸。
- **颜色：**改变原图的颜色和色温。

### 8.12 模型微调和迁移学习（最重要的技术）

**1. 迁移学习：别从零发明轮子**

深度学习里，训练一个大型神经网络就像训练一个婴儿——你得给他看几百万张猫、狗、汽车、飞机……他才能慢慢学会认东西。这个过程要花好几天、用好几百块显卡、电费都能把账本烧个洞。

聪明的办法是：**直接拿别人已经训练好的模型**。比如谷歌、Meta这些大公司已经用海量数据训练好了一个叫“ResNet”或者“BERT”的家伙。它虽然没见过你手头的问题，但它已经学会了很多通用的本领——比如“边缘检测”、“纹理识别”、“词语之间的关系”。

现在你要做的是：

- 把它的“大脑”拿过来，冻结前面90%的层（保留那些通用能力）。
- 只替换最后一两层（原来它输出1000类物体，你现在只要输出猫和狗两类）。
- 用你的少量数据（比如100张猫狗照片）去训练这最后两层。

这就像你去参加“速成物理竞赛”——不需要再自己推导牛顿定律，直接站在牛顿、麦克斯韦的肩膀上，只学那最后几道高级题的解法。

**迁移学习的关键**：源任务（别人的大数据任务）和目标任务（你的小任务）要有点相似。用“自行车经验”学“摩托车”很管用，但用来学“炒菜”就悬了。

**2. 模型微调：把别人的知识拧一拧**

微调是迁移学习的一种更细致的玩法。

别人那个训练好的模型，就像一把通用的瑞士军刀——能切、能锯、能开瓶，但可能不够锋利。你的任务需要它在“切奶酪”这个动作上特别精准。怎么办？

你不会重新打一把刀，而是**把刀刃再磨一磨**。

**例如：**

某神经网络：

> 输入图像 → [卷积层组1] → [卷积层组2] → ... → [卷积层组5] → [全局池化] → [全连接层（1000个神经元）] → Softmax → 输出1000类概率

- **前5个卷积层组**：学会了边、角、纹理、形状、物体部分（车轮、眼睛）等**通用特征**。
- **最后的全连接层**：把通用特征组合起来，专门识别 ImageNet 里的1000类（比如“金毛犬”、“校车”）。

**迁移学习（极少数据，每类5张图）的操作**

1. **删除原有的全连接层**（那1000个神经元）。
2. **创建一个新的全连接层**，它的输出维度等于你任务里的类别数（比如2类：猫和狗）。这个新层的参数是随机初始化的。
3. **冻结前面所有卷积层组**（设置 `layer.trainable = False`）。
4. **只训练这个新全连接层**（设置新层 `trainable = True`）。

**结果**：

- 前面的卷积层参数**一个字都没变**——它们还是原来识别“车轮”、“眼睛”的权重。
- 只有最后一层那几千个参数（比如 `2048×2`）被你的5张图片训练了。
- 训练结束后的模型：输入一张图 → 前面冻结的卷积层提取通用特征（例如“毛茸茸”、“两只耳朵”） → 新全连接层学习如何把这些特征映射到“猫”或“狗”的得分。

**所以迁移学习对原来网络做的改动是：**
**切掉原分类头，换上一个新分类头，然后只训练这个新头。原来的身体纹丝不动。**

**模型微调（数据较多，每类>100张）的操作**

1. 保留模型所有层（包括原分类头，或者也换一个新头）。
2. **不解冻所有层**（或者部分解冻，比如只解冻最后几个卷积块）。
3. 设置一个**很小的学习率**（比如原始训练学习率的 1/10 或更小）。
4. 用你的数据继续训练所有（或选定的）层。

**结果**：

- 原来的卷积层参数会发生微小的变化，从“通用特征”向“任务特有特征”偏移。
- 新分类头（如果有替换）快速学习映射。

**4. 极少数数据怎么办？（“几乎为零”）**

如果每类只有1-2张图，连训练新分类头都危险。这时候更极端的做法是：

- 把预训练模型当做一个固定的**特征提取器**。
- 对每张图，提取倒数第二层的特征向量（比如2048维的向量）。
- 用这些特征向量做**近邻分类（KNN）**或者直接算余弦相似度。整个过程中没有任何“训练”发生，网络参数完全不变。
- 这也属于迁移学习的一种，只是“改动”为零——你只是用了它的输出，连最后一层都没加。

**5. 总结表（费曼喜欢表格）**


| 方法                     | 训练数据量           | 对预训练模型层的改动                      | 哪些参数被更新                   |
| -------------------------- | ---------------------- | ------------------------------------------- | ---------------------------------- |
| **迁移学习（特征提取）** | 极少（每类<10张）    | 删除原分类层，新增分类层；冻结所有原层    | 只有新分类层的参数               |
| **部分微调**             | 中等（每类10-100张） | 冻结前几个卷积块，解冻后几个块 + 新分类头 | 后几个块 + 新分类头的参数        |
| **微调（全部层）**       | 较多（每类>100张）   | 不解冻任何层，或部分解冻                  | 所有层（或选定的层），用小学习率 |

**6. 费曼的总结（带点小幽默）**

迁移学习就像**借大脑**——别人学了一辈子的常识，你直接拿来用，只换最后一个“输出单元”去决定“这是披萨还是不是披萨”。

模型微调就像**借大脑并给它做个小手术**——不是换最后一个单元，而是轻轻拧一下前额叶的几根神经，让它更擅长分辨“芝加哥深盘披萨”和“那不勒斯薄底披萨”。

如果你手头数据特别少（比如10张图片），就用迁移学习（只训练最后一层），防止过拟合（别把偶然的噪点当真理）。
如果数据中等（比如1000张），就微调所有层，但学习率要小。
如果数据超大（比如100万张），那你就可以从零开始训练——但为啥呢？你又不是没脑子借用。

最后记住：**好的科学家从不去重新发现万有引力，他们只负责在引力上面荡秋千**。迁移学习和微调，就是让你在深度学习里荡得又快又稳的工具。

总结：

> 模型微调：用一个极小的学习了对所有参数进行微调也可以只调整部分参数。
> 迁移学习：前面所有的网络层的参数都不去修改，最后修改一整个输出层。

### 8.13 卷积层、BN层、激活层和池化层顺序（重要）

想象你在厨房里处理食材。你拿进来一筐土豆——那是原始图像。你得先洗、再切、再焯水、最后才下锅。顺序很重要——你不会先焯水再洗，那焯的水全是泥。卷积网络这几个层的顺序，道理一模一样。

**第一步：卷积。为什么是第一个？**

因为卷积是「看」的操作。它用一个小权重窗口在图像上滑动，计算加权和。这是线性变换——输入乘以权重，加起来。卷积出来的结果，数值分布什么样你没法预测。可能全偏在一边，可能整体偏大或偏小。

如果这时候直接把激活函数（比如ReLU，把所有负值变成零）怼上去，而卷积输出的均值偏到了负的那边，ReLU可能干掉一大半信号。你就剩寥寥几个活着的神经元，网络基本瞎了。所以卷积之后需要「整理」一下分布。

**第二步：BatchNorm。为什么在激活之前？**

BN把卷积的输出拉到均值为0、方差为1的分布上。激活函数——尤其ReLU——在0附近工作得最舒服。输入既不太大也不太小、正负各半的分布，让激活函数能做出有意义的「决定」：哪些该留、哪些该扔掉。

如果反过来——先激活再BN——你就是在归一化一堆已经被ReLU砍掉一半的、不对称的数据。ReLU把负值全变成0了，BN再在这个「残废」的分布上算均值和方差——算出来的东西就不对了。就像一个厨师先切了菜再洗：洗是能洗，但断面都泡烂了。而且激活之后的分布不是高斯型的——它是畸形的。BN假设数据大致对称，因为它用均值和方差来描述。ReLU之后的分布严重偏斜，BN效果打折扣。

**BN在激活之前，是因为它要给激活函数准备一份干净的输入。** 就这么简单。

**第三步：激活。为什么在BN之后、池化之前？**

激活函数是非线性。它让网络不再只是做加权平均——开始能表达「如果满足某个条件，就激活；否则不激活」这种复杂模式。

关键问题：非线性应该在空间压缩之前还是之后发生？如果你先池化再激活——先扔掉了空间信息，然后在缩小的图上做非线性变换。把细节扔了再去找细节——这不荒唐吗？反过来，先激活再池化：让网络先在完整分辨率上「判断」每个位置有没有特征，然后再压缩。池化保留下来的，是已经经过非线性变换的、有意义的特征响应，而不是原始线性响应的粗糙平均。

**池化应该压缩已经处理好的信息，而不是压缩还没处理的原材料。**

**第四步：池化。为什么是最后一个？**

池化做空间降采样——取一个窗口里的最大值或平均值。它是「总结」操作。把总结放在最后，道理和写论文一样：先收集数据、分析、得出结论——最后才写摘要。没人先写摘要再做分析。池化还有一个实际好处：减少计算量。放在最后这一步，它压缩的是已经充分处理过的特征图——信息密度高，丢掉的是冗余。**通常池化层是需要压缩图片的宽和高。**

**整体流水线：**

原始像素 → [卷积：线性提取] → [BN：整理分布] → [激活：非线性判断] → [池化：压缩总结] → 下一层

每一步都在为下一步做准备。这不是谁拍脑袋定的——它就是顺着数据流的自然需求走。

**一个有意思的反例**

ResNet v2（2016年）提出了另一种顺序：BN → Activation → Conv。把激活放在卷积之前。为什么？因为在很深的网络里，原始顺序（Conv → BN → ReLU）的信号路径有阻碍——梯度往回传的时候，经过ReLU被截断了一部分。如果把BN和ReLU放在Conv前面，梯度就不用穿过非线性。

这说明什么？说明这个顺序不是「真理」，它是一种工程选择。不同的约束条件（深还是浅、要不要残差）会导向不同的最优排列。但默认的 Conv → BN → Activation → Pooling 在绝大多数场景下就是最自然的、最不容易出错的。

**总结一句话**

卷积提取特征，BN把特征拉到合适的尺度，激活做非线性判断，池化把判断结果压缩——每一步的输入都是上一步精心准备的。换顺序不是不行，但你得知道你在打破什么、为什么要打破。

## 9. 序列模型

### 9.1 数学理解

**之前的模型都是独立的随机变量，即每一个样本之间没有任何关系，可以被随意打乱顺序。**

**序列模型的每一个样本是不独立的随机变量，也就是说样本之间的顺序也是信息的一部分，不能被随意打乱。**

**为了让模型能理解元素之间的顺序关系，我们需要用贝叶斯公式来表示。**

**有两种数学方案可以实现序列模型**

1. 马尔科夫

<img src="img/马尔可夫.png" alt="图片" style="zoom:33%;" />

> 假设当前数据之和前t个数据有关，那么求当前数据的方式就变成了在已知前t个的情况下进行预测下一个数据。
>
> 因为预测数据的时候是依据固定前t个数据，**我们可以把这t个数据以特征的形式送入线性模型中（多层感知机），然后用输出的结果来进行预测下一个数据。**

2. 潜变量模型

<img src="img/潜变量模型.png" alt="图片" style="zoom:33%;" />

> h是潜变量（隐变量），它代表前面所有数据对当前数据影响的总和，h和x是同时进行迭代更新的。
>
> 例如：**已知h和x是求h\`和x\`，我们先通过x和h来求得h\`，然后根据h\`和x来计算x\`，此时进行了两次计算，也可以看成是两个模型。**

### 9.2 RNN

#### 9.2.1 **RNN是潜变量自回归模型**

> Dense：稠密网络，全连接网络。

<img src="img/RNN具体原理.png" alt="图片" style="zoom:33%;" />

**更新隐藏状态h：**

h_t = tanh(W_hh · h_{t-1}  +  W_xh · x_t  +  b_h)
y_t = W_hy · h_t + b_y

> 就两行。拆开看：
> 第一行：新的记忆 = tanh(旧记忆经过线性变换 + 新输入经过线性变换)。tanh 是激活函数，把值压>到 [-1, 1]之间，防止记忆在循环中爆炸。
> 第二行：从当前记忆输出预测结果。比如情感分类就是输出「正面/负面」的概率。
> W_hh、W_xh、W_hy 是三组权重矩阵。每一步用的 W 是完全一样的——这就是「循环」的含义。

**根据记忆h_t-1和当前的输入x_t来更新现在的记忆h_t，然后通过更新之后的记忆h_t来计算预测的y_t+1，然后损失函数loss就是预测的y_t+1和真实的x_t+1之间的差值（训练过程是有每一个真实的x），然后进入下一轮更新，下一轮h_t+1的更新用真实的x_t+1来更新。以此类推。**

> 「用真实的 x_{t+1} 来算 h_{t+1}」——这个完全正确，而且很多人第一次学 RNN
> 会漏掉这一点。训练的时候，不管模型预测了什么，下一步的输入永远是真实的上一个词。这有个名字叫 teacher
> forcing——老师强行把正确答案塞回去，不让模型在自己的错误上越跑越偏。

#### 9.2.2 **RNN的4种工作模式**


| 模式                                         | 叫什么                   | 干什么                         | 例子                    |
| ---------------------------------------------- | -------------------------- | -------------------------------- | ------------------------- |
| 每一步都输出，输出的是下一个词，预测下一个词 | many-to-many（语言模型） | y_t预测x_t+1                   | GPT训练、文本生成、问答 |
| 只看最后一步输出                             | many-to-one              | 看完整个序列，给一个判断       | 情感分类、垃圾邮箱检测  |
| 生成一段序列                                 | one-to-many              | 根据一个词生成一段话           | 文本生成                |
| 对词进行分类                                 | many-to-many             | 根据一段序列生成对应的一段序列 | Tag生成                 |

#### 9.2.3 如何衡量一个模型的好坏（困惑度）

<img src="img/困惑度.png" alt="图片" style="zoom:33%;" />

> 困惑度就是对交叉熵去e指数，把交叉熵损失的值扩大。

#### 9.2.4 梯度剪裁

**RNN的梯度回传的时候，每一轮都要乘W_hh，所以很容易梯度爆炸。**

**可以有效预防梯度爆炸**

**具体如何剪裁：**

不管反向传播算出多大的梯度，我把它的范数压到一个上限以内。超了，就等比例缩小。

```python
if ||g||_2 > max_norm:
      g = g * (max_norm / ||g||_2)
```

> **就这么简单。算一下所有梯度的 L2 范数。** **如果超了阈值，等比例缩小到阈值。方向不变，只缩长度。**
> **比如 max_norm = 1.0，你算出来的 ||g|| = 5.0，那就把每个梯度分量乘以 0.2。**
> **方向保留，步长控制。 这是关键——你没改变梯度指向哪里，你只限制了它迈多大步。**

**关键问题：**

1. max_norm 设多少？

没有理论最优值。RNN 通常是 1.0 或 5.0。太小了训练慢（每一步都踩刹车），太大了等于没裁剪。你写 IMDB 的时候从 1.0开始，看 loss 曲线——如果 loss 还是偶尔爆炸就往下调，如果 loss 下降太慢就往上调。

2. CNN 需要梯度裁剪吗？

通常不需要。CNN 深度是固定的（比如 5 层卷积），梯度不会在时间维度连乘。但如果你写的是一个 200 层的ResNet——那也可以加，不过那是你想太多的问题。当前你不需要。

**适用范围：**

> 对任何可能产生梯度爆炸的模型都能使用

**RNN是“梯度剪裁”的头号用户**

不是因为 RNN 特殊。是因为 RNN 最容易炸。

BPTT 往回传 100 步，每一步乘一次 W_hh。就算 W_hh 每个元素都小于 1，连乘 100
次也可能在某个方向上放大。这不是概率问题——训练久了，终有一次炸。

全连接层不一样。一个 5 层的 MLP，反向传播只是 5 次连乘。炸的概率低得多。但这不是说它不会炸。

全连接层也会炸，虽然少见

**想象一个 100 层的 MLP，没有 BN，没有残差连接。反向传播，梯度从第 100 层传到第 1 层，每层乘一次权重矩阵。和 RNN一样的问题。**

你初始化不好（比如权重初始值偏大），第一轮训练就炸。或者你学习率设太大了，某一步的更新把参数推到了一个「一阶导很大」的
区域，下一轮梯度直接飞了。

这时候——梯度裁剪，通用的。```clip_grad_norm_```不管你是什么模型，它只管梯度范数大不大。

> RNN的每一步意味着MLP的每一层，所以RNN极容易产生梯度爆炸。

#### 9.2.5 RNN具体流程

RNN 解决的核心问题：**处理变长的、有顺序依赖的序列数据。**

CNN 的卷积核看的是固定大小窗口，但语言不是这样工作的——"我三年前在纽约认识的那个朋友，今天来北京看我了"，"我"和"朋友"之间隔了 8 个字。这些长距离依赖不是固定窗口能抓住的。

RNN 做的是：每读一个字，脑子里的"理解"就更新一次。带着一个不断更新的**记忆**往前走。

---

**完整管道：从原始文本到最终输出**

以"这部电影很好看" → 情感分类（正面/负面）为例。

**第 0 步：分词（Tokenization）**

做一张表，每个词给一个编号：

```
{'<PAD>':0, '<UNK>':1, '这':2, '部':3, '电影':4, '很':5, '好看':6, '难看':7, '无聊':8, '精彩':9}
```

"这部电影很好看" → [2, 3, 4, 5, 6]

分词就是查表换数字。PAD 是填充符号（短句子后面补 0），UNK 是未登录词。实际上 IMDB 英文分词更复杂（拆 don't、转小写），但本质就是查表。

**第 1 步：词向量（Embedding）**

整数不能直接喂给神经网络——"好看"(id=6)和"难看"(id=7)数值很接近，但意思完全相反。

Embedding 层：把每个整数映射成一个向量。向量里的数字是训练出来的——语义相近的词，向量在空间里靠得近。

embedding_dim = 4 为例：

```
词 id=2 '这'   → [ 0.12, -0.34,  0.56,  0.78 ]
词 id=3 '部'   → [ 0.91, -0.12, -0.45,  0.33 ]
词 id=4 '电影' → [-0.23,  0.67,  0.11, -0.89 ]
词 id=5 '很'   → [ 0.45,  0.23, -0.67,  0.12 ]
词 id=6 '好看' → [ 0.78, -0.45,  0.34, -0.56 ]
```

输入变成 5×4 矩阵：5 个时间步，每个时间步 4 维向量。

**第 2 步：RNN 细胞（核心机制）**

RNN 只有两组可学习参数：

```
W_xh: 输入 x 映射到隐藏空间    (input_size × hidden_size)
W_hh: 旧记忆映射到新记忆        (hidden_size × hidden_size)
b_h:  偏置
```

核心公式：h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)

hidden_size = 3 为例，h_0 = [0, 0, 0]。

**时间步 t=1，读入"这" x_1 = [0.12, -0.34, 0.56, 0.78]：**

h_0 · W_hh = [0, 0, 0]（第一步，旧记忆为零）

x_1 · W_xh：矩阵乘法得 [0.206, -0.096, -0.314]

h_1 = tanh([0.206, -0.096, -0.314]) = [0.203, -0.096, -0.304]

**h_1 = [0.203, -0.096, -0.304]**。这三个数字就是读完"这"之后的"理解"。

**时间步 t=2，读入"部" x_2 = [0.91, -0.12, -0.45, 0.33]：**

h_1 · W_hh = [0.076, -0.343, -0.074]

x_2 · W_xh = [0.107, -0.213, 0.275]

总和 = [0.183, -0.556, 0.201]

h_2 = tanh([0.183, -0.556, 0.201]) = [0.181, -0.506, 0.198]

注意：h_1 的信息已经混进 h_2 了——第二维从 -0.096 变成 -0.506，"部"这个词强化了负向。

**依此类推到 h_5，假设 h_5 = [0.723, -0.891, 0.456]**

这个向量是读完"这部电影很好看"五个字之后，脑子里的全部理解。

**第 3 步：分类头（many-to-one）**

情感分类只在最后一步输出：

```
FC 层: logits = h_5 · W_fc + b_fc
        = [0.723, -0.891, 0.456] · W_fc + [0.1, 0.1]
        = [1.733, -1.019]

Softmax: prob_正面 = e^1.733 / (e^1.733 + e^(-1.019)) ≈ 0.940
         prob_负面 = 0.060
```

输出：94% 正面。预测正确。

**第 4 步：Loss 计算**

交叉熵：loss = -[1 × ln(0.940) + 0 × ln(0.060)] = 0.062

反向传播（BPTT）从 h_5 沿时间往回传：h_5 → h_4 → ... → h_1 → Embedding → 更新所有参数。

---

**整条管道一览：**

```
输入文本 → 分词 → Embedding → RNN(h_0→h_1→...→h_5) → FC → softmax → loss
                              ↑W_xh,W_hh 同一组参数反复用
```

**Teacher Forcing：**

训练时每一步输入用的是真实的上一个词，不是模型预测的词。防止模型在自己的错误上越跑越偏。

**三个关键认知：**

1. Embedding 是训练出来的，不是查死的。训练中梯度会更新 Embedding 表，让语义相近的词在向量空间里真正靠近。
2. W_xh 和 W_hh 在所有时间步共享。不是 5 个不同的盒子，是同一个盒子用了 5 次。这就是"循环"的含义。
3. 梯度顺着时间反向传播时，W_xh 和 W_hh 的梯度是所有时间步累加的——同一个 W 在 5 个时间步各用了一次，总梯度来自 5 个位置的贡献之和。这就是 BPTT = 展开 + 反向传播。

> RNN = 同一个盒子在时间轴上反复用。权值共享从空间（CNN卷积核）换到了时间（RNN细胞）。

---

**RNN 四种工作模式：**


| 模式                   | 说明                     | 例子                   |
| ------------------------ | -------------------------- | ------------------------ |
| many-to-one            | 看完整个序列，给一个判断 | 情感分类、垃圾邮件检测 |
| many-to-many（对齐）   | 每步都输出，输入输出等长 | 词性标注、NER          |
| many-to-many（不等长） | 编码器读完→解码器输出   | 机器翻译、摘要         |
| one-to-many            | 一个输入→生成序列       | 图像描述               |

IMDB 情感分类是第一类 many-to-one——只在最后一步输出。

**API 速查：**

```python
lstm = nn.LSTM(input_size=256,      # Embedding 维度
               hidden_size=128,     # 隐状态维度
               num_layers=2,        # LSTM 层数
               batch_first=True,    # (batch, seq, features) 而非默认的 (seq, batch, features)
               bidirectional=True)  # 双向，输出维度翻倍

output, (h_n, c_n) = lstm(x)   # x: (batch, seq_len, input_size)
# output: (batch, seq_len, hidden*D)  每步输出
# h_n:    (layers*D, batch, hidden)   最后一步隐状态（many-to-one 用这个）
# c_n:    (layers*D, batch, hidden)   最后一步细胞状态
```

实际直接用 `nn.LSTM` 或 `nn.GRU`，不用原始 `nn.RNN`——原始 RNN 梯度消失严重，实际没人用。

> BPTT 梯度传播 = W_hh 连乘 seq_len 次。特征值 <1 → 指数衰减（梯度消失），>1 → 指数增长（梯度爆炸）。这是 RNN 的根本缺陷，LSTM/GRU 的门控机制就是为了修这个问题。

#### 9.2.6 GRU（门控循环单元）

**核心**

- z（更新门）：当前词值不值得写进记忆
- r（重置门）：旧记忆对理解当前词有没有用

**理解：**

z（更新门）和重要性：基本对应

z_t 决定「新候选在最终记忆中占多大比例」。

z_t ≈ 0 → 旧记忆原封不动。这个词不值得写进记忆。

读到"的"：
z_t = 0.08  →  h_t ≈ h_{t-1}
模型说：这个词不重要，我不更新。

z_t ≈ 1 → 覆盖旧记忆。这个词有值得写进去的新信息。

读到"好看"：
z_t = 0.91  →  h_t ≈ h_tilde_t（全新的记忆）
模型说：这是关键信息，记下来。

所以 z 和重要性确实直接相关——z 越小越不重要，z 越大越值得写。

---

r（重置门）和重要性：不对应

r_t 的逻辑不一样。它问的是：「在计算新候选时，旧记忆有没有参考价值？」

r_t ≈ 0 → 旧记忆没用，忽略它，从零开始理解当前词。
r_t ≈ 1 → 旧记忆有用，带上上下文一起理解当前词。

r_t 不论大还是小，都不能推导这个词重不重要。

什么时候 r_t 小？句子开头。
读到"这"：
r_t = 0.23  →  旧记忆基本被忽略
不是"这"不重要，是上一句的旧记忆和这一句没关系。模型需要重置——忘掉上句，重新开始。

什么时候 r_t 大？在句子中间。
读到"很好看"：
r_t = 0.88  →  旧记忆充分参与
不是"很好看"比"这"更重要，是需要前面的上下文才能理解它在修饰什么。

GRU 解决 RNN 的根本缺陷：RNN 每一步强制更新记忆，没有说"不"的能力。读入"的""了"这种虚词也必须更新 h——每一步经过 tanh 都在消耗信号，早期信息被洗没。


| z_t（更新门） | r_t（重置门） | 模型在说什么                                                                               |
| --------------- | --------------- | -------------------------------------------------------------------------------------------- |
| 低            | 低            | 旧记忆无关 + 不值得写 = 句子边界的虚词。比如新句开头第一个"的"——和上文无关，本身也不重要 |
| 低            | 高            | 旧记忆有关 + 不值得写 = 帮助理解但不值得记忆。比如定语从句里的"which"                      |
| 高            | 低            | 旧记忆无关 + 值得写 = 新话题的第一个实词。重置旧记忆，写入全新内容                         |
| 高            | 高            | 旧记忆有关 + 值得写 = 核心内容词。在已有理解上叠加新信息                                   |

GRU 给了 RNN 两个新能力：

1. **选择性遗忘**（重置门）：新输入和旧记忆无关？把旧记忆对应部分清零。
2. **选择性保留**（更新门）：新输入不值得写进记忆？留着旧记忆不变。

---

**四个公式：**

**第一道门：重置门 r_t**

r_t = sigmoid(W_xr · x_t + W_hr · h_{t-1} + b_r)

r_t 每个元素在 0~1。r_t→0：旧记忆这位在算候选时忽略；r_t→1：正常参与。

输入新句子第一个词时，旧记忆和这个词完全没关系——r_t 应该接近 0，让模型从零开始。

**第二道门：更新门 z_t**

z_t = sigmoid(W_xz · x_t + W_hz · h_{t-1} + b_z)

决定旧记忆和新候选怎么混合。z_t→1：用新的；z_t→0：保留旧的。

读到"的""了"时，z_t 应该接近 0——"这个信息不重要，我不更新"。

**候选记忆 h_tilde**

h_tilde_t = tanh(W_xh · x_t + W_hh · (r_t ⊙ h_{t-1}) + b_h)

⊙ 是逐元素乘法。和 RNN 的唯一区别：h_{t-1} 前面乘了 r_t。
r_t 全是 0 → 纯靠当前输入，完全不管过去。
r_t 全是 1 → 和原始 RNN 一模一样。

**最终记忆 h_t（GRU 的灵魂）**

h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h_tilde_t

线性插值——旧记忆和新候选各取一部分。

z_t = 0 → h_t = h_{t-1}。**旧记忆原封不动复制过来。** 梯度穿过这一项乘 1，不衰减。这就是 ResNet 的恒等映射——GRU 的梯度高速公路。

z_t = 1 → h_t = h_tilde_t。完全覆盖。

---

**具体数字例子：**

hidden_size=2，h_{t-1}=[0.8, -0.6]，x_t 对应虚词"的"：

重置门:
r_t = sigmoid([-1.2, 0.3]) = [0.23, 0.57]
→ 第一位 0.23：旧记忆第一位基本忽略
→ 第二位 0.57：第二位参与一半

更新门:
z_t = sigmoid([-2.5, -2.1]) = [0.08, 0.11]
→ 两位都接近 0：这次更新不重要，旧记忆应该保留

候选记忆 h_tilde = [0.15, -0.08]

最终记忆:
h_t[0] = (1-0.08)×0.8 + 0.08×0.15 = 0.736+0.012 = 0.748  几乎不变
h_t[1] = (1-0.11)×(-0.6)+0.11×(-0.08) = -0.534-0.009 = -0.543 几乎不变

旧记忆几乎原封不动。虚词被更新门关在外面。

---

**为什么梯度不消失：**

∂h_t/∂h_{t-1} = (1 - z_t) + ...（含 z_t 和 tanh 导数的项）

关键在 (1 - z_t)。如果 z_t 接近 0，这一项接近 1。梯度乘以接近 1 的东西，信号不衰减。

从 h_100 传到 h_1，路径上有大量 z_t≈0 的时间步，梯度近似乘一串 1——信息直接穿过。

这不是更复杂的激活函数，不是更好的初始化。是**结构上给了一条可以完全跳过更新的路径**。

---

**GRU vs RNN：**


|            | RNN       | GRU                    |
| ------------ | ----------- | ------------------------ |
| 更新方式   | 强制刷新  | 可跳过（更新门 z_t）   |
| 旧记忆利用 | 直接全用  | 选择性用（重置门 r_t） |
| 梯度流动   | W_hh 连乘 | 有乘 1 的路径          |
| 长序列     | 学不到    | 学得到                 |
| 参数量     | 2组W      | 3组W（多两道门的W）    |

---

**GRU 到 LSTM 的距离：**

GRU 四个公式搞懂了，LSTM 就多一件事：把"记忆"拆成两个——隐状态 h（对外输出）和细胞状态 c（对内记忆）。GRU 里 h_t 同时干了这两件事，LSTM 分开了。

其他完全一样：门用 sigmoid，候选用 tanh，逐元素乘法控制信息流。LSTM 多一个门（输出门），多一个变量（c），没有新概念。

> GRU 核心：两个门——重置门控遗忘，更新门控保留。更新门给了梯度一条不衰减的高速公路。LSTM 只是在这条路上多开了一条车道。

## LSTM 费曼教学深度笔记

> **背景：** 2026-06-19，前一天学完GRU（重置门、更新门），当天学LSTM。用费曼教学法进行了一轮深度问答，以下按重要程度排序记录。

---

### 🔴 问题一：为什么要有 c_t 和 h_t 两个流？

**原始困惑：** h_t 也能存信息、也能当工作台，GRU 不就这么干的吗？多一个 c_t 的优势在哪？

**答案核心：c_t 是被保护的存储器——它在时间上传播时，不经过非线性激活函数。**

```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
```

c_{t-1} 到 c_t，经过的操作只有**逐元素乘法**。没有 tanh，没有矩阵乘法。如果遗忘门 f_t 接近全 1，c_{t-1} 几乎原封不动就到了 c_t。

而 h_t 每次都要被塞进下一轮：和 x_t 拼起来、乘 W、过非线性。一个存了 100 步的信息，经过 100 次非线性变换，早就面目全非。

**类比修正过程：**

- ❌ 我最初类比：c_t = 笔记本（摘要），h_t = 嘴里说的（不准确——c_t 不压缩，它累加）
- ✅ 修正后：c_t = 不断被编辑的共享文档（删掉某行、加新段落、保留旧段落），h_t = 当前理解（从文档中挑出与当前情节相关的，不相关的就不提）
- ❌ 我另一个错误：认为"c_t 的参考价值永远比 h_t 高"——如果任务是预测下一个词，h_t 已经做了筛选，比 c_t 更有效。分工不同，不是谁比谁好。

**关键公式再看一遍：**

```
h_t = o_t ⊙ tanh(c_t)
```

c_t 先被 tanh 压到 [-1,1]（数值稳定需要），再被输出门筛选。每一步 h_t 都是"加工过的"。c_t 不用干活——它只负责存在那里。

> **类比定稿：** c_t 是银行卡真实余额（精确数字），h_t 是 App 上显示的进度条（被压缩过的可视化）。报税你得用真实数字（c_t），快速浏览余额用进度条就够了（h_t）。但你不能说余额永远比进度条好——快速扫一眼的时候，进度条比完整数字好用。

---

### 🔴 问题二：c̃_t 是不是"先不管有没有用，先记下来再说"？

**答案：对，就是这个意思。**

c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c) 是**候选新记忆**。这步不管 x_t 有没有用，先全算出来。

然后**输入门 i_t 当裁判**：

- i_t 接近 0：这份候选的这个维度，不收
- i_t 接近 1：收，全收

完整逻辑链：

1. 「这里有份新信息候选」→ c̃_t（草稿）
2. 「收不收？收多少？」→ i_t（审稿）
3. 「执行」→ i_t ⊙ c̃_t 加到 c_t 上（落笔）

> 先草稿、再审稿、再落笔。c̃_t 就是草稿。

---

### 🔴 问题三：为什么不能去掉输出门 o_t？把 c_t 全暴露给 h_t 不是更全面吗？

**原始困惑：** c_t 信息更多，去掉 o_t 直接暴露不是更好吗？

**答案：「更全的信息」≠「更好的信息」。**

**原因一：c_t 是杂货铺，不是陈列柜。**
c_t 存了所有东西——张三湖南人、李四四川人、上一章的伏笔……什么都堆在里面。但当前步你可能只需要「这个人能吃辣」来预测下一个词。把整个杂货铺倒给下一层——噪声远多于信号。o_t 的作用是：**在当前上下文的指导下，从杂货铺里拣出有用的。**

**原因二：c_t 没有范围限制。**
c_t 通过加法更新，理论上可以增长到任意大。如果直接当 h_t 用，下一步就要把它和 x_t 拼起来乘 W——没有边界的值塞进矩阵乘法，数值会爆炸。tanh 把 c_t 压到 [-1,1] 是数值稳定的硬需求。

**原因三：o_t 是根据上下文动态筛选的。**
对比一个去掉 o_t 的消融实验：h_t = tanh(c_t)。这意味着不管当前输入是什么，细胞状态的每一维都以同样的权重暴露。这就和「不管这篇论文讲什么，所有章节都放一样多的篇幅」一样蠢。o_t 看着 h_{t-1} 和 x_t 做决策——当前需要什么，它开什么。

> **核心洞察：** LSTM 比 GRU 表达力强的一个关键原因就是输出门——c_t 存了什么，和 h_t 暴露了什么，是**解耦**的。GRU 没有这个——存什么就暴露什么。

---

### 🟡 门对应关系：LSTM vs GRU


| 功能                       | GRU                              | LSTM                                         |
| ---------------------------- | ---------------------------------- | ---------------------------------------------- |
| 控制旧记忆保留/新信息写入  | **更新门 z_t**（一个门搞定）     | **遗忘门 f_t + 输入门 i_t**（拆成两个）      |
| 计算候选时控制用多少旧状态 | **重置门 r_t**                   | **没有**（LSTM 的 c̃_t 直接用完整 h_{t-1}） |
| 控制状态对外暴露多少       | **没有**（h_t 直接就是最终状态） | **输出门 o_t**                               |

- 为什么 LSTM 不需要重置门？因为 h_{t-1} 已经经过输出门过滤，不重要的信息已被屏蔽；且遗忘门可以先把 c_{t-1} 里的东西清掉。
- 为什么 GRU 不需要输出门？GRU 的 h_t 存什么就暴露什么，没有解耦的需求。

---

### 🟡 思维惯性问题：一堂元认知课

**自测题（原题）：** 如果 f_t 全固定为 1，i_t 全固定为 0，o_t 全固定为 1，这个 LSTM 会退化成什么？

正确答案：**死模型。** c_t = c_{t-1} = ... = c_0（冻结），h_t = tanh(c_0)（常数）。比普通 RNN 还惨——RNN 至少 x_t 还能进去。

**我犯的错和追溯：**


| 步骤     | 我当时的回答                       | 实际错在哪                                              |
| ---------- | ------------------------------------ | --------------------------------------------------------- |
| 第一反应 | "不能接受新知识但能精炼旧知识"     | 默认给问题加了"模型先正常学了一阵子"的前提              |
| 追问后   | "h_t = tanh(c_T)，每次对 c_T 提炼" | 把 tanh（固定压缩函数）当成了"精炼"（有意义的信息重组） |
| 追问后   | "o_t 还在学习啊"                   | 忽略了原题明确写了 o_t = 1 的条件                       |

**两个错误，同一根源：**

1. **思维惯性**：自动给问题加了一个题目没说的前提——"模型正常训练到一半然后被改参数"
2. **粗心**：大脑一旦认定了"模型还活着"的框架，所有和框架矛盾的信息（o_t=1）被自动降权

> **这不是智商问题，这是确认偏误——一旦你有了一个叙事，矛盾证据就被大脑调低音量。** 每个人都是这样。能追溯自己为什么走偏，比答对一道题重要。

**命名不等于理解的又一次验证：**
tanh = "激活函数" → 我脑补它在"加工信息"。但 tanh 只是一个有界的恒等映射——逐元素把数字压到 [-1,1]。没有跨维度交互，没有学习，没有重组。"激活函数"这个名字骗了我。

---

### 🟢 LSTM 核心教学内容总结

**LSTM 解决了什么？** RNN 的梯度经过 seq_len 次 W_hh 连乘，指数衰减/爆炸。LSTM 修了一条高速公路（细胞状态 c_t），信息可以在上面几乎不变地跑几百步。

**三步正向传播：**

```
1. 算四个东西：
   f_t = σ(W_f[h_{t-1}, x_t])    ← 遗忘门：哪些旧信息扔掉？
   i_t = σ(W_i[h_{t-1}, x_t])    ← 输入门：哪些新信息收进来？
   c̃_t = tanh(W_c[h_{t-1}, x_t]) ← 候选新记忆（草稿）
   o_t = σ(W_o[h_{t-1}, x_t])    ← 输出门：暴露什么？

2. 更新细胞状态（高速公路）：
   c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t

3. 更新隐藏状态（工作台 / 对外输出）：
   h_t = o_t ⊙ tanh(c_t)
```

**LSTM 比 RNN 强的根本原因：选择权。**

- RNN：每步被迫更新，旧信息被冲走，没有商量余地
- LSTM：遗忘门可以选择某些维度不动（梯度畅通），输入门可以选择性接收新信息
- 如果把三道门全焊死 → 比 RNN 还惨的铁疙瘩

**LSTM 比 GRU 复杂在哪？**

- 多一道流（c_t 和 h_t 解耦）
- 多一个门（输出门 o_t）
- 表达能力更强，但参数量更大、更慢
- 实际效果经常和 GRU 差不多（GRU 性价比高）

#### 9.4.1 词汇表构建

**词表是什么？** 文本和模型之间的唯一接口。神经网络不吃字符串只吃数字，词表就是**词→数字**的翻译器。

```
原始文本 → 分词 → 查词表 → 索引序列 → nn.Embedding → 向量序列 → LSTM
"this movie is bad"
   ↓ 分词
["this", "movie", "is", "bad"]
   ↓ 查 word2idx
[2, 3, 4, 5]
   ↓ nn.Embedding(25000, 128)
[[0.12, -0.03, ...],    ← "this" 的128维向量
 [0.45, 0.21, ...],     ← "movie"
 [0.08, -0.15, ...],    ← "is"
 [-0.32, 0.11, ...]]    ← "bad"   shape: (4, 128)
   ↓ nn.LSTM
最终分类
```

**特殊标记：**

- `<PAD>` = 0：填充符，短句子补到统一长度
- `<UNK>` = 1：未知词，测试时遇到词表没见过的词全映射到这个

**自己手写实现（20行，推荐）：**

```python
from collections import Counter

def build_vocab(texts, max_size=25000):
    counter = Counter()
    for text in texts:
        counter.update(text.split())
  
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in counter.most_common(max_size - 2):
        word2idx[word] = len(word2idx)
  
    idx2word = {v: k for k, v in word2idx.items()}
    return word2idx, idx2word
```

**PyTorch 没有现成的词表 API。** 备选方案：

- `torchtext.vocab.build_vocab_from_iterator`：能用但版本混乱，API 频繁变动
- HuggingFace tokenizers：工业级但太重，学 LSTM 项目杀鸡用牛刀
- 自己手写：最可控，出问题不看文档就知道哪里错

---

#### 9.4.2 词向量表（Embedding）

**本质：一张可学习的大表。**

```python
nn.Embedding(vocab_size=25000, embedding_dim=128)
```

这一行创建了一个 `(25000, 128)` 的矩阵：

```
第0行 (<PAD>):  [全0]           ← 填充符，始终全0（padding_idx=0）
第1行 (<UNK>):  [随机初始化]     ← 未知词
第2行 (this):   [随机初始化]     ← 每个词占一行
第3行 (movie):  [随机初始化]
第4行 (is):     [随机初始化]
第5行 (bad):    [随机初始化]
...
第24999行:      [随机初始化]
```

**查表操作：** 输入 `[2, 3, 4, 5]` → 取第 2、3、4、5 行 → 输出 `(4, 128)` 的向量序列。

**初始值：全是随机数。** 和 `nn.Linear` 的权重一样，PyTorch 用正态分布 N(0,1) 初始化，训练过程中梯度更新。

**语义是学出来的，不是人告诉它的：**

- 训练前：bad 向量 ≈ good 向量（都是随机的），模型分不清好坏
- 训练中：模型发现把 bad 和 good 拉远能降低 loss → 梯度更新9
- 训练后：bad 附近聚集了 terrible、awful、worst；good 附近聚集了 great、excellent、wonderful

**预训练词向量（可选，不是必须）：**

```python
embedding.weight.data.copy_(torch.from_numpy(glove_matrix))  # 用GloVe覆盖随机值
```

好处是训练快、小数据集效果好；坏处是多外部依赖、词表要对齐。IMDB 2.5万条数据够大，随机初始化 + 跟着训就行。

#### 9.4.3 多层LSTM的执行顺序

**问题背景：** 三层LSTM，第一层算出 h_1^(1) 之后，第二层能不能立刻拿来算 h_1^(2)？还是必须等第一层把所有时间步算完？

**数学上，可以立刻拿。** 第二层算 h_1^(2) 只需要 h_1^(1) 和 h_0^(2)，不需要 h_2^(1)。依赖图允许"算出一个传一个"。

**工程上，不这么做。** 实际框架（PyTorch、cuDNN）用的是**层优先**：层1算完整个序列 [h_1^(1), h_2^(1), ..., h_T^(1)]，然后层2一次性消费。为什么？

**原因不在数学，在硬件。**

GPU的设计哲学叫"单指令多数据"——它假设你要对一大批数做同一种运算。几千个核心，每一个都很简单，但合起来吞吐量巨大。为了喂饱它们，你得给大块数据。

LSTM每个时间步有两类计算：

- W·x_t（当前输入做矩阵乘法）——不同时间步之间**无依赖**。W是同一组参数，x_1、x_2、x_3 全已知。可以摞成一个大矩阵一次算完：W·[x_1, x_2, ..., x_T]。
- U·h_{t-1}（上一隐藏状态做矩阵乘法）——**有依赖**。算 h_t 之前必须有 h_{t-1}，这部分必须串行。

如果你用"算出一个传一个"的方式（时间优先），会发生两件事：

1. **内核启动开销吞噬计算时间。** 每次让GPU启动一个计算任务，调度指令本身花的时间，比你让几千个核算那个小矩阵乘法还要长。时间优先意味着启动几十上百次微型任务；层优先只启动几次大型任务。
2. **大部分核心在发呆。** 给你5000个核心，你扔一个3×4的矩阵乘法过去——可能只有12个核心在干活，剩下4988个在围观。这不是编程错误，是你给的数据块太小了，指令广播出去之后大部分核心没分到活。

所以层优先不是"更快"，而是"不这么做就是在浪费硅片"。把同一层所有时间步的 W·x 摞成一个大矩阵，喂一次，所有核心同时轰鸣，算完。然后处理必须串行的 U·h 部分。产出完整序列交给下一层。

**补充细节：** 如果序列特别长（比如一万个时间步），W·[x_1, ..., x_{10000}] 太大塞不进显存，框架会切成几块。每块内部还是层优先，U·h 的状态跨块传递。这是显存的限制，不是策略的选择。

**一句话：** 教科书上的LSTM公式告诉你"每个时间步做什么"，不告诉你"所有时间步怎么调度"。调度是硬件决定的——GPU要大口吃饭，你得把数据摞成一整盘端上去，不能一粒一粒往嘴里塞。

## 10.三个实战项目

### 10.1 CIFAR-10图像分类

> **这个项目教会我什么？** 从裸奔 CNN 的 37% 到最终 84.6%，最大的收益来自**BN**和**数据增强**。数据质量比模型深度更重要——先把进入网络的数据洗干净、喂多样，比堆层数高效得多。另一个收获是"独立 Debug 的肌肉记忆"：从报错→读报错→定位行→理解原因→修复，这个循环走了几十遍之后，不再害怕空白文件了。

#### 实战中出现的问题和解决方案

---

##### 一、数据处理（最高发问题区）

> **核心认知：** 数据处理和模型训练是两个独立阶段。Dataset 是"仓库"，Transform 是"加工工具"，DataLoader 是"物流"。Transform 在 `__getitem__` 被调用时才执行，不是在创建 Dataset 时执行。

**问题 1：`test_data[0]` 返回的是元组，不是张量**

```python
# ❌ 错误
img = ToPILImage(test_data[0])  # test_data[0] = (tensor, label)

# ✅ 正确
img_tensor, label = test_data[0]
img = ToPILImage()(img_tensor)
```

- **原因：** `CIFAR10.__getitem__` 返回 `(image, target)` 元组
- **教训：** 拿到数据集先 `print(dataset[0])` 看返回值的结构

**问题 2：`ToPILImage` 是类，不是函数**

```python
# ❌ 错误
img = transforms.ToPILImage(img_tensor)

# ✅ 正确
img = transforms.ToPILImage()(img_tensor)
#                      ↑↑ 第一对括号：实例化
#                         ↑↑           第二对括号：调用实例
```

**问题 3：`nn.Sequential` 不接受列表**

```python
# ❌ 错误 — 方括号把层变成了 Python list
self.model = nn.Sequential([...])

# ✅ 正确 — 把层作为位置参数传入
self.model = nn.Sequential(...)
```

- **错误信息：** `TypeError: list is not a Module subclass`
- **原因：** Sequential 期望的是 `nn.Module` 对象，不是 list 容器

**问题 4：`BatchNorm2d` vs `BatchNorm1d` 用错**

```python
# 数据流：Conv(4D) → Flatten(2D) → Linear(2D)
#                              ↑ 数据已经变成 (batch, features)

# ❌ 错误 — Flatten/Linear 之后还在用 BatchNorm2d
nn.Linear(4096, 256),
nn.BatchNorm2d(256),  # 期望 4D，拿到 2D，报错

# ✅ 正确 — 全连接层后面用 BatchNorm1d
nn.Linear(4096, 256),
nn.BatchNorm1d(256),  # 处理 2D 数据
```


| 数据维度                     | BN 类型       |
| ------------------------------ | --------------- |
| 4D`(N, C, H, W)` 卷积输出    | `BatchNorm2d` |
| 2D`(N, features)` 全连接输出 | `BatchNorm1d` |

**问题 5：`ToTensor()` 废弃警告**

```python
# ❌ 旧 API（已废弃）
transforms.ToTensor()

# ✅ 新 API（拆成两步）
v2.ToImage()                           # PIL → TV Image
v2.ToDtype(torch.float32, scale=True)  # uint8→float32, [0,255]→[0,1]
```

- 拆成两步是为了让整个 pipeline 统一成 Tensor 生态

**问题 6：测试集的 transform 不一致**

```python
# 训练和测试的 ToTensor / Normalize 必须用完全相同的参数
# 特别是 Normalize 的 mean/std 必须一样
```

---

##### 二、数据增强（最容易踩坑）

**核心原则：** 数据增强 = 模拟测试时可能遇到的变体。**只增强那些测试集里会出现的变化。**


| 增强                             | CIFAR-10 能用？ | 原因                                     |
| ---------------------------------- | :---------------: | ------------------------------------------ |
| `RandomHorizontalFlip`           |       ✅       | 猫朝左朝右都是猫                         |
| `RandomCrop(size=32, padding=4)` |       ✅       | 目标不一定在画面正中间                   |
| `RandomRotation(degrees=10)`     |       ✅       | 拍照角度微偏                             |
| `RandomVerticalFlip`             |       ❌       | 没人倒着拍 CIFAR，颠倒的图测试集不会出现 |

**错误案例：加了 `RandomVerticalFlip`，正确率反而下降**

- 原因：教模型学"倒过来的猫也是猫"，但测试集没倒过来的图——白学了，还挤占了正常特征的学习空间

**错误案例：`RandomCrop(size=16, 16)` 裁成 16×16**

- 原因：模型 `Linear(in_features=4096)` 是按 32×32 输入算的。裁成 16×16 后 `Flatten` 出来维度对不上，直接报错
- 正确做法：`RandomCrop(size=32, padding=4)` — 先填到 40×40 再裁回 32×32

**数据增强后 loss 高的正确理解：**

```
无增强：同一批题反复做 → loss 降到 0.06（背熟了答案）
                                        考试换了几道 → 37%（没学懂）
有增强：每次做的题都在变 → loss 停在 0.6（还在学规律）
                                        考试换了几道 → 74%（真学懂了）
```

> **数据增强后的 loss 不能与无增强时的 loss 直接对比。** 增强把训练难度抬高了一个数量级，loss 降得慢不是坏事，说明模型没在偷懒死记硬背。

**训练集变难 ≠ 测试集变难：**

```
训练集：RandomFlip + RandomCrop + RandomRotation（变难了）
测试集：只做 ToTensor + Normalize（从来没变过）
```

**Normalize 的 mean/std 从哪里来：**

- `(0.4914, 0.4822, 0.4465)` — CIFAR-10 5万张训练集三个通道各自的均值
- `(0.2023, 0.1994, 0.2010)` — 三个通道各自的标准差
- 这些值是从训练集统计出来的，社区早有标准值，不用自己算
- **训练和测试必须用同一套 mean/std**

---

##### 三、模型搭建问题

**问题 7：模型 Flatten 后维度计算**

```
32×32 → MaxPool → 16×16 → MaxPool → 8×8 → Flatten → 8×8×64 = 4096
加第三块后:
32×32 → 16×16 → 8×8 → 4×4 → Flatten → 4×4×128 = 2048
```

每次加卷积块/改 MaxPool 后，必须重算 `Linear` 的 `in_features`。

**BN 层的正确位置：**

```python
# 标准顺序（推荐）
nn.Conv2d(...) → nn.BatchNorm2d(...) → nn.ReLU() → nn.MaxPool2d(...)
```

**为什么是 Conv → BN → ReLU 而不是 Conv → ReLU → BN？**

1. **Conv 的输出是对称分布：** 卷积核的权重初始化后大致关于 0 对称（均值≈0）。BN 在这样的分布上做归一化效果最好——直接拉到均值 0、方差 1，干净利落。
2. **如果 BN 放在 ReLU 后面：** ReLU 把所有负值变成 0，输出的分布已经不对称了（均值被抬到 > 0），BN 再去强行归一化一个"被削了一半"的分布——效果打折。
3. **BN 的输出 ≈ N(0,1)，恰好是 ReLU 最舒服的输入区：** ReLU 的零点附近梯度变化最大，BN 把数据集中在这个区域，梯度信号最清晰。

```
Conv 输出：[-2, -1, 0, 1, 2]    对称分布
   ↓ BN
BN 输出：  [-1.2, -0.6, 0, 0.6, 1.2]   ≈ N(0,1)
   ↓ ReLU
ReLU 输出：[0, 0, 0, 0.6, 1.2]         有效激活
```

如果反过来，ReLU 先上：

```
Conv 输出：[-2, -1, 0, 1, 2]
   ↓ ReLU
ReLU 输出：[0, 0, 0, 1, 2]              负值全死了，分布已偏
   ↓ BN
BN 强拉：  [?, ?, ?, ?, ?]              BN 在扭曲的分布上工作，效果差
```

> **一句话：** BN 在 ReLU 前面 = 先给数据"摆好姿势"，再让 ReLU 做选择。BN 在 ReLU 后面 = ReLU 先把数据削掉一半，BN 在残局上强行干活。

**BN 为什么提升这么大（37% → 65%）：**

- 没有 BN：各层输出数值范围失控，全连接层 100 万参数收到的是"脏数据"，几乎在随机游走
- 有 BN：每层输出被拉到均值 0、方差 1，梯度方向对准了
- 真正大头是**卷积层的 BN**，它把进入 Flatten 的数据洗干净，FC 层才能正常工作。FC 后面那个 BN 只是锦上添花（+3%）

---

##### 四、模型调试思维

**判断过拟合 vs 欠拟合：**


| 状态             |         训练 loss         |     测试 acc     | 对策                           |
| ------------------ | :-------------------------: | :----------------: | -------------------------------- |
| 欠拟合（没学够） |           > 1.0           |   < 60% 还在涨   | 加 epoch / 加深度              |
| 学习中           |          0.4~1.0          |   60~80% 在涨   | 继续训练                       |
| 快到极限         |          0.2~0.4          |   80~90% 平了   | 加容量 / 换优化器              |
| 过拟合（背答案） | 训练 loss 远低于测试 loss | 测试涨不动甚至掉 | 加增强 / 加 Dropout / 减小模型 |

> **注意：** 有数据增强的情况下，训练 loss 天然偏高。不能用无增强时的标准（0.01）去要求有增强的训练 loss。

**调试决策链：**

```
loss 高 + acc 低 → 欠拟合 → 加深网络 / 加 epochs
loss 极低 + acc 不涨 → 过拟合 → 加增强 / 加 Dropout
loss 降得慢但 acc 在涨 → 正常学习中！继续
后面震荡不涨 → 接近极限 → 降学习率 / 换优化器
```

**epoch = 20，acc 只提升了 6%：**

- 先看 TensorBoard 里的 train_loss。loss=0.06（极低）说明**过拟合**——模型在背答案
- 解决方案是加数据增强，不是加深网络

---

##### 五、计算速度优化

**问题 8：GPU训练——数据搬运的坑**

```python
# ❌ 到处 if torch.cuda.is_available()，没 else 直接崩
if torch.cuda.is_available():
    mymodel = model().cuda()

# ✅ 用 device 统一管理
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mymodel = model().to(device)
feature = feature.to(device, non_blocking=True)
```

**各优化参数详解：**

**1. `num_workers` — 多进程数据加载**

```
num_workers=0：  主进程                          num_workers=2：  主进程
                  ↓                                                │
                 读磁盘 ──→ 增强 ──→ 喂GPU                          │
                  ↓                   ↓                  ┌──────┐  ┌──────┐
                  GPU空闲等下一批      GPU算              子进程1   子进程2
                                                          │        │
                                                          读磁盘+增强  读磁盘+增强
                                                          │        │
                                                          └──→ 排队等GPU ←──┘
```

- `num_workers=0`：主进程一个人干所有活——读文件 → 数据增强 → 喂 GPU。GPU 算完一批只能在原地等 CPU 处理下一批。**CPU 在忙时 GPU 在等，GPU 在算时 CPU 在等。**
- `num_workers>0`：子进程在后台并行读数据+做增强，主进程只负责喂 GPU。**GPU 算完立刻有下一批准备好的数据**，不用等。

**选择建议：** 一般 2-4 个就够了，不是越多越好。worker 太多→进程间通信开销超过收益。数据增强越重，worker 收益越大。

**2. `pin_memory=True` — 锁页内存**

```
普通内存（pageable）：                锁页内存（pinned）：
CPU 数据 ──→ 先拷到 staging buffer    CPU 数据 ──→ GPU 直接 DMA 读取
                └──→ GPU DMA 读取          ↑ 一次传输，快 ~1.5-2×
                ↑ 两次拷贝
```

- 操作系统默认的内存是"可分页的"——随时可能被换到磁盘。GPU 的 DMA 引擎不能直接从可分页内存读数据，需要先拷贝到一个固定的中间缓冲区。
- `pin_memory=True` 分配的是**锁页内存**——操作系统保证这块内存不会被换页。GPU 可以**直接 DMA 访问**，省掉中间拷贝这一步。
- 代价：锁页内存不会被换出，占用物理 RAM。但 CIFAR-10 的 batch 很小（几 MB），几乎无感。

**3. `non_blocking=True` — 异步传输**

```
同步传输（阻塞）：                      异步传输（non_blocking）：
    CPU: |传输 batch1|→闲着→|传输 batch2|     CPU: |发传输batch1|→预处理batch2→|发传输batch2|
    GPU: |闲着|→算batch1→|闲着|→算batch2      GPU: |收batch1|→算 batch1|收batch2|→算 batch2|
         时间→                                    时间→  传输和计算重叠 = 更快
```

- `feature.cuda()` 是**同步阻塞**的——CPU 发起传输后必须等传输完成才往下走。
- `feature.to(device, non_blocking=True)` 是**异步**的——CPU 发起传输立刻返回，GPU 在后台收数据。CPU 可以立刻去干下一件事（比如加载下一个 batch 的数据）。
- 必须和 `pin_memory=True` 搭配才有效——因为异步 DMA 只能从锁页内存读取。

**4. 综合对比：旧写法 vs 新写法**

```python
# 旧写法
if torch.cuda.is_available():
    feature = feature.cuda()     # 阻塞传输
    label = label.cuda()

# 新写法
feature = feature.to(device, non_blocking=True)  # 异步传输
label = label.to(device, non_blocking=True)
```

```
旧写法每一批的时间线：
  [CPU读数据+增强] → [CPU等GPU收数据] → [GPU算] → 重复
                              ↑ 这中间CPU纯闲着

新写法每一批的时间线：
  worker进程读+增强 ──→ 数据就绪 ──→ 异步传 ──→ GPU算 ──→ 异步传下一批
  主进程发传输 ──→ 不等待，立刻处理下一批 ──→ ...
                              ↑ CPU和GPU同时在工作
```

**实测效果：** 每个 epoch 快 3-5 秒。数据增强越重，收益越大。

**Windows 特殊问题：`num_workers > 0` 必须加 `if __name__ == '__main__'`**

- Windows 用 spawn 创建子进程（不是 Linux 的 fork）
- 子进程会重新导入 `main.py`，不保护会导致无限套娃
- 所有 import 和 class 定义放外面，训练代码全部包进 `if __name__ == '__main__':`

**优化器选择：**


| 优化器                               | 特点                        | 什么时候用         |
| -------------------------------------- | ----------------------------- | -------------------- |
| `SGD(lr=0.01)`                       | 基础款                      | 入门、需要完全控制 |
| `SGD(lr=0.01, momentum=0.9)`         | 加惯性，减少震荡            | 比纯 SGD 稳        |
| `AdamW(lr=0.001, weight_decay=1e-4)` | 自适应学习率 + 解耦权重衰减 | **推荐默认选择**   |

---

##### 六、最佳模型保存

```python
best_acc = 0
for epoch in range(epochs):
    # 训练 ...
    # 测试 ...

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(mymodel.state_dict(), 'best_model.pth')
```

- 不是最后一个 epoch 的模型最好，而是测试准确率最高的那个
- 保存 `state_dict()`（只存参数），不存整个模型（省空间）

---

##### 七、完整训练流程总结

```
1. 数据准备
   ├── Transform 定义（训练：增强+归一化 / 测试：只归一化）
   ├── Dataset 创建（绑定 transform）
   └── DataLoader 打包（batch_size, shuffle, num_workers, pin_memory）

2. 模型搭建
   ├── Conv块 ×N（Conv → BN → ReLU → MaxPool）
   ├── Flatten → Linear → BN1d → ReLU → Linear
   └── 手动验证每一层的 shape

3. 训练配置
   ├── device 统一管理
   ├── CrossEntropyLoss（内置 Softmax）
   └── AdamW 优化器

4. 训练循环
   ├── model.train() → 训练 → model.eval() → 测试
   ├── 每 epoch 记录 train_loss / test_loss / test_acc
   └── 保存最佳模型

5. TensorBoard 观察
   ├── 看 train_loss 和 test_loss 是否分叉（分叉=过拟合）
   └── 看 test_acc 是否还在涨（平了=到极限）
```

---

##### 本项目的最终成果


| 阶段          |  正确率  | 关键改动                                 |
| --------------- | :---------: | ------------------------------------------ |
| 裸奔          |    37%    | Conv×2 + FC，无 BN，无增强              |
| + BN          |    65%    | 卷积层加 BatchNorm                       |
| + Normalize   |    68%    | 像素归一化                               |
| + 数据增强    |    74%    | RandomFlip + RandomCrop + RandomRotation |
| + 第三卷积块  |    76%    | Conv(64→128)                            |
| + AdamW       | **77.5%** | SGD → AdamW                             |
| + epochs = 30 | **84.6%** | epoch = 15 -> 30                         |

**最终配置：** 3 卷积块 + BatchNorm + 数据增强 + AdamW + 15 epochs = **84.6%**

```text
第1份 正确率：0.8480 (848/1000)
第2份 正确率：0.8360 (836/1000)
第3份 正确率：0.8520 (852/1000)
第4份 正确率：0.8330 (833/1000)
第5份 正确率：0.8530 (853/1000)

均值：0.8444  标准差：0.0083
```

### 10.2 classify-leaves

#### 项目概述

树叶分类任务：输入 3×224×224 叶片俯视图，输出 176 个树种之一。自建 CNN + 训练全流程。

#### 一、数据导入

**问题：标签是字符串，PyTorch 不认**

`train.csv` 的 label 列是 `maclura_pomifera` 这种字符串，`CrossEntropyLoss` 只接受整数类索引（`torch.long`）。

解决：构建 `str → int` 映射字典，`__getitem__` 中查表转换：

```python
idx_class_list = {'abies_concolor': 0, ..., 'zelkova_serrata': 175}
label = self.idx_class_list[self.dataframe.iloc[idx, 1]]  # str → int
```

推理时反向映射 `int → str`：

```python
idx_to_name = {v: k for k, v in idx_class_list.items()}
```

> **拓展**：也可以直接用 `sklearn.preprocessing.LabelEncoder`，`fit_transform` 编码，`inverse_transform` 解码，代码更简洁。

**问题：变量名覆盖**

```python
# 错误示范
test_train_csv = train_test_split(...)  # DataFrame
test_train_csv = imgdataset(test_train_csv, ...)  # 覆盖成了 Dataset
```

后续想拿原始 DataFrame 时已经丢了。Dataset 类型变量应取明确名字：

```python
train_train_csv, test_train_csv = train_test_split(...)
train_train_dataset = imgdataset(train_train_csv, ...)
test_train_dataset = imgdataset(test_train_csv, ...)
```

#### 二、网络设计

**问题：卷积输出尺寸算不对**

网络 Flatten 后期望 `[B, 2048]`，实际 `[B, 512]`，`Linear` 层直接报错。

核心公式：

$$
H_{out} = \left\lfloor\frac{H_{in} + 2p - k}{s}\right\rfloor + 1

$$

两个常见陷阱：


| 陷阱                                                                   | 说明 |
| ------------------------------------------------------------------------ | ------ |
| `MaxPool2d(k=3)` 没写 stride → 默认 s=3，不是 s=1                     |      |
| `MaxPool2d(k=3)` 没写 padding → 默认 p=0                              |      |
| `Conv2d(p=2, s=1)` → padding 比 kernel 多，特征图不减反增（224→228） |      |

> **拓展**：标准做法是卷积结束后加 `nn.AdaptiveAvgPool2d((1,1))`，不管前面输出多大都能稳定到指定维度，彻底不用手算尺寸。几乎所有现代 CNN（ResNet、EfficientNet）都用这招。

**最终架构**（5个卷积块 + 1个全连接分类头）：

```
输入 [3, 224, 224]

Block 1: Conv(3→32,k3,s1,p1) + BN + ReLU          → [32, 224, 224]
Block 2: Conv(32→64,k3,s2,p1) + BN + ReLU + Pool   → [64, 56, 56]
Block 3: Conv(64→128,k3,s2,p1) + BN + ReLU + Pool  → [128, 14, 14]
Block 4: Conv(128→256,k3,s2,p1) + BN + ReLU + Pool → [256, 4, 4]
Block 5: Conv(256→512,k3,s2,p1) + BN + ReLU + Pool → [512, 1, 1]

Flatten → [512]
Dropout(0.5) → Linear(512, 176)
```

设计原则：Conv(s=2) 负责通道翻倍 + 空间减半，MaxPool(s=2) 负责进一步降维。两者配合，空间路线：224 → 112 → 56 → 28 → 14 → 7 → 4 → 2 → 1。

**Dropout 放哪里？**


| 位置       | 加？ | 原因                                                            |
| ------------ | ------ | ----------------------------------------------------------------- |
| 卷积层间   | ❌   | BN 已有正则作用；特征图像素有空间关联，随机丢一个像素邻居还存着 |
| 全连接层前 | ✅   | 神经元没有空间结构，Dropout 防止共适应效果最强                  |

#### 三、训练阶段

**问题：过拟合**

训练 loss=0.014，测试 loss=0.637（45 倍差距），正确率 84% 不再涨。

解决——四件套一起上：


| 方法                | 原理                                      | 本项目取值             |
| --------------------- | ------------------------------------------- | ------------------------ |
| **Dropout**         | 训练时随机置零 50% 神经元，强迫学冗余特征 | `p=0.5`                |
| **Label Smoothing** | one-hot 目标变软标签，防止过度自信        | `label_smoothing=0.1`  |
| **weight_decay**    | L2 正则，限制权重大小                     | `1e-3`（比默认大10倍） |
| **Early Stopping**  | 测试 loss 连续 N 轮不降就停，保存最优权重 | `patience=5`           |

过拟合 vs 欠拟合判断：


|        | 训练 Loss  | 测试 Loss  | 趋势             |
| -------- | ------------ | ------------ | ------------------ |
| 欠拟合 | 高，持续降 | 高，持续降 | 还没学够，继续跑 |
| 正常   | 降         | 降         | 继续跑           |
| 过拟合 | 极低       | 升高       | 在背答案         |
| 震荡   | 降         | 剧烈波动   | lr 太大          |

**问题：测试 Loss 剧烈震荡**

```
Epoch 20: test_loss=1.61  acc=81.7%
Epoch 21: test_loss=2.51  acc=51.0%  ← 权重被一个大步更新"踢飞"
```

原因：学习率太大（0.001），训练后期梯度变小但步长不变，一个 batch 的波动就能踢飞权重。

解决：`lr` 从 `1e-3` 降到 `5e-4`。

> **拓展——AdamW vs 学习率调度器（ReduceLROnPlateau）**：
>
>
> |            | AdamW                              | ReduceLROnPlateau        |
> | ------------ | ------------------------------------ | -------------------------- |
> | 调节对象   | 每个参数独立的步长缩放（梯度层面） | 全局绝对步长（泛化层面） |
> | 触发条件   | 梯度一阶/二阶矩统计                | 测试 loss 不降           |
> | 解决的问题 | 不同参数收敛速度不同               | 后期步长太大导致震荡     |
> | 关系       | 管"怎么走"                         | 管"走多少"               |
> |            | **互补，不是重复**                 |                          |
>
> ```python
> scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
>     optimizer, mode='min', factor=0.5, patience=3
> )
> scheduler.step(test_avg_loss)  # 每个 epoch 测试完调用
> ```

**问题：GPU 跑一段闲一段**

根因：Dataset 每次 `read_image` 读磁盘，几千个小 JPG 随机读取，磁盘寻道 > GPU 算力。

解决：


| 手段                      | 作用                             |
| --------------------------- | ---------------------------------- |
| `num_workers=8`           | 8 个进程并行读图                 |
| `persistent_workers=True` | worker 跨 epoch 复用，省重建开销 |
| `prefetch_factor=4`       | 每个 worker 提前备 4 批          |
| `pin_memory=True`         | 锁页内存 → GPU DMA 直拷         |
| `non_blocking=True`       | CPU→GPU 传输和计算异步重叠      |
| `read_image` 替代 PIL     | libjpeg-turbo 比 PIL 快 2~3 倍   |

⚠️ Windows 上 `num_workers > 0` 必须加 `if __name__ == '__main__':` 保护，否则子进程 import 无限递归。

> **拓展**：终极方案是初始化时把图全读到内存（uint8 tensor），`__getitem__` 只做 transform。磁盘 I/O 降为 0，但需要 ~2.7GB 内存。

#### 四、代码调试

**1. `nn.Sequential` 参数列表缺逗号 → 语法错误**

```python
# ❌ 两个对象间没逗号
nn.Dropout(p=0.5)
nn.Linear(512, 176),

# ✅
nn.Dropout(p=0.5),
nn.Linear(512, 176),
```

`Sequential` 本质是普通函数调用，参数之间必须有逗号。

**2. `exit()` 调试残留** — 循环里调试用 `exit()` 忘删，训练只跑 Epoch 1 就退出。

**3. import 失败但源码无误** — `__pycache__` 缓存了旧 `.pyc`，删掉文件夹即解决。

#### 五、推理阶段

容易漏的必备设置：

```python
model.eval()                     # ① 关 Dropout + BN 用全局统计
with torch.no_grad():            # ② 不计算梯度，省显存
    for img, _ in test_dataloader:
        img = img.to(device)      # ③ 数据和模型同一设备
        output = model(img)       # ④ logits [B, 176]
        pred = output.argmax(1)   # ⑤ 最大值的索引 → 类编号
```

漏掉 ① Dropout 仍生效，每次结果不同。漏掉 ② 额外占显存。

#### 六、知识速查


| 概念                          | 一句话                                              |
| ------------------------------- | ----------------------------------------------------- |
| `state_dict()`                | 把模型所有可学习参数导出成字典                      |
| `torch.save(obj, path)`       | 字典存盘                                            |
| `model.load_state_dict(dict)` | 字典装回模型                                        |
| `CrossEntropyLoss`            | 内置 Softmax + NLLLoss，吃 logits 不吃 softmax 结果 |
| `label_smoothing`             | one-hot 变软标签，防过拟合                          |
| `pin_memory`                  | CPU 锁页内存，GPU 直接 DMA 拷贝                     |
| `non_blocking`                | `.to(device)` 不阻塞 CPU，传输和计算可并行          |
| `persistent_workers`          | worker 存活整个训练周期                             |
| `prefetch_factor`             | 每个 worker 提前取 N 批                             |
| `AdaptiveAvgPool2d((1,1))`    | 自适应池化，永远不用手算特征图尺寸                  |

#### 七、最终效果

**80%正确率，176分类**

### 10.3 手写ResNet

```ptyhon
import torch
import torch.nn as nn


class BasicBlock(nn.Module):


    def __init__(self, in_channle, out_channle, stride = 1, downsample = False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = in_channle, out_channels = out_channle, kernel_size = 3, stride = stride, padding = 1, bias = False)

        self.bn1 = nn.BatchNorm2d(num_features = out_channle)


        self.conv2 = nn.Conv2d(in_channels = out_channle, out_channels = out_channle, kernel_size = 3, stride = 1, padding = 1, bias = False)

        self.bn2 = nn.BatchNorm2d(num_features = out_channle)

        self.downsample = downsample

        self.identity_conv = nn.Conv2d(in_channels = in_channle, out_channels = out_channle, kernel_size = 1, stride = stride, padding = 0, bias = False)

        self.identity_bn = nn.BatchNorm2d(num_features = out_channle)

        self.relu = nn.ReLU()


    def forward(self, x):
        identity = x

        if self.downsample:
            identity = self.identity_conv(identity)
            identity = self.identity_bn(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out

class ResNet34(nn.Module):

    # 传入的shape得是3 * 224 * 224的
    def __init__(self, block, block_num, num_classes):
        super().__init__()
        self.in_channel = 64

        # 3 * 224 * 224
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = self.in_channel, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(num_features = self.in_channel)

        # 64 * 112 * 112
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        # 64 * 56 * 56
        self.layout1 = self.make_layer(block = block, in_channel = self.in_channel, out_channel = 128, block_num = block_num[0], downsample = True)

        # 128 * 28 * 28
        self.layout2 = self.make_layer(block = block, in_channel = 128, out_channel = 256, block_num = block_num[1], downsample = True)

        # 256 * 14 * 14
        self.layout3 = self.make_layer(block = block, in_channel = 256, out_channel = 512, block_num = block_num[2], downsample = True)

        # 512 * 7 * 7
        self.layout4 = self.make_layer(block = block, in_channel = 512, out_channel = 1024, block_num = block_num[3], downsample = True)

        # 1024 * 4 * 4
        self.avgpool = nn.AvgPool2d(kernel_size = 5, stride = 2, padding = 1)

        # 1024 * 1 * 1
        self.fc = nn.Linear(in_features = 1024, out_features = num_classes)


        self.relu = nn.ReLU()


    def forward(self, x):
      
        # 7 * 7卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layout1(x)
        x = self.layout2(x)
        x = self.layout3(x)
        x = self.layout4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
```

**使用Normalize时，数据集太大，无法计算该数据集mean和std时，可以使用相近数据集的mean和std，这里可以使用ImageNet的mean和std**

```python
train_transform = v2.Compose([
      v2.RandomHorizontalFlip(p=0.3),
      v2.RandomVerticalFlip(p=0.3),
      v2.RandomResizedCrop(size=(224, 224), antialias=True),
      v2.RandomRotation(degrees=30),
      v2.ToImage(),
      v2.ToDtype(dtype=torch.float32, scale=True),
      v2.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet标准值
                   std=[0.229, 0.224, 0.225])
])
```

### 10.4 迁移学习ResNet

#### 10.4.1 只微调预训练模型的最后一层（全连接分类层）

**数据集与ImageNet比较像的可以只修改分类头**

```python
model = resnet18(weights = 'DEFAULT')

# 修改全连接层，使得全连接层的输出与当前数据集类别数对应
# 新建的层默认 requires_grad = True
model.fc = nn.Linear(model.fc.in_features, n_class)
# 优化器只传入最后一层的参数
optimizer = optim.Adam(model.fc.parameters())
```

#### 10.4.2 微调预训练所有层

**数据集与ImageNet不怎么像的可以微调所有训练层**

```python
#载入预训练模型
model = resnet18(weights = 'DEFAULT')
model.fc = nn.Linear(model.fc.in_feature, n_class)
#传入全部参数就可以对全部参数进行微调
optimizer = optim.Adam(model.parameters())
```

关闭其他参数的梯度，只打开分类头的梯度

```python
# 关闭所有的grad
for param in model.parameters():
	param.requires_grad = False
# 只打开分类头的grad
for param in model.fc.parameters():
	param.requires_grad = True
```

#### 10.4.3 随机初始化模型全部权重，重新训练

**数据集与ImageNet完全不一样**

```python
# 只载入模型框架
model = resnet18(weights = 'DEFAULT')
model.fc = nn.Linear(model.fc.in_feature, n_class)
# 传入全部参数
optimizer = optim.Adam(model.parameters())
```

#### 10.4.4 迁移学习用于Leaves

```python




import torch
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from torchvision.io import read_image
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
import time
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter



idx_class_list = {'abies_concolor': 0, 'abies_nordmanniana': 1, 'acer_campestre': 2, 'acer_ginnala': 3, 'acer_griseum': 4, 'acer_negundo': 5, 'acer_palmatum': 6, 'acer_pensylvanicum': 7, 'acer_platanoides': 8, 'acer_pseudoplatanus': 9, 'acer_rubrum': 10, 'acer_saccharinum': 11, 'acer_saccharum': 12, 'aesculus_flava': 13, 'aesculus_glabra': 14, 'aesculus_hippocastamon': 15, 'aesculus_pavi': 16, 'ailanthus_altissima': 17, 'albizia_julibrissin': 18, 'amelanchier_arborea': 19, 'amelanchier_canadensis': 20, 'amelanchier_laevis': 21, 'asimina_triloba': 22, 'betula_alleghaniensis': 23, 'betula_jacqemontii': 24, 'betula_lenta': 25, 'betula_nigra': 26, 'betula_populifolia': 27, 'broussonettia_papyrifera': 28, 'carpinus_betulus': 29, 'carpinus_caroliniana': 30, 'carya_cordiformis': 31, 'carya_glabra': 32, 'carya_ovata': 33, 'carya_tomentosa': 34, 'castanea_dentata': 35, 'catalpa_bignonioides': 36, 'catalpa_speciosa': 37, 'cedrus_atlantica': 38, 'cedrus_deodara': 39, 'cedrus_libani': 40, 'celtis_occidentalis': 41, 'celtis_tenuifolia': 42, 'cercidiphyllum_japonicum': 43, 'cercis_canadensis': 44, 'chamaecyparis_pisifera': 45, 'chamaecyparis_thyoides': 46, 'chionanthus_retusus': 47, 'chionanthus_virginicus': 48, 'cladrastis_lutea': 49, 'cornus_florida': 50, 'cornus_kousa': 51, 'cornus_mas': 52, 'crataegus_crus-galli': 53, 'crataegus_laevigata': 54, 'crataegus_phaenopyrum': 55, 'crataegus_pruinosa': 56, 'crataegus_viridis': 57, 'cryptomeria_japonica': 58, 'diospyros_virginiana': 59, 'eucommia_ulmoides': 60, 'evodia_daniellii': 61, 'fagus_grandifolia': 62, 'ficus_carica': 63, 'fraxinus_nigra': 64, 'fraxinus_pennsylvanica': 65, 'ginkgo_biloba': 66, 'gleditsia_triacanthos': 67, 'gymnocladus_dioicus': 68, 'halesia_tetraptera': 69, 'ilex_opaca': 70, 'juglans_cinerea': 71, 'juglans_nigra': 72, 'juniperus_virginiana': 73, 'koelreuteria_paniculata': 74, 'larix_decidua': 75, 'liquidambar_styraciflua': 76, 'liriodendron_tulipifera': 77, 'maclura_pomifera': 78, 'magnolia_acuminata': 79, 'magnolia_denudata': 80, 'magnolia_grandiflora': 81, 'magnolia_macrophylla': 82, 'magnolia_stellata': 83, 'magnolia_tripetala': 84, 'magnolia_virginiana': 85, 'malus_baccata': 86, 'malus_coronaria': 87, 'malus_floribunda': 88, 'malus_hupehensis': 89, 'malus_pumila': 90, 'metasequoia_glyptostroboides': 91, 'morus_alba': 92, 'morus_rubra': 93, 'nyssa_sylvatica': 94, 'ostrya_virginiana': 95, 'oxydendrum_arboreum': 96, 'paulownia_tomentosa': 97, 'phellodendron_amurense': 98, 'picea_abies': 99, 'picea_orientalis': 100, 'picea_pungens': 101, 'pinus_bungeana': 102, 'pinus_cembra': 103, 'pinus_densiflora': 104, 'pinus_echinata': 105, 'pinus_flexilis': 106, 'pinus_koraiensis': 107, 'pinus_nigra': 108, 'pinus_parviflora': 109, 'pinus_peucea': 110, 'pinus_pungens': 111, 'pinus_resinosa': 112, 'pinus_rigida': 113, 'pinus_strobus': 114, 'pinus_sylvestris': 115, 'pinus_taeda': 116, 'pinus_thunbergii': 117, 'pinus_virginiana': 118, 'pinus_wallichiana': 119, 'platanus_acerifolia': 120, 'platanus_occidentalis': 121, 'populus_deltoides': 122, 'populus_grandidentata': 123, 'populus_tremuloides': 124, 'prunus_pensylvanica': 125, 'prunus_sargentii': 126, 'prunus_serotina': 127, 'prunus_serrulata': 128, 'prunus_subhirtella': 129, 'prunus_virginiana': 130, 'prunus_yedoensis': 131, 'pseudolarix_amabilis': 132, 'ptelea_trifoliata': 133, 'pyrus_calleryana': 134, 'quercus_acutissima': 135, 'quercus_alba': 136, 'quercus_bicolor': 137, 'quercus_cerris': 138, 'quercus_coccinea': 139, 'quercus_imbricaria': 140, 'quercus_macrocarpa': 141, 'quercus_marilandica': 142, 'quercus_michauxii': 143, 'quercus_montana': 144, 'quercus_muehlenbergii': 145, 'quercus_nigra': 146, 'quercus_palustris': 147, 'quercus_phellos': 148, 'quercus_robur': 149, 'quercus_shumardii': 150, 'quercus_stellata': 151, 'quercus_velutina': 152, 'quercus_virginiana': 153, 'robinia_pseudo-acacia': 154, 'salix_babylonica': 155, 'salix_caroliniana': 156, 'salix_matsudana': 157, 'salix_nigra': 158, 'sassafras_albidum': 159, 'staphylea_trifolia': 160, 'stewartia_pseudocamellia': 161, 'styrax_japonica': 162, 'taxodium_distichum': 163, 'tilia_americana': 164, 'tilia_cordata': 165, 'tilia_europaea': 166, 'tilia_tomentosa': 167, 'tsuga_canadensis': 168, 'ulmus_americana': 169, 'ulmus_glabra': 170, 'ulmus_parvifolia': 171, 'ulmus_procera': 172, 'ulmus_pumila': 173, 'ulmus_rubra': 174, 'zelkova_serrata': 175}




class m_dataset(Dataset):

    def __init__(self, dataframe : pd.DataFrame, idx_class_list, root_path, transforms = None):
        super().__init__()
        self.dataframe = dataframe

        self.root_path = root_path

        self.transforms = transforms

        self.idx_class_list = idx_class_list


    def __getitem__(self, idx):

        img = read_image(path = os.path.join(self.root_path, self.dataframe.iloc[idx, 0]))

        if self.transforms is not None:
            img = self.transforms(img)
      
        label = self.idx_class_list[self.dataframe.iloc[idx, 1]]

        return img, label
  
    def __len__(self):
        return len(self.dataframe)



if __name__ == '__main__':

    # 参数区
    m_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 30
    learn_rate = 1e-4
    batch_size = 64
    num_workers = 8
    root_path = r'D:\AIProject\TorchVisionProject\classify-leaves\data'
    num_classes = len(idx_class_list)
    patience = 5
    best_avg_loss = float('inf')
    no_improve = 0

    # 数据预处理

    data_df = pd.read_csv(r'data\train.csv')

    train_df, test_df = train_test_split(
        data_df, 
        test_size = 0.2, 
        stratify = data_df['label'], 
        random_state = int(time.time())
    )

    # print('Train_Test_Split')

    # 模型、优化器、损失函数、数据块、数据迭代器、转换器

    model = resnet18(weights = 'DEFAULT')
    model.fc = nn.Linear(in_features = model.fc.in_features, out_features = num_classes)
    model = model.to(m_device)

  
    optimizer = torch.optim.Adam(params = model.parameters(), lr = learn_rate)

    loss = nn.CrossEntropyLoss().to(m_device)

    writer = SummaryWriter(r'logs')

    train_transforms = v2.Compose([
        v2.RandomHorizontalFlip(p = 0.5),
        v2.RandomVerticalFlip(p = 0.5),
        v2.RandomRotation(degrees = 30), 
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True), 
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True), 
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = m_dataset(train_df, idx_class_list, root_path, train_transforms)

    test_dataset = m_dataset(test_df, idx_class_list, root_path, test_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, pin_memory = True, persistent_workers = True, prefetch_factor = 4)

    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = True, persistent_workers = True, prefetch_factor = 4)

    train_step = 0
    test_step = 0
    for epoch in range(epochs):

        # 训练模型
        model.train()
        loss_sum = 0
        strat_time = time.time()
        for feature, true_label in train_dataloader:
          
            feature = feature.to(m_device, non_blocking = True)
            true_label = true_label.to(m_device, non_blocking = True)

            predict_label = model(feature)

            l = loss(predict_label, true_label)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_step += 1
            loss_sum += l.item()
            writer.add_scalar(tag = 'loss/train', scalar_value = l.item(), global_step = train_step)

        end_time = time.time()
        print('=' * 20)
        print(f'第{epoch + 1}次训练完毕，loss = {loss_sum / len(train_dataloader)}，用时{end_time - strat_time}')
        print('=' * 20)

        # 测试
        strat_time = time.time()
        with torch.no_grad():

            model.eval()
            loss_sum = 0
            correct_count = 0
            for feature, true_label in test_dataloader:

                feature = feature.to(m_device, non_blocking = True)
                true_label = true_label.to(m_device, non_blocking = True)

                predict_label = model(feature)

                l = loss(predict_label, true_label)

                predict_label = predict_label.argmax(1)

                correct_count += (predict_label == true_label).sum().item()

                loss_sum += l.item()

                test_step += 1

                writer.add_scalar(tag = 'loss/test', scalar_value = l.item(), global_step = test_step)
          
            end_time = time.time()
            print('*' * 20)
            print(f'第{epoch + 1}次测试，loss = {loss_sum / len(test_dataloader)}，用时{end_time - strat_time}')
            print(f'第{epoch + 1}次测试，正确率 = {correct_count / len(test_dataset)}')
            print('*' * 20)
      
            test_avg_loss = loss_sum / len(test_dataloader)
            if best_avg_loss > test_avg_loss:
                best_avg_loss = test_avg_loss
                torch.save(model.state_dict(), 'best_model.pth')
                no_improve = 0
            else: 
                no_improve += 1
                if no_improve >= patience:
                    print(f'训练早停于{epoch + 1}次训练')
                    break



        # 完成一轮epoch


    writer.close()

    print('=' * 20, "end", '=' * 20)
```

#### 10.4.5 迁移学习中，什么时候只对分类头训练，什么时候要对全部参数微调

**一、只训练分类头（特征提取模式）适用场景：**


| 条件                      | 原因                                           |
| --------------------------- | ------------------------------------------------ |
| 新数据与ImageNet高度相似  | 预训练backbone提取的特征直接可用               |
| 数据量很小（几百~几千张） | 全参数微调容易过拟合，冻结backbone是天然正则化 |
| 算力/时间受限             | 只训练最后几层，速度快                         |
| 任务本身是ImageNet子集    | 比如"猫 vs 狗"，ImageNet已有对应类别           |

**二、整体微调适用场景：**

大部分实际工程中，整体微调更常见。因为预训练模型学的是通用特征（边缘、纹理、形状基元），而下游任务需要特定的细粒度特征，中间有gap，需要让整个网络去适应。

**关键操作——分层学习率（discriminative fine-tuning）：**

- 分类头（新层）：lr = 1e-3 或更高（随机初始化，需要大步更新）
- 骨干网络（预训练层）：lr = 1e-4 或 1e-5（已训练好，小步微调）

比全局统一学习率效果更好。

**三、直觉类比：**

预训练模型 = 一个已会画画的人

- 只训分类头：换一支新笔写字，手不动。任务和他以前画的像 → 没问题；完全不同风格 → 笔再好也没用。
- 整体微调（小lr）：整个手都可以动，但幅度很小。既保留已有功底，又适应新风格。

**四、实验验证方法——逐层解冻：**

Step 1 → 只训分类头
Step 2 → 解冻最后一层残差块
Step 3 → 解冻最后两层
Step 4 → 全部解冻

观察每层解冻带来的准确率增量，能直观看到"哪一层对新任务特征最敏感"。解冻更多层不涨点 → 数据量不够支撑更深微调。

> 实例：ResNet18做叶子分类，只训分类头仅60%，全参数微调（小lr）达94%。叶子分类需要叶脉纹理等细粒度特征，ImageNet预训练未覆盖，必须让网络去适应新特征空间。




### 10.5 Plant Pathology 2021 - FGVC8项目

#### 10.5.1 项目概况

- **任务**：苹果叶片病害分类，数据集 ~18k 张（4000×2672 高清 JPEG），12 个类别
- **模型**：ResNet18，全参数微调
- **硬件**：RTX 4060 8GB，双系统（Windows 数据分区 + Linux 训练环境）

#### 10.5.2 数据增强（Data Augmentation）

**项目中的演变过程**：

| 阶段 | transforms | 问题 |
|------|-----------|------|
| 初版 | `RandomHorizontalFlip` + `RandomRotation(30)` + `Resize(512)` + `RandomCrop(512)` | `RandomCrop` 在 512×512 图上裁 512 永远是原图，形同虚设 |
| 改进 | 替换为 `RandomResizedCrop(512, scale=(0.5, 1.0))` | 随机缩放+裁切，真正起增强作用 |
| 过拟合后 | 加入 `ColorJitter`、`RandomErasing`、`label_smoothing=0.1` | 丰富颜色/光照变化 + 随机遮挡 + 软化标签 |

**最终训练 transforms**：

```python
transforms = v2.Compose([
    v2.RandomResizedCrop(512, scale=(0.5, 1.0)),   # 随机缩放+裁切
    v2.RandomHorizontalFlip(0.5),                    # 水平翻转
    v2.RandomVerticalFlip(0.2),                      # 垂直翻转（叶片方向不固定）
    v2.RandomRotation(15),                           # 轻微旋转
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.ToDtype(torch.float32, scale=True),           # uint8 → float32 [0,1]
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.RandomErasing(p=0.2, scale=(0.02, 0.1)),     # 随机遮挡，防模型只盯一个区域
])
```

**核心经验**：

1. **`RandomResizedCrop` 优于 `Resize + RandomCrop`**：前者在每个 epoch 对每张图做不同缩放比例和位置的裁切，后者当图尺寸等于裁切尺寸时退化到 no-op。`scale` 参数控制缩放范围，`(0.5, 1.0)` 意味着随机缩到原图的 50%-100%，模拟近景远景变化。

2. **`ColorJitter` 对病害分类很重要**：真实场景光照差异大，模型不应依赖特定色调。brightness/contrast/saturation 设 0.2 左右不会破坏叶片纹理，但能迫使模型关注形状和病斑纹理而非颜色。

3. **`RandomErasing` 强制多区域学习**：随机擦除一块矩形，模型不能只靠最显著的一片病斑判断，必须学会综合多处区域。`scale` 控制擦除区域占比，0.02-0.1 不会盖掉整个病斑。erase 值默认填 0，在 Normalize 后对应 ImageNet 均值色，视觉上自然；也可放 Normalize 前（填纯黑），效果类似。

4. **`label_smoothing=0.1`**：交叉熵损失要求模型输出 one-hot 般的高置信度，容易过拟合。平滑后 soft target（如 `[0.08, 0.08, ..., 0.92]` 而非 `[0, 0, ..., 1]`）让模型不那么"死记"，对多分类（尤其是类别数多）减轻过拟合效果明显。

5. **`Normalize` 必须在 `ToDtype` 之后**：`scale=True` 将 uint8 [0,255] 转为 float [0,1]，然后 Normalize 用 ImageNet 均值/标准差做标准化。缺这一步预训练权重直接失效。注意 `ToImage()` 在 torchvision ≥0.15 后通常不需要手动加，`ToDtype` 已处理。

6. **训练/验证 transforms 不对称原则**：训练用 `RandomResizedCrop`（随机裁切），验证用 `Resize + CenterCrop`（确定性中心裁切），避免验证时引入随机性。验证集不加 ColorJitter/RandomErasing 等只对训练有效的增强。

#### 10.5.3 性能调试

**GPU 利用率忽高忽低**：

GPU-Util 从 90%+ 骤降到 ~0 再回升，是**数据加载瓶颈**的典型症状。GPU 算完一个 batch 后，CPU 还没来得及把下一个 batch 准备好，GPU 空转等待。

**解决路径**：

| 优先级 | 方法 | 原理 |
|--------|------|------|
| 1 | 预处理：一次性 resize 所有图到目标尺寸 | 4000×2672 → 512×512，跳过实时 decode+resize，I/O 量降 ~20 倍，每张图从 2-4MB 降到 ~80KB |
| 2 | `torch.backends.cudnn.benchmark = True` | 固定输入尺寸下，cuDNN 前几次迭代试跑所有卷积算法后自动选最快，比默认启发式快 10-20% |
| 3 | 数据从 NTFS 分区拷到 Linux 本地磁盘 | 消除跨文件系统挂载开销，实测慢 10-20%，不多但能省则省 |
| 4 | 调整 `num_workers` | worker 不是越多越好。有 I/O 瓶颈时 8→4 反而更稳，避免多 worker 争抢同一磁盘带宽 |
| 5 | `pin_memory=True` + `non_blocking=True` | 锁定内存页加速 Host→Device 传输；异步拷贝不阻塞 GPU 当前计算 |

**扩展示例——用 torch.profiler 定位瓶颈更精确**：

```python
# 初级：用 time 看每个 epoch 耗时
# 进阶：用 torch.profiler 看 DataLoader 时间和 GPU 时间占比
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    for batch in dataloader:
        ...
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

如果 DataLoader 的 CPU 时间占比 > GPU kernel 时间，就确认是 I/O 瓶颈。

**显存管理实践**：

- 4000×2672 原图解码为 float32 tensor 约 `3×4×4000×2672 ≈ 122MB`，batch=32 仅输入就是 ~3.9GB
- Resize 到 512×512 后单张 ~3MB，batch=32 约 100MB
- ResNet18（11.7M 参数）+ AdamW 优化器状态 ≈ 200MB
- 训练时显存稳定在 ~6.9GB/8GB（85%），有余量但不算充裕。若 OOM，优先降 batch_size 或改用混合精度 AMP

**预处理脚本——一次性操作，永久受益**：

```python
# 遍历原图 → PIL resize 到目标尺寸 → 存到独立目录
# 之后训练直接读小图，DataLoader 不再做昂贵的实时 decode+resize
output_dir = 'data/train_img_512/'
os.makedirs(output_dir, exist_ok=True)
for img_name in img_list:
    img = Image.open(os.path.join(src_dir, img_name))
    img = img.resize((512, 512))
    img.save(os.path.join(output_dir, img_name))
```

20k 张图预处理约 25 分钟，但此后每个 epoch 都免去实时 decode 开销。处理完后图从总计 20-40GB 缩到 1-2GB。

**训练速度对照**（4060, 512×512）：

| 模型 | 参数量 | 速度 | 每 epoch |
|------|--------|------|----------|
| ResNet34 | 21.3M | ~30 img/s | ~9 min |
| ResNet18 | 11.7M | ~55 img/s | ~5 min |

ResNet18 比 ResNet34 快 ~45%，精度掉 1-3 个百分点，在 12 分类场景下可以接受。

#### 10.5.4 训练策略

| 决策 | 理由 |
|------|------|
| 全参数微调（lr=1e-4） | 20k 数据足够，叶片纹理与 ImageNet 差异大，只训分类头学不到病害特征 |
| AdamW vs SGD | AdamW 指数移动平均动量 + 解耦权重衰减，收敛稳定；SGD 需精细调参，容易翻车 |
| weight_decay=1e-4 | L2 正则惩罚系数，偏保守；过拟合时可提至 5e-4 甚至 1e-3 |
| `ReduceLROnPlateau(patience=3, factor=0.5)` | 验证 loss 连续 3 epoch 不降时自动将 lr 缩半，比固定 StepLR 对过拟合更敏感 |
| epochs=30 + 早停(patience=5) | 双层保护：早停在先，固定轮数在后。20k 数据全参数微调通常 10-20 epoch 即收敛 |

#### 10.5.5 核心教训：多标签 vs 多分类

**90% 准确率瓶颈不在代码，在标签体系**。

数据集本质是**多标签任务**——一张叶片可同时患多种病（如 `scab + frog_eye_leaf_spot`），数据提供方将 6 种病的所有组合拍平成 12 个"伪多分类"类别。

后果：
- 纯单病类（healthy, scab, powdery_mildew）样本充沛，正确率 95%+
- 复合病类样本极少（`rust complex` 仅 3 张，`rust frog_eye_leaf_spot` 仅 9 张），模型完全学不到，正确率 0%
- 复合类本质共享单病特征（`scab frog_eye_leaf_spot` 和 `scab` 共享 scab 特征），被强行独立后信息断裂，复合类学不到任何有用信号

**正确做法**：6 个独立二分类头 + `BCEWithLogitsLoss`，每张图同时预测 6 个标签。推断时若同时命中 `scab` 和 `frog_eye_leaf_spot`，即为该复合病。稀有组合也能从单病样本中共享特征，样本量不再为零。

> 这个问题在医学影像、农业病害、遥感等需要多标签分类的场景普遍存在。如果数据集有复合标签，务必在开始训练前想清楚：拍平多分类还是多标签？拍平的代价是稀有复合类 = 0 样本。

#### 10.5.6 早停思路：Generalization Gap

训练 loss 和验证 loss 的比值/差值可作为补充早停信号——即便验证 loss 还在震荡，若 gap 持续扩大，说明模型已开始死记训练集。

该方法最早见于 **Prechelt (1998)** "Early Stopping — But When?"，提出 GL (Generalization Loss) = (当前验证 loss / 历史最低验证 loss - 1) × 100。Kaggle 社区也常用 train/test loss ratio。建议配合验证 loss 判断，不单独使用，避免训练早期 gap 大时误停。

#### 10.5.7 环境速查

| 问题 | 解决 |
|------|------|
| torchvision ≥0.15 `read_image` 路径变更 | `from torchvision.io import read_image` |
| Tensor 无 `.show()` | `to_pil_image(tensor).show()` |
| 模型 fc 替换后仍在 CPU | 先 `model.fc = nn.Linear(...)` 再 `model.to(device)` |
| `CrossEntropyLoss` 要求 target shape | `(batch,)` 一维整数类标，`(batch, 1)` 会报错 |
| Ubuntu 锁屏不影响训练，休眠会中断 | `sudo systemctl mask sleep.target suspend.target` |



## 11. 注意力机制

### 11.1 Encoder和Decoder

**Encoder将数据和特征转化为一个中间态（state），然后Decoder将中间态转化为输出。例如：卷积神经网络中所有的卷积部分就是Encoder，将图片转化为张量，而最后的全连接层就是Decoder，将展平的张量转化为最终结果。**

### 11.2 Seq2seq

#### 11.2.1 **Seq2seq结构图**

<img src="img/Seq2seq结构图.png" style="zoom:33%;" />

> 其中Encoder因为通常都是能看到全部的序列，所以可以设计成双向序列模型
> 但是Decoder因为要预测，所以不能设计成双向序列模型

#### 11.2.2 **具体工作流程**

1. Encoder先对完整序列进行编码，也就是用RNN或者LSTM进行学习，将最后得到的隐藏状态向量传给Decoder。
2. 然后Decoder接收到隐藏状态向量，然后再接收一个输入，例如图中输入```<bos>```，然后Decoder根据```hidden state```再加上输入的```<bos>```得到输出```bonjour```，然后将这个预测的```bonjour```再作为输入结合```hidden state```继续进行预测。
3. 最后遇到```<eos>```就停下，否则会一直输出。

#### 11.2.3 Seq2seq中解码器和编码器具体工作流程

<img src="img/Seq2seq编码器解码器细节.png" style="zoom:33%;" />

> 其中Encoder是不需要输出的，也就是不需要全连接层，Encoder输出的Hidden state直接伴随着Decoder的输入，进入到Decoder中，也就是Hidden state从Decoder的最底层往上传。
> 然后Encoder输出的Hidden state也会作为Decoder的初始Hidden state（隐状态）。

#### 11.2.4 训练流程

<img src="img/Seq2seq训练.png" style="zoom:33%;" />

1. 训练过程中Encoder正常工作
2. 然后Decoder开始工作时，先根据传入的隐状态和输入进行预测，这个**预测不管对不对都不会当成下一层的输入**， 因为训练是知道正确的输出的，所以不管预测的对不对，只会拿正确输出作为下一轮的输入，防止模型在训练过程中严重学偏。

#### 11.2.5 衡量模型好坏的指标

<img src="img/衡量模型好坏的指标.png" style="zoom:33%;" />

**其中的n-gram精度是指预测序列中，连续的n个预测是否出现在标签序列（非原序列而是标签序列）的概率。
例如：1-gram，就是连续一个预测，预测序列是：A B B C D，其中A在标签序列中出现了，B出现了，第二个B重复了，不算，以此类推，P1 = 4/5；
2-gram，就是连续两个预测，AB出现在标签序列了，BB没有，BC有，CD有，P2 = 3/4。**

**BLEU是越大越好**

> 理论上预测越短精度越高，但是这明显是不符合直觉的，如果只预测一个A的话，只要标签序列出现过A精度就永远是1。
> 所以为了防止这种情况出现，BLEU设置了惩罚项。

### 11.3 注意力机制（重点）

#### 11.3.1 为什么需要注意力

核心问题：传统Seq2Seq把整句话压缩成一个固定长度的「句子向量」，再从这个向量解码出翻译。句子越长，压缩丢失的信息越多。

注意力机制解决的就是这个问题——翻译每个词时，允许模型「回头看」原文，选择性地关注相关的部分。不再把整句话压成一个死向量。

**一句话总结**：注意力就是让模型学会「看哪里」，用相似度做权重，对所有信息做加权平均。

---

#### 11.3.2 注意力机制的7行灵魂代码

```python
def attention(query, keys, values):
    # 第1步：算 query 和每个 key 的相似度（点积）
    scores = [dot(query, key) for key in keys]
  
    # 第2步：把相似度变成权重（加起来等于1）
    weights = softmax(scores)
  
    # 第3步：用权重对所有 value 做加权平均
    output = sum(w * v for w, v in zip(weights, values))
  
    return output, weights
```

就这么点东西。

---

#### 11.3.3 软注意力 vs 硬注意力（关键理解点）

**软注意力（Soft Attention）**：不是选一个最相关的词，而是**把所有词的信息按权重混合**。这意味着「it」可以同时关注「cat」和「mat」，只是关注程度不同。这是注意力机制真正神奇的地方。

**硬注意力（Hard Attention）**：直接选权重最高的那个词，忽略其他。可微分性差，训练困难。

---

#### 11.3.4 Q、K、V 的含义——用淘宝搜索理解


| 字母 | 全称  | 作用                                               | 类比                                |
| ------ | ------- | ---------------------------------------------------- | ------------------------------------- |
| Q    | Query | 「我想找什么」——当前要生成的词需要什么信息       | 你在淘宝搜索框输入「机械键盘 红轴」 |
| K    | Key   | 每个位置的「标签」——用于算相似度、决定「该看谁」 | 每个商品的标题——帮你找到它        |
| V    | Value | 每个位置「实际携带的信息」——真正做加权平均的内容 | 每个商品的详情——你要读的内容      |

**核心理解**：K和V都蕴含这个token的信息——

- **K**：token全方位的总结（语义 + 结构 + 功能 + 位置……），负责吸引注意力
- **V**：token的具体语义含义，负责贡献实际内容

**为什么K和V必须分开？**

- 如果K=V，结构词（的、了、着、and、the、of……）会失效
- 这些词语义内容几乎为空（V接近零向量），但结构指示性极强
- 如果K=V，它们的K也接近零向量 → 没人会注意到它 → 修饰关系丢失
- 后果：生成文本中形容词占据绝大多数，结构词消失——这不符合自然语言的直觉

**验证**：如果把V全部设成零向量，注意力机制会正常运作（权重分布合理），但输出全为零。这是「货物崇拜注意力」——形式全部到位，信息为零。K决定「看哪里」，V决定「看到什么」，缺一不可。

---

#### 11.3.5 自注意力 vs 交叉注意力（别搞混）

**自注意力（Self-Attention）**：Q、K、V 都来自同一句话。让一句话里的词互相看对方。

> 「The cat sat on the mat」→ 每个词都和句子里所有其他词算注意力

**交叉注意力（Cross-Attention）**：Q 来自一个地方（你要生成的），K、V 来自另一个地方（原文）。

> 翻译场景：Q 来自目标语言（要写的中文），K、V 来自源语言（英文原文）


|       | 自注意力               | 交叉注意力             |
| ------- | ------------------------ | ------------------------ |
| Q来源 | 同一句话               | 目标序列（你要生成的） |
| K来源 | 同一句话               | 源序列（原文）         |
| V来源 | 同一句话               | 源序列（原文）         |
| 作用  | 让词之间互相理解上下文 | 让生成时能看原文的哪里 |

---

#### 11.3.6 Scaled Dot-Product Attention

##### 完整流程（5步）

```
输入: 三个token "I love you"，d_model = 4

Step 1: 每个token投影出 Q, K, V
Step 2: Q 和 所有K 做点积 → 分数(scores)
Step 3: 分数 / √d → 缩放
Step 4: softmax(缩放后分数) → 权重(weights)，总和=1
Step 5: 权重 × V → 加权求和 → 输出
```

**用词区分（重要）**：

```
Q·K  →  分数(scores)  →  /√d  →  softmax  →  权重(weights)
        ↑ 缩放在这里                          ↑ 这里出来的才叫权重
```

缩放的是**分数**，不是权重。权重是softmax之后的结果。

##### 具体数值例子（翻译"I love you"的第三个词）

```python
Q_you = [0.2, -0.6, -0.4, 0.7]   # "you"想找什么
K_I    = [-0.2, 0.5, 0.3, -0.1]  # "I"的身份标签
K_love = [0.4, -0.3, 0.6, 0.2]   # "love"的身份标签
K_you  = [0.1, 0.7, -0.5, 0.3]   # "you"的身份标签
V_I    = [1.0, 0.0, 0.5, -0.3]   # "I"携带的语义
V_love = [0.2, 1.5, -0.2, 0.8]   # "love"携带的语义
V_you  = [-0.5, 0.3, 1.2, 0.4]   # "you"携带的语义

# Step 2: Q_you 和 每个K 点积
Q_you · K_I    = -0.390
Q_you · K_love =  0.625
Q_you · K_you  =  0.070

# Step 3-4: 缩放 + softmax → 权重
权重: [0.132,  0.557,  0.311]
        ↑I       ↑love    ↑you

# Step 5: 加权求和
output = 0.132×V_I + 0.557×V_love + 0.311×V_you
```

解读：翻译"you"时，模型觉得"love"最相关（55.7%），"you"自己其次（31.1%），"I"最不相关（13.2%）。

##### 一张图钉死整个过程

```
                    Q_you
                     │
                     │  点积（算相似度）
                     ▼
     ┌───────────────┬───────────────┬───────────────┐
     │    K_I        │    K_love     │    K_you      │
     │    -0.390     │    0.625      │    0.070      │  ← 分数(scores)
     └───────────────┴───────────────┴───────────────┘
                     │
                     │  /√d + softmax
                     ▼
     ┌───────────────┬───────────────┬───────────────┐
     │    0.132      │    0.557      │    0.311      │  ← 权重(weights)
     └───────────────┴───────────────┴───────────────┘
                     │
                     │  加权求和
                     ▼
     output = 0.132×V_I + 0.557×V_love + 0.311×V_you
   
     Q是提问者，K是应答标签，V是实际内容
     Q·K 决定"听谁的"，V 决定"听到什么"
```

##### 为什么必须缩放——实验验证

点积有一个性质：向量维度d越高，点积的方差越大（约等于d）。高维空间里softmax输入过于「尖锐」→ 某个分数极大，其他趋近于零 → 梯度消失 → 模型学不动。

**d=512时的实际效果**：

```
不缩放：
Softmax权重: [~0,  1.0,  0,  0,  ~0,  ~0,  ~0,  ~0,  ~0,  ~0]
最大权重: 1.0000  最小权重: 0.000000
→ 退化成了硬注意力——只看一个词，其他全瞎

缩放后 (除以√512 ≈ 22.6)：
Softmax权重: [0.10, 0.38, 0.004, 0.007, 0.097, 0.082, 0.112, 0.073, 0.088, 0.057]
最大权重: 0.3801  最小权重: 0.0038
→ 分布合理，能看到多个相关位置
```

**为什么高维导致方差大**：点积 Q·K = q1k1 + q2k2 + ... + qdkd。d=512时，加了512个独立随机项。大数定律：维度越高，点积的绝对值倾向于越大 → softmax越尖锐 → 梯度越接近零。

**解决办法**：除以 √d

```python
scores = dot(query, keys) / sqrt(d)  # 就一行，拯救整个训练
```

一个除号，决定你的模型是**读了整句话**，还是**只读了一个词**。不加scale，注意力不是在「注意力」，是在「赌博」——每次都只赌一个词，赌错了就死。

---

#### 11.3.7 多头注意力（Multi-Head Attention）

语言中有无数种关系：语法关系、语义关系、指代关系、位置关系……

**做法**：并行跑多套注意力，每套有自己的Q、K、V投影矩阵。就像戴8副不同的眼镜看同一句话——一副看「谁修饰谁」，一副看「代词指什么」，一副看「动词宾语关系」……

```python
def multi_head_attention(Q, K, V, num_heads=8):
    outputs = []
    for head in range(num_heads):
        Q_h = project(Q, W_q[head])  # 每个头用自己的投影
        K_h = project(K, W_k[head])
        V_h = project(V, W_v[head])
        outputs.append(scaled_dot_product_attention(Q_h, K_h, V_h))
    return project(concat(outputs), W_o)  # 拼起来再投影
```

**工程洞察**：如果训练后发现多个头几乎一样，说明这个任务不需要那么多头——可以砍掉省算力。

---

#### 11.3.8 复杂度

自注意力：每个词和全文每个词算一次相似度，n个词 → n×n次计算 → **O(n²)**。

---

#### 11.3.9 Q、K、V 的 Shape 理解——从教学到实战

**核心困惑**：教学时Q是一个向量 `(d,)`，写代码时Q是三维张量 `(B, n, d)`。为什么？

**答案**：逻辑完全一样，只是把多个词、多个句子同时算而已。

##### 三层递进

**级别1：一个词（教学用）**

```
Q = [0.2, -0.6, -0.4, 0.7]    shape = (d)         ← 一个查询向量
K = [[...], [...], [...]]      shape = (n, d)      ← n个键向量

Q · K[0] = 分数0
Q · K[1] = 分数1     →  得到n个分数，softmax后得n个权重
Q · K[2] = 分数2
```

##### 级别2：一句话（n个词同时算）

一句话3个词 "I love you"，每个词都有自己的Q：

```
Q = [[ 0.3, -0.1,  0.8,  0.2],   ← Q_I   词1在问"该看谁"
     [-0.5,  0.4,  0.1, -0.3],   ← Q_love 词2在问"该看谁"
     [ 0.2, -0.6, -0.4,  0.7]]   ← Q_you 词3在问"该看谁"
   
shape = (n, d)    n=seq_len, d=d_model
```

矩阵乘法一次性算出所有词对：

```
scores = Q @ K.T     (n,d) @ (d,n) = (n,n)

          K_I  K_love  K_you
   Q_I  [ 0.8   0.2    0.1  ]   ← Q_I 对各词的注意力分数
Q_love  [ 0.1   0.9    0.5  ]   ← Q_love 对各词的注意力分数
 Q_you  [-0.4   0.6    0.07 ]   ← Q_you 对各词的注意力分数

每一行 = 一个词对整句话的注意力分布
scores[i][j] = 第i个词认为第j个词有多重要
```

##### 级别3：一个batch（B句话并行）

GPU不喜欢一次只算一句话，一次扔进去32句：

```
Q shape = (B, n, d)
          ↑   ↑   ↑
        batch  seq  d_model
        32句  每句 每个token
              5词 的向量维度
```

32句话并行，矩阵乘法逻辑完全一样——只是B维度原样保留：

```python
scores = Q @ K.transpose(-2, -1)
# (B, n, d) @ (B, d, n) = (B, n, n)
#  32句话   每句话得到一个 n×n 的注意力矩阵
```

##### 三个 Shape 对照表


|           | Q shape     | K shape     | V shape     | scores shape | output shape |
| ----------- | ------------- | ------------- | ------------- | -------------- | -------------- |
| 一个词    | `(d,)`      | `(n, d)`    | `(n, d)`    | `(n,)`       | `(d,)`       |
| 一句话    | `(n, d)`    | `(n, d)`    | `(n, d)`    | `(n, n)`     | `(n, d)`     |
| 一个batch | `(B, n, d)` | `(B, n, d)` | `(B, n, d)` | `(B, n, n)`  | `(B, n, d)`  |

##### 关键操作：`.transpose(-2, -1)` 不是 `.T`

```python
# ✗ 错误：.T 把所有维度全反转，batch维度也被动了
scores = Q @ K.T   # (B,n,d) @ (d,n,B) → 维度错乱

# ✓ 正确：只交换最后两维，batch维度不动
scores = Q @ K.transpose(-2, -1)  # (B,n,d) @ (B,d,n) = (B,n,n)
```

`.transpose(-2, -1)` = 倒数第二维和最后一维交换。
K从 `(B, n, d)` → `(B, d, n)`，batch维度原封不动。

##### 本质

Q里的每一个词向量做的事，和单向量教学版**完全一样**——点积、算分数、加权V。只是把所有词、所有句子**同时做**了而已。这就是矩阵乘法的力量。

### 11.4 正则表达式

#### 11.4.1 为什么NLP需要正则

英文文本预处理（清洗、分词、去噪）离不开正则表达式。一个正则能省几十行手动处理代码。

#### 11.4.2 基本语法速查

##### 字符匹配


| 符号 | 含义                        | 示例    | 匹配结果            |
| ------ | ----------------------------- | --------- | --------------------- |
| `.`  | 任意一个字符（除换行）      | `a.b`   | `aab`, `acb`, `a b` |
| `\d` | 数字 [0-9]                  | `\d+`   | `123`, `456`        |
| `\w` | 字母数字下划线 [a-zA-Z0-9_] | `\w+`   | `hello`, `var_1`    |
| `\s` | 空白符（空格、tab、换行）   | `a\s+b` | `a b`, `a  b`       |
| `\D` | 非数字                      | `\D+`   | `abc`               |
| `\W` | 非字母数字下划线            | `\W+`   | `!!!`, `@#$`        |
| `\S` | 非空白符                    | `\S+`   | `hello`             |

##### 数量


| 符号    | 含义      | 示例                                      |
| --------- | ----------- | ------------------------------------------- |
| `*`     | 0次或多次 | `ab*c` 匹配 `ac`, `abc`, `abbbc`          |
| `+`     | 1次或多次 | `ab+c` 匹配 `abc`, `abbbc`（不匹配 `ac`） |
| `?`     | 0次或1次  | `ab?c` 匹配 `ac`, `abc`                   |
| `{n}`   | 恰好n次   | `\d{3}` 匹配 `123`, `456`                 |
| `{n,}`  | 至少n次   | `\d{3,}` 匹配 `123`, `456789`             |
| `{n,m}` | n到m次    | `\d{2,4}` 匹配 `12`, `123`, `1234`        |

##### 边界


| 符号 | 含义       |
| ------ | ------------ |
| `^`  | 字符串开头 |
| `$`  | 字符串结尾 |
| `\b` | 单词边界   |

> `^The` → 以"The"开头的字符串
> `\bcat\b` → 只匹配单词"cat"，不匹配"caterpillar"或"concatenate"

##### 字符类


| 写法       | 含义                      |
| ------------ | --------------------------- |
| `[abc]`    | a、b、c 中任意一个        |
| `[a-z]`    | a到z任意一个小写字母      |
| `[A-Z]`    | A到Z任意一个大写字母      |
| `[a-zA-Z]` | 任意一个字母              |
| `[^abc]`   | **不是**a、b、c的任意字符 |

##### 分组与或


| 符号        | 含义                             |
| ------------- | ---------------------------------- |
| `(abc)`     | 分组，把abc当成一个整体          |
| `(abc|def)` | abc 或 def                       |
| `(?:abc)`   | 非捕获分组（只用于组织，不提取） |

#### 11.4.3 NLP中最常用的几个正则

```python
import re

# 1. 转小写 + 去掉标点符号
text = "Hello, World! How are you?"
cleaned = re.sub(r'[^\w\s]', '', text.lower())
# → "hello world how are you"

# 2. 去掉HTML标签
text = "<p>Hello <b>World</b></p>"
cleaned = re.sub(r'<[^>]+>', '', text)
# → "Hello World"

# 3. 去掉多余空格（包括换行、tab）
text = "Hello   World\n\t  !"
cleaned = re.sub(r'\s+', ' ', text).strip()
# → "Hello World !"

# 4. 去掉数字
text = "I have 3 apples and 15 bananas"
cleaned = re.sub(r'\d+', '', text)
# → "I have  apples and  bananas"

# 5. 只保留字母和空格
text = "Hello!!! 123 World @#$"
cleaned = re.sub(r'[^a-zA-Z\s]', '', text)
# → "Hello  World "

# 6. 匹配网址
text = "Visit https://www.google.com now"
urls = re.findall(r'https?://\S+', text)
# → ['https://www.google.com']

# 7. 分词（split）
text = "Hello, World! How are you?"
tokens = re.split(r'\s+|[,!?]', text.lower())
tokens = [t for t in tokens if t]  # 去掉空字符串
# → ['hello', 'world', 'how', 'are', 'you']
```

#### 11.4.4 核心理解

**`r'...'`**：Python里用 `r` 前缀，让反斜杠 `\` 不被转义。`r'\d'` 等于正则里的 `\d`，如果用普通字符串得写成 `'\\d'`。

**`re.sub(pattern, replacement, text)`**：查找并替换。`re.sub(r'\d+', '', text)` = 把所有数字删掉。

**`re.findall(pattern, text)`**：返回所有匹配项的列表。

**`re.split(pattern, text)`**：按模式切分字符串。

**`re.search(pattern, text)`**：找到第一个匹配，返回Match对象（或None）。

##### 贪婪 vs 非贪婪（重点）


| 写法  | 含义                 |
| ------- | ---------------------- |
| `.*`  | 贪婪：尽可能多匹配   |
| `.*?` | 非贪婪：尽可能少匹配 |

```python
text = "<p>Hello</p><p>World</p>"
re.findall(r'<p>.*</p>', text)   # → ['<p>Hello</p><p>World</p>']  全吞了
re.findall(r'<p>.*?</p>', text)  # → ['<p>Hello</p>', '<p>World</p>']  正确
```

NLP清洗中**几乎永远用非贪婪**`.*?`，否则一句话里的标签会被一股脑吞掉。

#### 11.4.5 collections.Counter —— 词频统计的瑞士军刀

`Counter` 是 `collections` 模块里最常用的类，NLP建词表时**离不开它**。

##### 基本用法

```python
from collections import Counter

# 创建：传入可迭代对象
words = ['i', 'love', 'you', 'i', 'love', 'love']
c = Counter(words)
print(c)
# Counter({'love': 3, 'i': 2, 'you': 1})

# 也可以从字符串创建（统计字符）
c = Counter('hello')
print(c)
# Counter({'l': 2, 'h': 1, 'e': 1, 'o': 1})
```

##### 常用操作

```python
c = Counter(['i', 'love', 'you', 'i', 'love', 'love'])

# 1. 查频率
print(c['love'])   # 3
print(c['cat'])    # 0（不存在的返回0，不报错！）

# 2. 出现次数最多的前k个
print(c.most_common(2))
# [('love', 3), ('i', 2)]

# 3. 转普通字典
d = dict(c)
# {'i': 2, 'love': 3, 'you': 1}

# 4. 更新（累加）
c.update(['i', 'cat'])
# Counter({'love': 3, 'i': 3, 'you': 1, 'cat': 1})

# 5. elements() —— 按计数展开
print(list(c.elements()))
# ['i', 'i', 'love', 'love', 'love', 'you']
```

##### NLP实战：从文本到词表

```python
from collections import Counter
import re

texts = [
    "The cat sat on the mat.",
    "The dog sat on the log.",
    "The cat and the dog played."
]

# 1. 清洗 + 分词 + 统计
all_tokens = []
for text in texts:
    cleaned = re.sub(r'[^\w\s]', '', text.lower())  # 去标点 + 小写
    tokens = cleaned.split()
    all_tokens.extend(tokens)

counter = Counter(all_tokens)
print(counter)
# Counter({'the': 6, 'cat': 2, 'sat': 2, 'on': 2, 'dog': 2, 'mat': 1, 'log': 1, 'and': 1, 'played': 1})

# 2. 建词表（word → idx）
# 特殊token：<PAD>=0（填充）, <UNK>=1（未知词）
vocab = {'<PAD>': 0, '<UNK>': 1}
for word, freq in counter.most_common():
    vocab[word] = len(vocab)

print(f"词表大小: {len(vocab)}")  # 11

# 3. 低频词过滤——只保留出现≥2次的词
min_freq = 2
frequent_words = [w for w, f in counter.items() if f >= min_freq]
print(frequent_words)
# ['the', 'cat', 'sat', 'on', 'dog']

# 4. 句子 → 索引序列
def text_to_ids(sentence, vocab):
    cleaned = re.sub(r'[^\w\s]', '', sentence.lower())
    tokens = cleaned.split()
    return [vocab.get(t, 1) for t in tokens]  # 未知词用<UNK>=1

print(text_to_ids("The cat and the bird.", vocab))
# [2 ('the'), 3 ('cat'), 9 ('and'), 2 ('the'), 1 ('bird'=<UNK>)]
```

##### Counter 为什么比手动统计好


| 手动写法                     | Counter                |
| ------------------------------ | ------------------------ |
| `d[key] = d.get(key, 0) + 1` | `Counter(list)`        |
| 取top-k要`sorted` + 切片     | `.most_common(k)` 一行 |
| 访问不存在的key会报错        | 返回0，不报错          |

##### 小结

```
Counter(list)          → 生成计数
c.most_common(k)       → 词频最高k个（建词表）
c.get(word, 0)          → 查词频，没有返回0
dict(c.most_common())   → 转{word: freq}字典
```

建词表流程：`读数据 → 分词 → Counter统计 → 过滤低频 → word→idx映射`。
