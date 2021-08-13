#!/usr/bin/env python
# coding: utf-8

# <p align=center><img src="./images/pytorch-logo-dark.png" alt="pytoch_logo" style="width:400px;"/></p>

# # Pytorch介绍

# PyTorch是Torch在Python上的衍生，于2017年初诞生于Facebook。
# 
# PyTorch的前身神经网络库Torch, Torch虽然很好用, 但是其开发语言Lua又不是特别流行, 所以开发团队将基于Lua的Torch移植到了更流行的语言 Python上

# ## Pytorch vs Tensorflow

# **动态图 vs 静态图**

# 尽管PyTorch和TensorFlow都在张量上工作，但PyTorch和Tensorflow之间的主要区别在于PyTorch使用动态计算图，而TensorFlow使用静态计算图。
# 
# 但随着Tensorflow 2.0的发布，目前Tensorflow开启Eager模式后也能支持动态图

# **模型部署**

# Tensorflow最棒的地方在于对生产环境的全方位支持。你可以通过Tensorflow Serving将任务模型进行大规模部署。
# 
# Pytorch在这方面比较欠缺，但随着TensorRT和ONNX等跨框架的部署工具的出现，这一差距正在被缩小。

# **现状**

# *Tensorflow*
# 
# 工业界很多系统已经使用Tensorflow 1.x稳定运行上线，且各大厂的攻城狮早期就和master分支分叉，又魔改Tensorflow，导致Tensorflow 2.x难以推广，更不敢轻易更换为PyTorch。
# 
# Tensorflow 2.x其实在易用性上已经和PyTorch持平了，但Google把Keras合并后却没有好好推广，导致目前TF2的使用率不高
# 
# *PyTorch*
# 
# 几乎已经制霸学术圈了，各大顶会论文几乎都是使用PyTorch开发的。如果你想了解最新某领域的发展的话，必然搜到的清一色PyTorch的代码。但工业界部署能力羸弱的问题还是存在的，可能再过几年会追上Tensorflow

# ## Pytorch安装

# Pytoch的安装还是很方便的，前往[官方主页](https://pytorch.org/get-started/locally/)就能根据自己设备选择合适的安装方式
# 
# 如果是需要安装CUDA的话，这里推荐使用Conda进行安装，会自动下载编译好的cudatoolkit

# ![pytorch_install](./images/pytorch_install.png)

# # 张量

# 张量（Tensor）是一种特殊的数据结构，与数组和矩阵非常相似。在PyTorch中，我们使用张量对模型的输入和输出以及模型的参数进行编码。

# 张量可以理解为可以使用GPU加速运算的NumPy的ndarray。如果你对NumPy的操作熟悉的话，PyTorch几乎可以无缝衔接

# In[3]:


import torch
import numpy as np


# In[ ]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Pytorch version: {torch.__version__}, available device: {device}")


# In[6]:


print(torch.__version__)
torch.cuda.is_available()


# ## 初始化张量

# 张量可以直接从数据中创建。数据类型是自动推断的

# In[3]:


data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)


# 张量也可以从`NumPy`数组创建

# In[4]:


np_array = np.array(data)
x_np = torch.from_numpy(np_array)


# 也可以基于已有张量构建，除非明确覆盖，否则新张量保留参数张量的属性（形状、数据类型）

# In[5]:


x_ones = torch.ones_like(x_data) # 保留x_data的属性
print(f"全一张量: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # 重新定义x_data的数据类型
print(f"随机张量: \n {x_rand} \n")


# 创建随机或者常量张量
# 
# `shape`定义张量的维度

# In[6]:


shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"随机张量: \n {rand_tensor} \n")
print(f"全一张量: \n {ones_tensor} \n")
print(f"全零张量: \n {zeros_tensor}")


# ## 张量属性

# 张量属性描述了它们的形状、数据类型和存储它们的设备。

# In[7]:


tensor = torch.rand(3,4)

print(f"张量形状: {tensor.shape}")
print(f"张量数据类型: {tensor.dtype}")
print(f"张量的存储设备: {tensor.device}")

tensor = tensor.to("cuda") # 存储到gpu上
print(f"张量的存储设备: {tensor.device}")


# ## 张量操作

# 如果您熟悉`NumPy`API，您会发现`Tensor`API使用起来轻而易举。

# **索引和切片**

# In[8]:


tensor = torch.ones(4, 4)
print('第一行: ',tensor[0])
print('第一列: ', tensor[:, 0])
print('最后一列:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)


# **张量合并**

# 在**旧**维度上，连接一系列张量

# In[9]:


t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
print(t1.shape)


# 在**新**维度上，连接一系列张量

# In[10]:


t2 = torch.stack([tensor, tensor, tensor], axis=1)
print(t2)
print(t2.shape)


# **算术运算**

# In[11]:


# 对两个张量进行矩阵运算
# y1, y2, y3的运算结果是一致的
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)
print(y3)


# 对两个张量进行按元素相乘运算
# z1, z2, z3的运算结果是一致的
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z3)


# **单元素张量**

# 将张量的所有值聚合为一个值，您可以使用`item()`将其转换为`Python`数值

# In[12]:


agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))


# **就地操作**

# 将结果存储到当前张量中的操作被称为就地。它们由 _ 后缀表示。例如：x.copy_(y), x.t_()，会直接改变x的值

# In[13]:


print(tensor, "\n")
tensor.add_(5)
print(tensor)


# **与NumPy连接**

# 张量转为NumPy数组

# In[14]:


t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")


# 张量的变化反映在`NumPy`数组中

# In[15]:


t.add_(1)
print(f"t: {t}")
print(f"n: {n}")


# NumPy数组转为张量

# In[16]:


n = np.ones(5)
t = torch.from_numpy(n)


# 同样地，`NumPy`数组的变化也会反映张量在中

# In[17]:


np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")


# ## 自动求导

# 在训练神经网络时，最常用的算法是反向传播。在该算法中，参数（模型权重）根据损失函数相对于给定参数的梯度进行调整。

# 为了计算这些梯度，PyTorch有一个叫做`torch.autograd`的内置微分引擎。它支持任何计算图的梯度自动计算。

# 为了方便演示，我们定义一个最简单的MLP网络。网络的计算图如下：

# ![simple_mlp](./images/simple_mlp.png)

# In[18]:


x = torch.ones(5)  # 输入的张量
y = torch.zeros(3)  # 期望的输出
w = torch.randn(5, 3, requires_grad=True) # 通过设置requires_grad=True，来指明需要优化的参数
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)


# In[19]:


print('Gradient function for z =',z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)


# **计算梯度**

# 为了优化神经网络中参数的权重，我们需要计算损失函数关于参数的导数
# 
# 调用`loss.backward()`方法，就能获取到每个张量的梯度

# In[20]:


loss.backward()
print(w.grad)
print(b.grad)


# **关闭梯度追踪**

# 默认情况，每个张量的`requires_grad`参数都会设置为`True`，来追踪计算历史。
# 
# 然而，有时候我们需要手动关闭梯度的更新。如当模型进行预测时，我们只需要计算网络的前传结果

# 可以通过`torch.no_grad()`块和张量的`detach()`方法来实现

# In[21]:


z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)


# In[22]:


z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)


# # 数据集

# 处理数据样本经常很混乱且难以复现。
# 
# PyTorch中提供了`torch.utils.data.DataLoader`和`torch.utils.data.Dataset`两个api，允许您使用预加载的数据集以及您自己的数据。
# 
# 其中，`Dataset`用来存储样本和标签，`DataLoder`将`Dataset`包装成一个可迭代对象，以便轻松遍历样本。

# ## 加载数据集

# In[23]:


import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt


# 以下是如何从`TorchVision`加载`Fashion-MNIST`数据集的示例
# 
# 该数据集包含60000张训练图片和10000张测试图片，每张图片是28*28大小的灰度图像，并来自其中的10个标签之一

# **数据转换函数**

# `ToTensor`将`PIL`图像或`NumPy ndarray`转换为`FloatTensor`，并在 [0., 1.] 范围内缩放图像的像素强度值

# 你也可以自己定义Lambda函数作为转换函数。如下面的代码中，将标签转换为One-Hot格式

# In[24]:


target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

training_data = datasets.FashionMNIST(
    root="data", # 数据存放的根目录
    train=True, # 是否为训练训练数据
    download=True, # 若本地路径不存是否进行下载
    transform=ToTensor() # 指定特征和标签的数据转换方式
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=target_transform
)


# ## 遍历和查看数据集

# 和访问Python的List一样，使用index直接可以拿到对应的数据

# In[13]:


labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}


# In[26]:


figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3  # 设定画布中的图片个数为3*3

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()  # 随机挑选一张图片
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


# ## 创建你自己的数据集

# 你需要重构以下三个成员函数：
# 
# 1. `__init__`：根据每个数据集的不同，初始化一些成员变量，如数据路径，标签的映射字典，数据转换函数等
# 2. `__len__`：返回我们数据集中的样本数
# 3. `__getitem__`：重写数据集的读取方式

# In[27]:


import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file) # 标签文件
        self.img_dir = img_dir # 数据集路径
        self.transform = transform # 特征转换函数
        self.target_transform = target_transform # 标签转换函数

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# ## 实例化DataLoader

# 推荐使用DataLoader用于模型的训练，而不是自己编写数据迭代函数
# 
# DataLoader的运行机制：保持数据同步，让GPU能够满血输出，而不是等待CPU处理好某一个batch的数据再进行计算

# In[28]:


from torch.utils.data import DataLoader

# 这里传入定义好的Dataset对象，并告诉dataloader每一次产生的数据个数为64
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True) # 训练集保持shuffle开启，防止模型过拟合
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)


# ## 遍历DataLoader

# In[29]:


# 每一次运行都会消耗掉64张图片
train_features, train_labels = next(iter(train_dataloader))
print(f"特征形状: {train_features.size()}")
print(f"标签形状: {train_labels.size()}")

img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"标签: {label}-{labels_map[int(label.numpy())]}")


# In[30]:


test_features, test_labels = next(iter(test_dataloader))
print(f"特征形状: {test_features.size()}")
print(f"标签形状: {test_labels.size()}")

img = test_features[0].squeeze()
label = test_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"标签: {label.numpy()}")


# # 搭建神经网络

# 神经网络由对数据执行操作的层或模块组成。
# 
# `torch.nn`命名空间提供了构建自己的神经网络所需的所有构建块。PyTorch中的每个模块都是`nn.Module`的子类。

# 接下来的代码示例将继续以`Fashion-MNIST`数据集为例

# In[31]:


import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# In[32]:


# 获取设备，如果没有gpu，会默认使用cpu

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


# ## 定义网络

# 我们通过继承`nn.Module`来定义我们的神经网络，并在`__init__`中初始化神经网络层。
# 
# 每个`nn.Module`子类都在`forward`方法中实现对输入数据的操作。

# In[33]:


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten() # 样本的输入形状为 1*28*28，这里对样本特征进行展开
        # 然后连接多层MLP+ReLU的组合层
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        # x表示输入的样本，这里实现网络的前传计算逻辑
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# In[34]:


model = NeuralNetwork().to(device)
print(model)


# 将输入数据传递给模型就可获取到模型的前传结果，不需要显式地调用`forward`方法

# In[35]:


X = torch.rand(1, 28, 28, device=device)
logits = model(X)


# 调用模型会返回一个 10 维张量，其中包含每个类的原始预测值。
# 
# 我们通过将其传递给`nn.Softmax`模块的实例来获得预测概率。

# In[36]:


print(f"预测结果形状：{logits.shape}")
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"预测类别: {y_pred}")


# ## 模型的层

# 接下来我们对`FashionMNIST`模型中的层进行分解。我们将取一个由3张大小为28x28的图像组成的小批量样本，看看当我们通过网络传递它时会发生什么

# In[37]:


input_image = torch.rand(3,28,28)
print(input_image.size())


# ### nn.Flatten

# 我们初始化`nn.Flatten`层以将每个28x28大小的2D图像转换为784个像素值的连续数组（并保持小批量维度（dim=0））

# In[38]:


flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())


# ### nn.Linear

# 线性层使用其存储的权重和偏差对输入应用线性变换

# In[39]:


layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())


# ### nn.ReLU

# 非线性激活是在模型的输入和输出之间创建出复杂映射的原因。它们在线性变换之后被应用以引入非线性，帮助神经网络学习各种各样的数据分布。

# 这里，我们在线性层间使用`nn.ReLU`作为激活函数

# In[40]:


print(f"应用ReLU前: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"应用ReLU后: {hidden1}")


# ### nn.Sequential

# `nn.Sequential`是一个有序的模块容器。数据按照定义的顺序通过所有模块。
# 
# 这是编写我们的神经网络的一种较为简单的方法。

# In[41]:


seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)


# ### nn.Softmax

# 网络的最后一个线性层返回`logits`中的原始值，然后传递给`nn.Softmax`模块
# 
# `logits`被缩放到[0, 1]区间，代表模型对每个类的预测概率。
# 
# `dim`参数表示所选维度的值的总和必须为`1`。

# In[42]:


softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print(pred_probab)


# ## 模型参数

# In[43]:


print("模型结构: ", model, "\n\n")

for name, param in model.named_parameters():
    print(f"网络层: {name} | 形状: {param.size()} | 值: {param[:2]} \n")


# # 模型训练

# 我们继续使用前面章节的数据加载和构建模型的代码

# In[44]:


import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
print(model)


# ## 超参数

# 超参数是可调节的参数，可让您控制模型的优化过程。不同的超参数值会影响模型训练和收敛速度

# 常见的训练超参数有：
# 
# * 迭代轮数（epoch）：完整遍历数据集的次数
# * 批大小（batch size）：在参数更新之前通过网络传播的数据样本的数量
# * 学习率（learning rate）：在每个batch或epoch更新模型参数的比例。较小的值会导致学习速度变慢，而较大的值可能会导致训练过程中出现不可预测的行为（如梯度消失）。

# In[45]:


learning_rate = 1e-3
batch_size = 64
epochs = 5


# ## 优化过程

# 每一轮迭代包含两个部分：
# * 训练阶段：迭代训练数据集并尝试收敛到最佳参数
# * 评估/测试阶段：迭代测试数据集以检查模型性能是否有所改善

# **损失函数**

# 损失函数衡量得到的结果与目标值的不相似程度，是我们在训练过程中想要最小化的损失函数。
# 
# 为了计算损失，我们使用给定数据样本的输入进行预测，并将其与真实数据标签值进行比较。

# 常见的损失函数有：
# * `nn.MSELoss`：均方误差，适合回归任务
# * `nn.NLLLoss`：负对数似然，适合分类任务
# * `nn.CrossEntropyLoss`：交叉熵，结合了`nn.NLLLoss`和`nn.LogSoftmax`

# In[46]:


loss_fn = nn.CrossEntropyLoss()


# **优化器**

# 优化是在每个训练步骤中调整模型参数以减少模型误差的过程。而优化算法定义了这个过程是如何执行的。
# 
# 最常见的优化算法是随机梯度下降（SGD），还有Adam和RMSProp，分别适用不同的模型和数据

# 我们通过注册模型需要训练的参数并传入学习率超参数来初始化优化器。

# In[47]:


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# ## 完整实现

# 训练过程中，优化过程主要有三件事需要完成：
# * 调用`optimizer.zero_grad()`来重置模型参数的梯度。默认情况下渐变相加；为了防止重复计算，我们在每次迭代时明确地将它们归零
# * 通过调用`loss.backwards()`来反向传播计算出损失。`PyTorch`会将损失的梯度存入每个需要计算梯度的参数。
# * 调用`optimizer.step()`来通过反向传播中收集的梯度来调整参数。

# In[48]:


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # 计算预测结果和损失
        pred = model(X)
        loss = loss_fn(pred, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"损失: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # 测试阶段，梯度不需要被更新
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"测试集误差: \n 准确率: {(100*correct):>0.1f}%, 平均损失: {test_loss:>8f} \n")


# In[49]:


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"迭代轮数 {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("完成!")


# 多层MLP的模型最终取得58%左右的分类准确率

# # 模型可视化

# 早期，PyTorch用户都比较羡慕Tensorflow用户可以使用Tensorboard可视化模型的训练进程。现在PyTorch与TensorBoard集成了，TensorBoard 是一种用于可视化神经网络训练运行结果的工具。

# 现在我们将设置TensorBoard，从`torch.utils`导入`tensorboard`并定义一个`SummaryWriter`，这是我们将信息写入TensorBoard的关键对象。

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor


# In[2]:


# 默认`log_dir`是"runs"目录 - 我们这里设置的更具体些
writer = SummaryWriter('runs/fashion_mnist_experiment_1')


# 与之间的模型不同的是，这里我们定义一个更适合处理图像数据的卷积模型——第一代卷积网络LeNet5，来看看这个网络的结构

# ![alexnet](./images/alexnet.png)

# 使用`nn.Conv2d`构建卷积层，其输入参数为：
# 
# * 输入通道数：`input_channels`，即输入的图像通道数
# * 输出通道数：`out_channels`，即特征图的个数
# * 卷积核大小：`kernel_size`，即卷积核的长和宽
# 
# 使用`nn.MaxPool2d`构建池化层，其输入参数为：
# 
# * 窗口大小：`kernel_size`，应用最大值计算的窗口大小
# * 步长：`stride`，每次窗口移动的步长，默认和窗口大小一致

# In[6]:


# 完全按照图中的模型参数设置

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 输入形状：batch_size*28*28，后面注释默认省去batch_size
        # 第一次卷积池化
        # 形状变化：28*28 -> 6*24*24 -> 6*12*12
        x = self.pool(F.relu(self.conv1(x))) 
        # 第二次卷积池化
        # 形状变化：6*12*12 -> 16*8*8 -> 16*4*4 
        x = self.pool(F.relu(self.conv2(x)))
        # 展开摊平
        # 形状变化：(16, 4, 4) -> (256,)
        x = x.view(-1, 16 * 4 * 4)
        # 连接两层线性层
        # (256,) -> (256, 120) -> (120, 84)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 输出层
        # (120, 84) -> (84, 10)
        x = self.fc3(x)
        return x

model = LeNet5()
print(model)


# ## 写入Tensorboard

# 我们现在来写入一些图片到Tensorboard中

# In[7]:


# 定义数据迭代器
batch_size = 64

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


# In[8]:


# 随机获取一些训练图片
dataiter = iter(train_dataloader)
images, labels = dataiter.next()

# 创建图片的网格（多张图片组合在一起）
img_grid = torchvision.utils.make_grid(images)

# 写入Tensorboard
writer.add_image('64_fashion_mnist_images', img_grid)


# 通过在命令行敲入命令`tensorboard --logdir=runs`就可以看到在本地浏览器中打开tensorboard了

# 然而，这个例子其实可以在Jupyter Notebook中完成，TensorBoard真正擅长的是创建交互式可视化

# ## 使用TensorBoard检查模型

# TensorBoard的优势之一是其可视化复杂模型结构的能力。我们来可视化我们之前构建的模型

# In[9]:


writer.add_graph(model, images)


# 继续并双击`NeuralNetwork`以查看其展开，查看构成模型的各个操作的详细视图

# ## 追踪模型训练

# 在之前的例子中，我们每100个批次打印一次模型的训练损失。我们现在使用Tensorboard来替代手动打印loss

# In[10]:


def matplotlib_imshow(img):
    """
    定义一个显示图片的函数
    """
    img = img.mean(dim=0)
    npimg = img.numpy()
    plt.imshow(npimg, cmap="Greys")


def images_to_probs(net, images):
    """
    给一个训练好的模型传入一组图片数据，产生预测结果和对应的预测概率
    """
    output = net(images)
    # 转化输出概率为对应的类别
    preds_tensor = torch.argmax(output, dim=1)
    preds = np.squeeze(preds_tensor.numpy())
    probs = [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]
    return preds, probs


def plot_classes_preds(net, images, labels):
    """
    使用训练好的网络生成带有网络预测最有可能的标签和真实标签的图片，并以不同颜色表示预测和真实标签的结果
    这里我们使用images_to_probs函数来计算得到预测标签和预测概率
    """
    preds, probs = images_to_probs(net, images)
    # 画出这个batch中的一些图片，并展示它们的预测标签和真实标签
    # 这里只展示其中的4张图片
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx])
        ax.set_title(
            "{0}, {1:.1f}%\n(label: {2})".format(
                labels_map[preds[idx]], probs[idx] * 100.0, labels_map[labels[idx]]),
            color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


# 这里我们对先前定义的`train_loop`函数和`test_loop`稍作改动
# 
# 1. 使用`add_scalar`方法把每100个batch将结果写入tensorboard中
# 2. 使用`add_image`方法可视化一些图片的预测和真实结果

# In[11]:


def train_loop(train_iter, test_iter, model, loss_fn, optimizer):
    size = len(train_iter.dataset)
    running_loss = 0.0
    
    for epoch in range(10):
        for batch, (images, labels) in enumerate(train_iter):
            # 将参数梯度归零
            optimizer.zero_grad()
            
            # 计算预测结果和损失
            preds = model(images)
            loss = loss_fn(preds, labels)

            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 记录每一个batch的loss
            running_loss += loss.item()
            
            if batch % 100 == 0:  # 每一百个batch记录一次
                # 当前是第几个batch
                global_steps = epoch * len(train_iter) + batch
                print(f"Epoch {epoch}: [{batch * batch_size}/{size}]")
                
                # 写入训练损失，这里是记录每100个batch的平均损失
                writer.add_scalar("训练损失", running_loss / 100, global_steps)
                
                # 同时写入一些图片展示模型的预测结果
                writer.add_figure("预测标签 vs. 真实标签",
                                  plot_classes_preds(model, images, labels.numpy()),
                                  global_step=global_steps)
                # 将累计损失重置为0
                running_loss = 0.0
                
                # 评估模型
                test_loop(test_iter, model, loss_fn, global_steps)
                

def test_loop(dataloader, model, loss_fn, global_steps):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # 测试阶段，梯度不需要被更新
    with torch.no_grad():
        for images, labels in dataloader:
            pred = model(images)
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    
    # 写入测试集准确率
    writer.add_scalar("测试损失", test_loss, global_steps)
    writer.add_scalar("测试准确", correct, global_steps)


# In[14]:


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_loop(train_dataloader, test_dataloader, model, loss_fn, optimizer)


# ![alexnet_acc](./images/alexnet_acc.png)

# 我们可以看到AlexNet提升明显，平均准确率来到了82.71%

# # 模型保存和加载

# In[48]:


import torch
import torch.onnx as onnx
import torchvision.models as models


# ## 保存和加载模型权重

# PyTorch模型将学习到的参数存储在名为`state_dict`的内部状态字典中。调用`save`方法实现模型持久化

# In[49]:


model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), "model_weights.pth")


# 要加载模型权重，您需要先创建相同模型的实例，然后使用`load_state_dict()`方法加载参数。

# In[50]:


model = models.vgg16() # 没有明确pretrained=True, 这样只会初始化一个模型实例
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()


# ## 保存完整的模型

# 只保存权重在使用的时候会不太方便，我们可能希望将此类的结构与模型一起保存，在这种情况下，我们可以将模型直接传递给`save`函数

# In[51]:


torch.save(model, 'model.pth')


# 加载模型也十分方便

# In[52]:


model = torch.load('model.pth')


# ## 导出模型为ONNX格式

# Pytorch支持原生的ONNX模型的导出。
# 
# 鉴于PyTorch执行图的动态特性，导出过程必须遍历执行图以生成持久化的ONNX模型
# 
# 因此，需要传递一个符合模型输入的测试变量给到导出函数

# In[53]:


input_image = torch.zeros((1,3,224,224))
onnx.export(model, input_image, 'model.onnx')


# 您可以使用ONNX模型做很多事情，包括在不同平台和不同编程语言上运行推理。

# ## 代码实战

# In[18]:


import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Lambda


# In[16]:


train_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)


# In[20]:


target_transform = Lambda(
    lambda y: torch.zeros(
        10,
        dtype=torch.float
    ).scatter_(
        dim=0,
        index=torch.tensor(y),
        value=1
    )
)

test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=target_transform
)


# In[29]:


train_data.shape


# In[30]:


figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(
        len(train_data),
        size=(1,)
    ).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(
        img.squeeze(),
        cmap="gray"
    )
plt.show()

