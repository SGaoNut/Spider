#!/usr/bin/env python
# coding: utf-8

# ## Tensorflow 2版本

# ![title](./img/2.png)

# ### 既然出了2版本跟1.x相比有哪些优势呢？
# 
# - 2版本并不是难度增大，而是大大简化建模的方法和步骤，与1.X版本相比更简单实用了，难度更小了！
# 
# - 终于把Keras Api当做核心了，使用起来一句话概述就是简单自己给自己打造了一扇简单的门。。。
# 
# - eager mode使得调试起来终于不那么难受了！
# 

# ![title](./img/1.png)

# ## Tensorflow 2版本安装

# #### 1.安装Anaconda：https://www.anaconda.com/
# - 这个是Python全家桶，必备！选择跟自己环境配套的最新版本就可以
# - 网速慢的点这里：https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/

# #### 2.安装Tensorflow: https://tensorflow.google.cn/
# - CPU版本安装：pip install tensorflow
# - GPU版本安装：pip install tensorflow-gpu 需要配置好CUDA10版本
# - 如果安装报错了，那就自己下载一份：
# https://pypi.org/project/tensorflow/#files
# https://pypi.org/project/tensorflow-gpu/#files
# 

# #### 3.创建虚拟环境
# - 创建虚拟换 conda create -n python3.8_tf2.4gpu python=3.8
# - 激活虚拟环境 conda activate python3.8_tf2.4gpu
# - 安装ipykernel conda install ipykernel
# - 将虚拟环境加入notebook显示中 python -m ipykernel install --user --name python3.8_tf2.4gpu --display-name python3.8_tf2.4gpu

# ### 新特性快速入门

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu


# In[2]:


# 导入tensorflow 包 as 别名函数
import tensorflow as tf

# python走向大众的一个关键包 numpy
import numpy as np


# In[3]:


# 从本机中查看物理机上是否有可访问的GPU
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    gpu0 = gpus[0] #如果有多个GPU，仅使用第0个GPU
    
    tf.config.experimental.set_memory_growth(gpu0, True) #设置GPU显存用量按需使用
    
    tf.config.set_visible_devices([gpu0], "GPU") 
    
print(gpus)


# ### tf 1.X  到 tf2 的蜕变

# <img src="./img/8.png" alt="FAO" width="500">
# <img src="./img/9.png" alt="FAO" width="500">

# #### tensorflow api文档
# - https://tensorflow.google.cn/api_docs/python/tf
# 

# In[4]:


tf.__version__


# - https://tensorflow.google.cn/api_docs/python/tf/optimizers

# - tf.optimizers.Adadelta
# - tf.optimizers.Adagrad
# - tf.optimizers.Adam

# Oh My God，能直接得出结果！！！

# In[5]:


# Eager Execution
# 创建一个变量
x = [[2., 5], 
     [2., 5]]

# 进行乘法操作 x * x
m = tf.matmul(x, x)
print(m)


# In[6]:


# 创建一个常数变量
x = tf.constant([[1,9],[3,6]])
x


# In[7]:


# 加法操作
x = tf.add(x, 1)
x


# ### Tensor是什么

# - 当做是可以进行GPU加速计算的矩阵就可以

# 转换格式

# In[8]:


# 将数据转换成numpy
x.numpy()


# In[9]:


# 对变量进行类型转换
x = tf.cast(x, tf.float32)
x


# In[10]:


# 对每个元素进行相乘
x1 = np.ones([3,3])
x2 = tf.multiply(x1, 2)
x2


# ##  神经网络全连接案例实战
# - 第一层为输入层
# - 第二层 至 第四层 是隐藏层
# - 最后一层是输出层

# <img src="./img/7.jpg" alt="FAO" width="790">

# ### 回归问题预测
# - Tensorflow2版本中将大量使用keras的简介建模方法
# 

# In[11]:


# 算法处理的三个基础包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras
import warnings
import random

# 忽略警告信息
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# #### 读取数据

# In[12]:


features = pd.read_csv('temps.csv')


# #### 看看数据长什么样子

# In[13]:


features.head()


# 数据表中
# * year,moth,day,week分别表示的具体的时间
# * temp_2：前天的最高温度值
# * temp_1：昨天的最高温度值
# * average：在历史中，每年这一天的平均最高温度值
# * actual：这就是我们的标签值了，当天的真实最高温度
# * friend：这一列是普通人随机猜测的结果，你的朋友猜测的可能值，用来对比模型结果和人们瞎猜的差异，也是所谓的Baseline

# #### 查看数据维度 

# In[14]:


print('数据维度:', features.shape)


# #### 数据预处理

# In[15]:


# 处理时间数据
import datetime

# 分别得到年，月，日
years = features['year']
months = features['month']
days = features['day']

# datetime格式
# 1、组合时间数据 year-month-day
# 2、讲字符串转换成时间类型的函数
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]


# In[16]:


dates[:5]


# In[17]:


features.head()


# ### 可视化数据分析

# #### 设置布局
# - fig 表示一个整体
# - ax1,ax2,ax3,ax4 分别是我们的四个子
# - nrows表示我们的行数，nclos表示我们的列数
# - figsize表示我们每个图的整体大小

# In[18]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize = (20,20))

# 统一横坐标时间
fig.autofmt_xdate(rotation = 45)

# 标签值
ax1.plot(dates, features['actual'])
ax1.set_xlabel('')
ax1.set_ylabel('Temperature')
ax1.set_title('Max Temp')

# 添加网格信息
ax1.grid(True, linestyle="--", alpha=0.5)

# 昨天
ax2.plot(dates, features['temp_1'])
ax2.set_xlabel('')
ax2.set_ylabel('Temperature')
ax2.set_title('Previous Max Temp')
ax2.grid(True, linestyle="--", alpha=0.5)

# 前天
ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date')
ax3.set_ylabel('Temperature')
ax3.set_title('Two Days Prior Max Temp')
ax3.grid(True, linestyle="--", alpha=0.5)

# 随即猜测的结果
ax4.plot(dates, features['friend'])
ax4.set_xlabel('Date')
ax4.set_ylabel('Temperature')
ax4.set_title('Friend Estimate')
ax4.grid(True, linestyle="--", alpha=0.5)


plt.tight_layout(pad=2)


# ### 折线图
# - 显示数据变化趋势，反应事务的变化情况 
# - plt.plot(x, y)
# 
# ### 散点图
# - 判断变量之间是否存在关联趋势，展示离群点（分布规律
# - plt.scatter(x, y)
# 
# ### 柱状图
# - 绘制离散的数据，能够一眼看出各个数据的大小，比较数据之间的差别。（统计和对比）
# - plt.bar(x, width, align='center', color='r')
# 

# In[19]:


# 原始结果
features.head()


# In[20]:


# 独热编码
one_hot_features = pd.get_dummies(features)
one_hot_features.head(5)


# In[21]:


# 标签
labels = np.array(one_hot_features['actual'])
friend = np.array(one_hot_features['friend'])

# 在特征中去掉标签
x_features= one_hot_features.drop(['actual', 'friend'], axis = 1)

# 特征名称单独保存一下，以备后患
feature_list = list(x_features.columns)


# In[22]:


feature_list


# In[23]:


# 转换成合适的格式
x_features = np.array(x_features)


# In[24]:


x_features.shape


# ### 数据预处理比较常用的包
# 
# - https://scikit-learn.org/0.16/modules/classes.html#module-sklearn.preprocessing

# In[25]:


from sklearn import preprocessing

# 数据的标准化操作，对数字进行缩放
input_features = preprocessing.StandardScaler().fit_transform(x_features)


# 作用：取均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本。 
# 
# 标准差标准化（standardScale）使得经过处理的数据符合标准正态分布，即均值为0，标准差为1，其转化函数为：
# 
# <!-- <img src="./img/StandardScaler.png" alt="FAO" width="790"> -->
# ![title](./img/StandardScaler.png)
# 
# 其中μ为所有样本数据的均值，σ为所有样本数据的标准差。

# In[26]:


print(feature_list)
print(x_features[0].tolist())


# ### 对比特征转换后的结果
# - 重要知识点: batch normalization的思想来源

# In[27]:


input_features[0]


# ## 开始构建模型

# ### 基于Keras构建全连接网络模型
# 一些常用参数已经列出，如下所示：

# - activation：激活函数的选择，一般常用relu
# - kernel_initializer, bias_initializer：权重与偏置参数的初始化方法，有时候不收敛换种初始化就突然好使了。。。玄学
# - kernel_regularizer，bias_regularizer：要不要加入正则化，
# - inputs：输入，可以自己指定，也可以让网络自动选
# - units：神经元个数

# In[28]:


# 清空当前会话
from tensorflow.keras import backend as K
K.clear_session()

# 简单的序列化建模
model = tf.keras.Sequential()

# dense wx+b
model.add(layers.Dense(16))
model.add(layers.Dense(32))
model.add(layers.Dense(1))


# ### 模型编译
# 
# - 优化器api https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers
# - 优化器 SGD 梯度下降 学习率 0.001
# - 损失函数，mean_squared_error 均方差

# In[29]:


model.compile(optimizer=tf.keras.optimizers.SGD(0.001), loss='mean_squared_error')
# model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error')


# ### 模型训练
# - validation_split 验证集数据切分比率
# - epochs 多少轮次
# - batch_size 最小迭代的大小

# In[30]:


model.fit(input_features, labels, validation_split=0.25, epochs=10, batch_size=64)


# - 模型收敛出现问题

# ### 查看模型结构

# In[31]:


model.summary()


# ### 参数计算
# - wx+b
# - param input_features.shape[1] * 16 + 16

# In[32]:


13 * 16 + 16


# #### 加入初始化权重

# In[33]:


model = tf.keras.Sequential()
# tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
model.add(layers.Dense(16,kernel_initializer='random_normal'))
model.add(layers.Dense(32,kernel_initializer='random_normal'))
model.add(layers.Dense(1,kernel_initializer='random_normal'))
model.compile(optimizer=tf.keras.optimizers.SGD(0.001),
             loss='mean_squared_error')
# model.compile(optimizer=tf.keras.optimizers.Adam(0.011),
#              loss='mean_squared_error')
model.fit(input_features, labels, validation_split=0.25, epochs=10, batch_size=64)


# #### 加入正则惩罚项

# In[34]:


model = tf.keras.Sequential()
model.add(layers.Dense(16,kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.add(layers.Dense(32,kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.03)))
model.add(layers.Dense(1,kernel_initializer='random_normal',kernel_regularizer=tf.keras.regularizers.l2(0.03)))

model.compile(optimizer=tf.keras.optimizers.SGD(0.001),
             loss='mean_squared_error')
model.fit(input_features, labels, validation_split=0.25, epochs=20, batch_size=64)


# ### 预测模型结果

# In[35]:


predict = model.predict(input_features)


# ### 压缩维度 并且转换成列表

# In[36]:


output = predict.reshape(-1).tolist()


# In[37]:


list(zip(output, labels, friend))


# ### 测试结果并进行展示

# In[38]:


# 转换日期格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# 创建一个表格来存日期和其对应的标签数值
true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})

# 同理，再创建一个来存日期和其对应的模型预测值
months = x_features[:, feature_list.index('month')]
days = x_features[:, feature_list.index('day')]
years = x_features[:, feature_list.index('year')]

test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]

test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]

predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predict.reshape(-1)}) 


# In[39]:


# 真实值
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')

# 预测值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
plt.xticks(rotation = '60'); 
plt.legend()

# 设置图的标签名称
plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual and Predicted Values');


# In[40]:


# 转换日期格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]

# 创建一个表格来存日期和其对应的标签数值
true_data = pd.DataFrame(data = {'date': dates, 'actual': labels})

# 同理，再创建一个来存日期和其对应的模型预测值
months = x_features[:, feature_list.index('month')]
days = x_features[:, feature_list.index('day')]
years = x_features[:, feature_list.index('year')]

test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]

test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]

predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': friend}) 


# In[41]:


# 真实值
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')

# 预测值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
plt.xticks(rotation = '60'); 
plt.legend()

# 图名
plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual and Predicted Values');


# ### 神经网络碾压式的结果

# In[42]:


tf.losses.mean_squared_error(features['actual'], features['friend']).numpy()


# In[43]:


tf.losses.mean_squared_error(features['actual'], predict.reshape(-1)).numpy()


# #### 保存模型

# In[44]:


model.save('tmps.h5')


# #### 加载模型

# In[45]:


from tensorflow import keras


# In[46]:


model = keras.models.load_model('tmps.h5')


# In[ ]:




