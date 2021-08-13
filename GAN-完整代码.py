#!/usr/bin/env python
# coding: utf-8

# # Vanllia GAN

# ## GAN的定义

# 首先我们来介绍一部经典电影《猫鼠游戏》

# <div align=center><img src="./images/catch_me_if_you_can.png" alt="catch_me_if_you_can" style="width:400px;"/></div>

# 片中的莱昂纳多有高超的观察与模仿能力。无论是驾照上的出生年份、泛美航空的工作证还是各种支票、大学的毕业证书等，只需要让他观察一段时间，他就能用他的双手把它制作出来，而且达到以假乱真的地步。抓捕他的FBI探员一开始总是被骗，但是在这个过程中，探员鉴别假材料的技巧也越来越高，最终成功将他抓捕归案。

# 这个警察和假币犯互相斗智斗勇的过程与GAN的工作机制极其相似。

# 生成对抗网络（Genrative Adversarial Network，以下简称GAN）由Ian Goodfellow于2014年其读博期间发明，被Yann LeCun称赞为“20年来机器学习领域最酷的想法”。

# 论文链接：[Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)

# ## 训练过程

# GAN由生成器和判别器两部分组成。
# * 生成器学习生成合理的数据。生成的实例成为判别器的负训练实例。
# * 判别器学习区分生成器的假数据和真实数据。判别器惩罚生成器以产生以假乱真的结果。

# 当训练开始时，生成器产生明显的假数据，判别器很快学会辨别它是假的：

# ![gan_train_begin](./images/gan_train_begin.svg)

# 随着训练的进行，生成器越来越接近于产生可以欺骗判别器的输出：

# ![gan_train_mid](./images/gan_train_mid.svg)

# 最后，如果生成器训练顺利，判别器在区分真假时会变得更糟。它开始将虚假数据归类为真实数据，其准确性降低

# ![gan_train_end](./images/gan_train_end.svg)

# 再来举一个例子，这里生成器化身为“艺术家”，学习创造看起来真实的画作，而判别器转变为“艺术评论家”学习区分真假画作。

# ![cat1](./images/cat1.png)

# 训练过程中，生成器在生成逼真画作方面逐渐变强，而判别器在辨别这些作品的能力上也逐渐变强。当判别器不再能够区分真实画作和伪造的画作时，训练过程达到平衡。

# ![cat2](./images/cat2.png)

# 将上述两个例子抽象化，我们来看下GAN的整体结构：

# ![gan_diagram](./images/gan_diagram.svg)

# 生成器和判别器都是神经网络。生成器输出直接连接到判别器作为输入。
# 
# 通过反向传播，判别器的分类提供了生成器用来更新其权重的信号。

# 接下来我们以一个简单的实战案例，来详细阐述GAN的工作原理

# ## 代码实战——一元二次曲线生成

# In[1]:


import os
import datetime
import time
import PIL
from pathlib import Path

import imageio
import IPython
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import Image, HTML, display
from tensorflow import keras

plt.rcParams['axes.unicode_minus'] = False


# In[2]:


# 检测设备中显卡
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

tf.__version__


# ### 数据展示

# 我们使用生成一元二次方程的图像为例，作为我们GAN之旅的起点
# 
# 首先我们选择$x^2$为例，来搭建判别网络

# In[3]:


# 简单的一元二次方程x^2
def calculate(x):
    return x * x
 
# 定义输入
inputs = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]
# calculate outputs
outputs = [calculate(x) for x in inputs]
# plot the result 采用描点的方式画图
plt.plot(inputs, outputs)
plt.show()


# 我们可以使用这个函数为我们的判别器生成真实样本。每个样本包含两个元素，第一个为输入，第二个为方程$x^2$的结果

# In[4]:


# 从x^2生成n个随机样本
def generate_samples(n):
    # 随机生成在[-0.5, 0.5]内的点
    x1 = np.random.rand(n) - 0.5
    # 生成x^2方程的输出
    x2 = x1 * x1
    # 拼接数组
    x1 = x1.reshape(n, 1)
    x2 = x2.reshape(n, 1)
    X = np.hstack((x1, x2))
    return X

data = generate_samples(100)
plt.scatter(data[:, 0], data[:, 1])
plt.show()


# ### 判别器

# GAN中的判别器只是一个分类器。它试图将真实数据与生成器创建的数据区分开来。它可以使用适合其分类数据类型的任何网络架构。

# 判别器的训练数据来自两块：
# * 真实数据：如人的照片，判别器在训练中视它们为正例
# * 虚假数据：由生成器产生，判别器在训练中视它们为负例

# ![gan_diagram_discriminator](./images/gan_diagram_discriminator.svg)

# 由上图可知，两个`Sample`块表示输入给判别器的两种数据来源。判别器训练时，生成器不会进行训练。它在产生给判别器训练数据时，它的权重保持不变。

# 判别器的训练过程：
# 1. 判别器对真实数据和由生成器产生的虚假数据进行分类
# 2. 判别器的损失对判别器将真实实例错分为虚假实例或假实例为真实例进行惩罚
# 3. 判别器通过反向传播方式对判别器网络中的权重进行更新

# 我们使用`keras.Sequential`搭建一个简单的顺序网络作为判别器
# 
# 输入张量：方程值，形状：(batch_size, num_inputs)
# 
# 输出张量：预测概率，形状：(batch_size, 1)

# 由于，这里是一个判断真假的二分类任务，我们定义判别器的损失函数为交叉熵
# 
# 更多关于交叉熵的解释可以查看[这里](https://zhuanlan.zhihu.com/p/89391305)

# In[5]:


def define_discriminator(num_inputs=2):
    model = keras.Sequential(name="discriminator")
    model.add(keras.layers.Dense(32, activation="relu", input_dim=num_inputs))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


# In[6]:


discriminator = define_discriminator()
discriminator.summary()


# 现在我们来验证下判别器的有效性
# 
# 我们构建假数据生成器，与真实数据拼一起输入给网络，进行二分类任务训练，标签1表示真实数据，标签0表示虚假数据

# In[7]:


# 生成带有类别标签的n个真实样本
def generate_real_samples(n):
    # 随机生成在[-0.5, 0.5]内的点
    x1 = np.random.rand(n) - 0.5
    # 生成x^2方程的输出
    x2 = x1 * x1
    # 拼接数组
    x1 = x1.reshape(n, 1)
    x2 = x2.reshape(n, 1)
    X = np.hstack((x1, x2))
    # 生成类别标签
    y = np.ones((n, 1))
    return X, y
X, y = generate_real_samples(5)
print(X)
print(y)


# In[8]:


# 生成带有类别标签的n个假样本
def generate_fake_samples(n):
    # 随机生成在[-1, 1]内的点
    x1 = -1 + np.random.rand(n) * 2
    # 随机生成在[-1, 1]内的点
    x2 = -1 + np.random.rand(n) * 2
    # 拼接数组
    x1 = x1.reshape(n, 1)
    x2 = x2.reshape(n, 1)
    X = np.hstack((x1, x2))
    # 生成类别标签
    y = np.zeros((n, 1))
    return X, y
X, y = generate_fake_samples(5)
print(X)
print(y)


# 下面的`train_discriminator`函数实现如下功能：训练模型512个batch，每个batch使用128个样本，64个真样本，64个假样本
# 
# 我们这里不使用`fit()`这种深度封装方法训练模型，而是使用`train_on_batch`方法对模型进行训练，适合像GAN这样需要分步进行训练的模型

# In[9]:


# 训练判别器
def train_discriminator(model, num_batches=512, batch_size=128):
    half_batch = int(batch_size / 2)
    # 手动编写训练过程
    for i in range(num_batches):
        # 生成真实样本
        X_real, y_real = generate_real_samples(half_batch)
        # 更新模型
        model.train_on_batch(X_real, y_real)
        # 生成假样本
        X_fake, y_fake = generate_fake_samples(half_batch)
        # 更新模型
        model.train_on_batch(X_fake, y_fake)
        # 评估模型
        # verbose参数：0表示不打印日志信息，1表示输出进度条记录，2表示每个epoch输出一行记录
        # 另一个返回
        loss_real, acc_real = model.evaluate(X_real, y_real, verbose=0)
        loss_fake, acc_fake = model.evaluate(X_fake, y_fake, verbose=0)
        if i % 10 == 0:
            print(f"Batch {i}: acc_real - {acc_real}, acc_fake - {acc_fake}")


# In[10]:


# 定义判别器模型
model = define_discriminator(2)
# 训练模型
train_discriminator(model)


# 在这种情况下，模型可以快速学习以完美的准确率正确识别真样本，并且非常擅长以80%到90%的准确率识别假样本

# ### 生成器

# GAN的生成器部分通过结合来自判别器的反馈来学习创建假数据。它的学习目标就是让判别器将其输出分类为真实的。

# 生成器训练需要生成器和判别器之间的集成程度比判别器训练时更紧密。训练生成器的GAN部分包括：

# * 随机输入
# * 生成器网络，将随机输入转为一个数据示例
# * 判别器网络，对生成的数据进行分类
# * 判别器输出
# * 生成器损失函数，因为未能欺骗判别器而惩罚生成器

# ![gan_diagram_generator](./images/gan_diagram_generator.svg)

# 通常来说，GAN选择随机噪声数据作为输入。然后生成器将噪声转化为有意义的输出。
# 
# 实验表明，噪声数据的分布并不重要，我们可以选择一些容易采样的分布，如均匀分布。
# 
# 为方便起见，采样噪声的空间的维度通常小于输出空间的维度。

# 训练生成器的步骤如下：
# 1. 随机采样一些噪声
# 2. 根据噪声让生成器生成一些输出
# 3. 对生成器的数据，调用判别器进行真假分类
# 4. 计算判别器的损失
# 5. 通过判别器和生成器的反向传播获得梯度
# 6. 使用这些梯度只用来改变生成器的权重

# 输入张量：噪声空间的点，形状：(batch_size, latent_dim)
# 
# 输出张量：有意义的分布的点，形状：(batch_size, num_outputs)
# 
# 与判别器定义代码不同的是，这里我们也可以把定义的网络层放入一个列表中，一次性传给`Sequential`

# In[11]:


def define_generator(latent_dim, num_outputs=2):
    model = keras.Sequential(
        [
            keras.Input(shape=(latent_dim,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(num_outputs)
        ], name="generator")
    return model

generator = define_generator(5)
generator.summary()


# 在噪声空间中生成点作为生成器的输入 -> 生成器转换为有意义的数据分布

# In[12]:


def generate_latent_points(latent_dim, n):
    # 在噪声空间生成一些点
    x_input = np.random.randn(latent_dim * n)
    # 转换数据形状以适应网络输入形状
    x_input = x_input.reshape(n, latent_dim)
    return x_input


# In[13]:


def generate_fake_samples(generator, latent_dim, n):
    # 在噪声空间生成一些点
    x_input = generate_latent_points(latent_dim, n)
    # 转换为有意义的分布
    X = generator.predict(x_input)
    y = np.zeros((n, 1))
    return X, y


# 生成100个虚假数据，由于网络还未进行过训练，每次生成的数据分布都不太一样

# In[14]:


latent_dim = 5
model = define_generator(latent_dim)
X, _ = generate_fake_samples(generator, latent_dim, 100)
# 画出这些点
plt.scatter(X[:, 0], X[:, 1])
plt.show()


# 构建完整的GAN的生成器，输入首先经过生成器网络，然后再传递到判别器网络，判别器网络对生成器生成的输入进行判断

# In[15]:


def define_gan(generator, discriminator):
    # 设置判别器不进行训练（生成器阶段）
    discriminator.trainable = False
    # 连接判别器和生成器
    model = keras.Sequential()
    # 加入生成器
    model.add(generator)
    # 加入判别器
    model.add(discriminator)
    # 模型配置
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model    


# ### 模型训练

# **交替训练**
# 
# GAN训练过程是交替的：
# 1. 判别器训练1-N个epoch
# 2. 生成器训练1-N个epoch
# 3. 重复步骤1和2，不断交替训练判别器和生成器网络
# 
# 需要注意的是，
# 1. 判别器训练时生成器保持不变。它必须学会如何识别生成器的缺陷。
# 2. 同样的，生成器训练时判别器保持不变。否则，生成器会遇到不断变化的目标，且可能永远不会收敛
# 
# 如果没有强大的判别器的话，也不会存在与之匹敌的生成器。判别器如果弱到连初始化的生成器产生的输出也无法区分的话，GAN的训练是无法进行的。

# 定义评估函数

# In[16]:


# 评估判别器并绘制真假点
def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
    # 准备真实样样本
    x_real, y_real = generate_real_samples(n)
    # 用判别器对真实样本进行评估
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    # 准备虚假样本
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
    # 用判别器对虚假样本进行评估
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    print(epoch, acc_real, acc_fake)
    # 画出真实和虚假数据的图像
    plt.scatter(x_real[:, 0], x_real[:, 1], color='red')
    plt.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
    plt.show()


# In[17]:


def train(g_model, d_model, gan_model, latent_dim, num_steps=10000, batch_size=128, eval_steps=500):
    # batch大小的一半为真实样本，一半为虚假样本
    half_batch = int(batch_size / 2)
    for i in range(num_steps):
        # 准备真实样本
        x_real, y_real = generate_real_samples(half_batch)
        # 准备虚假样本
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # 更新判别器
        d_model.train_on_batch(x_real, y_real)
        d_model.train_on_batch(x_fake, y_fake)
        # 准备噪声数据作为生成器的输入
        x_gan = generate_latent_points(latent_dim, batch_size)
        # （重要）为假样本创建倒置标签，即告诉判别器这是真实数据
        # 诱导判别器给出“真数据”的梯度下降方向，让生成器知道有哪些不足
        y_gan = np.ones((batch_size, 1))
        # 通过判别器的误差来更新生成器
        gan_model.train_on_batch(x_gan, y_gan)
        # 每eval_steps个batch后评估下模型
        if (i + 1) % eval_steps == 0:
            summarize_performance(i, g_model, d_model, latent_dim)


# In[18]:


latent_dim = 5
num_inputs = 2

discriminator = define_discriminator(num_inputs)
generator = define_generator(latent_dim, num_inputs)
gan_model = define_gan(generator, discriminator)

train(generator, discriminator, gan_model, latent_dim)


# 我们可以看到，5000个batch后，GAN的生成器已经能够基本能够画出$x^2$的曲线了

# **收敛问题**
# 
# 随着生成器不断提升性能，判别器的性能会变差，因为它无法准确区分真实和虚假的样本。如果生成器达到完美状态，判别器的准确率来到50%，相当于掷硬币来决定真假。
# 
# 因为判别器得到的反馈越来越没有价值，GAN的收敛出现问题。如果GAN继续在判别器给出的无效反馈上训练的话，生成器开始陷入坍塌状态。

# ### 另一个例子——更复杂的曲线

# 选取一个稍微复杂一些的一元二次方程$ax^2+a-1$，并构造生成器来生成的方程y值

# #### 构造真实数据

# 这个例子的真实数据与上一个例子不太一样，每一个批次的数据中x的值是确定的，输出为不同形状的曲线的y的取值，意味着模型需要学习的数据变多了

# In[19]:


def get_real_data(data_dim, batch_size, num_batches):
    for i in range(num_batches): # 每一个epoch的batch数
        a = np.random.uniform(1, 2, size=batch_size)[:, np.newaxis] # 改变不同a的值，我们就能获得不同的曲线样式
        x = np.linspace(-1, 1, data_dim)[np.newaxis, :].repeat(batch_size, axis=0) # x的取值范围为[0, data_dim - 1]
        yield a * np.power(x, 2) + (a - 1)

# 图示方程曲线
for data in get_real_data(8, 4, 1):
    print(data.shape)
plt.plot(data.T)


# #### 使用Model类来定义GAN

# 这里我们将模型的定义、配置、梯度更新过程全部放入`Model`类中，定义一个完整的GAN网络，使得我们的代码看上去更整洁和统一。

# In[20]:


class GAN(keras.Model):
    def __init__(self, data_dim, latent_dim):
        super(GAN, self).__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim        
        self.discriminator = self.get_discriminator()
        self.generator = self.get_generator()
    
    def compile(self, d_optimizer, g_optimizer, loss_fn, metric_fn):
        # 配置用于训练的模型
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.d_loss = keras.metrics.Mean(name="d_loss")  # 计算给定值的平均值，每个batch更新一次
        self.g_loss = keras.metrics.Mean(name="g_loss")
        self.d_acc = keras.metrics.Mean(name="d_acc")
        self.g_acc = keras.metrics.Mean(name="g_acc")

    @property
    def metrics(self):
        return [self.d_loss, self.g_loss, self.d_acc, self.g_acc]

    def call(self, n):
        # 根据输入获取模型前传结果
        return self.generator(tf.random.normal(shape=(n, self.latent_dim)))

    def get_generator(self):
        # 定义生成器
        # 生成器的模型结构为：输入层 -> 全连接层 -> 输出层
        # 使用`keras.Sequential`来构建模型
        # 输入层：使用`keras.Input`，输入维度是多少呢？
        # 全连接层：使用`keras.layers.Dense`，输出维度为32，激活函数为relu
        # 输出层：使用`keras.layers.Dense`，输出维度是多少呢？
        # 请在此写下你的代码
        model = keras.Sequential([
            keras.Input(shape=(self.latent_dim,)),
            keras.layers.Dense(32, activation=keras.activations.relu),
            keras.layers.Dense(self.data_dim),
        ], name="generator")
        model.summary()
        return model

    def get_discriminator(self):
        # 定义判别器
        # 判别器的模型结构也是：输入层 -> 全连接层 -> 输出层
        # 同样使用`keras.Sequential`来构建模型
        # 输入层：使用`keras.Input`，输入维度是多少呢？
        # 全连接层：使用`keras.layers.Dense`，输出维度为32，激活函数为relu
        # 输出层：使用`keras.layers.Dense`，输出维度是多少呢？激活函数为sigmoid
        # 请在此写下你的代码
        model = keras.Sequential([
            keras.Input(shape=(self.data_dim,)),
            keras.layers.Dense(32, activation=keras.activations.relu),
            keras.layers.Dense(1, activation="sigmoid")
        ], name="discriminator")
        model.summary()
        return model

    def train_step(self, real_curves):
        # 从潜在空间生成一些随机的噪声
        batch_size = tf.shape(real_curves)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        # 编码生成假的曲线      
        generated_curves = self.generator(random_latent_vectors)
        
        # 与真实曲线合并为同一批数据
        combined_curves = tf.concat([generated_curves, real_curves], axis=0)
        
        # 设置真曲线和假曲线的标签
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
        
        # 训练判别器
        # 这里是tf2特有的写法
        with tf.GradientTape() as tape:
            # 获取判别器对`combined_curves`的预测结果
            # 并计算与真实标签`labels`的损失d_loss
            # 在这里写下你的代码
            predictions = self.discriminator(combined_curves)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )
        d_acc = self.metric_fn(labels, predictions)
        
        # 从潜在空间生成一些随机的噪声
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        # 设置这些曲线为“真曲线”标签
        misleading_labels = tf.zeros((batch_size, 1))
        
        # 训练生成器（注意：我们这时候不应该再更新判别器的梯度）
        with tf.GradientTape() as tape:
            # 获取判别器对生成器生成的曲线的预测结果
            # 并计算与“真曲线”标签`misleading_labels`的损失g_loss
            # 在这里写下你的代码
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        # （重要）这里只更新生成器的参数
        # 获取生成器每个参数的梯度
        # 将梯度应用于对应的参数上
        # 在这里写下你的代码
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights)
        )
        g_acc = self.metric_fn(misleading_labels, predictions)
        
        # 更新评估指标
        self.d_loss.update_state(d_loss)
        self.d_acc.update_state(d_acc)
        self.g_loss.update_state(g_loss)
        self.g_acc.update_state(g_acc)
        return {
            "d_loss": self.d_loss.result(),
            "d_acc": self.d_acc.result(),
            "g_loss": self.g_loss.result(),
            "g_acc": self.g_acc.result()
        }


# #### 初始化模型实例

# 由于我们需要分别训练两个网络，判别器和生成器的优化器是不同的。

# In[21]:


# 模型实例化
latent_dim = 10
data_dim = 16

gan = GAN(data_dim, latent_dim)
# 定义优化器，loss函数，评估函数
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=keras.losses.BinaryCrossentropy(),
    metric_fn=keras.metrics.BinaryAccuracy()
)


# #### 结果展示

# * 每一个epoch保存最新的曲线图像
# * 训练结束将每一个epoch图像合成一张gif动图

# In[22]:


image_folder = Path("gan_gen_images/curve")
image_folder.mkdir(parents=True, exist_ok=True)

def save_curve_per_epoch(epoch, model, num_curves=5): 
    data = model(num_curves, training=False).numpy()
    plt.plot(data.T)
    plt.xticks((), ())
    plt.title(f"Epoch {epoch}")
    plt.savefig(f"{image_folder}/{epoch}.png")
    plt.close()

def gen_gif(shrink=5):
    imgs = []
    for img_fn in sorted(Path(image_folder).glob("*.png"), key=os.path.getmtime):
        img = PIL.Image.open(img_fn)
        img = img.resize((img.width // shrink, img.height // shrink), PIL.Image.ANTIALIAS)
        imgs.append(img)
    gif_fn = f"{image_folder}/curve.gif"
    imgs[0].save(gif_fn, append_images=imgs, optimize=False, save_all=True, duration=400, loop=0)


# #### 模型训练

# In[23]:


epochs = 30 # 训练轮数
steps_per_epoch = 256 # 每轮的batch数
batch_size = 32 # 每个batch的样本数
num_inputs = 16 # 每个样本的y值个数

t0 = time.time()
for i in range(epochs):
    for t, data in enumerate(get_real_data(num_inputs, batch_size, steps_per_epoch)):
        metrics = gan.train_step(data)
        d_loss, d_acc, g_loss, g_acc = metrics["d_loss"], metrics["d_acc"], metrics["g_loss"], metrics["g_acc"]
        if t % 128 == 0:
            t1 = time.time()
            print(
                  "epoch={} step={} | time={:.1f} | t={} | d_acc={:.2f} | g_acc={:.2f} | d_loss={:.2f} | g_loss={:.2f}".format(
                      i, t, t1 - t0, t, d_acc, g_acc, d_loss, g_loss))
            t0 = t1
    save_curve_per_epoch(i, gan)
gen_gif(1)


# Epoch 0 - Epoch 29 的曲线变化

# ![curve](./gan_gen_images/curve/curve.gif)

# ### 改进点

# 1. 尝试不同的方程曲线，如高斯分布
# 2. 使用更复杂的网络结构，如更深的网络
# 3. 尝试不同的激活函数

# # DCGAN

# 我们知道深度学习中对图像处理应用最好的模型是CNN，那么如何把CNN与GAN结合？DCGAN是这方面最好的尝试之一
# 
# 论文链接：[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
# 
# DCGAN，全称深度卷积生成对抗网络（Deep Convolutional Generative Adversarial Network）

# ## 与GAN的差异

# DCGAN与GAN的原理是一样的。主要变动如下：
# 1. 将生成器和判别器更换为更适合表征图片的卷积神经网络
# 2. 对CNN的结构做了一些改变，以提高样本的质量和收敛的速度
# 
#     * 取消池化层（pooling）。生成器网络中使用微步幅卷积进行上采样，判别器网络中用卷积层替代全连接层
#     * 生成器和判别器网络中均采用批量归一化技巧（Batch normalization）
#     * 生成器和判别器均去掉全连接层，网络变为全卷积网络
#     * 生成器网络中使用ReLU作为激活函数，最后一层使用tanh
#     * 判别器网络中使用LeakyReLU作为激活函数

# ### 反卷积

# 直观的理解的话，
# 
# * 卷积：图片 -> 特征向量
# * 反卷积：特征向量（噪声）-> 图片
# 
# 具体是如何实现的呢？

# <table><tr><td><img src='./images/conv.gif' style=width:400px;></td><td><img src='./images/deconv.gif' style=width:600px;></td></tr></table>

# 正常的卷积操作，是将上面大的图片信息抽取到下面小的特征图中。而反卷积则是将下面的特征当作输入，输出为上面的特征图

# 反卷积有分为转置卷积和微步卷积，两者的区别在于padding的方式不同，原文中强调使用的是微步卷积

# <table><tr><td><img src='./images/transpose_conv.gif' style=width:400px;></td><td><img src='./images/stride_conv.gif' style=width:400px;></td></tr></table>

# ### 生成器网络

# ![transpose_conv](./images/transpose_conv.png)

# 上图为DCGAN生成器的工作原理图示
# 
# 100维的噪声输入首先通过全连接，再reshape成一个三维张量，图中变为 $4*4*1024$ 大小的张量
# 
# 然后，经过三次的的上采样和一次卷积操作后得到的一张 $64*64*3$ 大小的一张图片

# 论文中并没有采用Pooling层
# 
# 经典CNN中，我们常用pooling来进行信息筛减。
# 
# 但是对于生成中，需要图片信息扩充这种操作的时候，pooling并不能很有效地做到这点，因为pooling不是矩阵运算，而是简单的求平均或者是取最大值。

# ### LeakyReLU

# LeakyReLU与ReLU很相似，仅在输入小于0的部分有差别，ReLU输入小于0的部分值都为0，而LeakyReLU输入小于0的部分，值为负，且有微小的梯度。函数图像如下图

# <div align=center><img src="./images/leaky_relu.png" alt="leaky_relu" style="width:500px;"/></div>

# 在反向传播过程中，对于LeakyReLU激活函数输入小于零的部分，也可以计算得到梯度（而不是像ReLU一样值为0）

# GAN的稳定性会因为引入了稀疏梯度受到影响，替换为没有稀疏梯度的LeakyReLU激活能提升GAN的训练稳定性

# ### 数据处理

# 我们将使用来自**CelebA**数据集的人脸图来完成人脸生成的任务
# 
# 数据集下载，若已完成下载，可跳过

# In[5]:


# 生成一些数据、模型、日志存放的目录
Path("celeba_gan").mkdir(parents=True, exist_ok=True)
Path("dcgan_gen_images").mkdir(parents=True, exist_ok=True)
Path("log").mkdir(parents=True, exist_ok=True)


# In[6]:


from zipfile import ZipFile

output = Path("celeba_gan/data.zip")
if not output.exists():
    url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
    output = "celeba_gan/data.zip"
    gdown.download(url, output, quiet=True)

img_foler = Path("celeba_gan/img_align_celeba")
if not output.exists():
    with ZipFile("celeba_gan/data.zip", "r") as zipobj:
        zipobj.extractall("celeba_gan")  
    
print("data is ready!")


# 构建数据集迭代器，并调整图像大小为64x64
# 
# 从我们的文件夹创建一个数据集，并将图像像素值重新缩放到[0-1]范围

# 这里我们使用一个只有1万张照片的数据集进行实验

# In[8]:


dataset = keras.preprocessing.image_dataset_from_directory(
    "celeba_gan", label_mode=None, image_size=(64, 64), batch_size=32
)
dataset = dataset.map(lambda x: x / 255.0)


# 展示人脸图片

# In[9]:


for x in dataset:
    plt.axis("off")
    plt.imshow((x.numpy() * 255).astype("int32")[0])
    break


# ## 代码实战——人脸生成

# ### 生成器

# 生成器使用`Conv2DTranspose`（上采样）层来从种子（随机噪声）中产生图片。以一个使用该种子作为输入的`Dense`层开始，然后多次上采样直到达到所期望的 $64*64*3$ 的图片尺寸
# 
# 除了输出层使用`tanh`之外，其他每层均使用`LeakyReLU`作为激活函数

# In[10]:


latent_dim = 128

generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        keras.layers.Dense(8 * 8 * 128),
        keras.layers.Reshape((8, 8, 128)),
        keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(3, kernel_size=5, padding="same", activation="tanh"),
    ],
    name="generator",
)
generator.summary()


# 使用（尚未训练的）生成器创建一张图片。`

# In[11]:


noise = tf.random.normal([1, latent_dim])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0])


# ### 判别器

# 可以理解为一个全卷积的图片二分类网络，用来判断是否为真实的图像
# 
# 需要注意的是
# 
# 1. 网络中不存在pooling层，直接将特征展开，不做信息的筛减
# 2. 有些论文说`BatchNormalization`可以有效增强生成的效果，有些又说不好，本教程中统一没有加

# In[12]:


discriminator = keras.Sequential(
    [
        keras.Input(shape=(64, 64, 3)),
        keras.layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        keras.layers.LeakyReLU(alpha=0.2),  # 参考论文中的参数设置
        keras.layers.Flatten(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)
discriminator.summary()


# 使用（尚未训练的）判别器来对图片的真伪进行判断。模型将被训练为为真实图片输出正值，为伪造图片输出负值。

# In[13]:


decision = discriminator(generated_image)
print(decision)


# ### 模型定义

# 由于训练方式与GAN如出一辙，直接拷贝前面章节中GAN网络类，并做适当的修改

# 更多GAN的训练技巧可以参考这里的[博文](https://zhuanlan.zhihu.com/p/74663048)

# In[14]:


class DCGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn, metric_fn):
        super(DCGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.d_loss = keras.metrics.Mean(name="d_loss")
        self.g_loss = keras.metrics.Mean(name="g_loss")
        self.d_acc = keras.metrics.Mean(name="d_acc")
        self.g_acc = keras.metrics.Mean(name="g_acc")

    @property
    def metrics(self):
        return [self.d_loss, self.g_loss, self.d_acc, self.g_acc]

    def train_step(self, real_images):
        # 从潜在空间生成一些随机的噪声
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # 编码生成假的图片
        generated_images = self.generator(random_latent_vectors)

        # 与真实图片合并为同一批数据
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # 设置真图片和假图片的标签
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # 重要技巧：给标签增加一些扰动的噪声 => 标签平滑，可以降低训练难度，即判别器不需要绝对的分类准确
        # 如果样本是fake，替换label值为0~0.03
        # 如果样本是real，替换label值为1~1.03
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # 训练判别器
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        # 计算判别器当前batch的准确率
        d_acc = self.metric_fn(labels, predictions)

        # 从潜在空间生成一些随机的噪声
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # 设置这些曲线为“真图片”标签
        misleading_labels = tf.zeros((batch_size, 1))

        # 训练生成器（注意：我们这时候不应该再更新判别器的梯度）
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        # 计算生成器当前batch的准确率
        g_acc = self.metric_fn(misleading_labels, predictions)
        
        # 更新评估指标
        self.d_loss.update_state(d_loss)
        self.d_acc.update_state(d_acc)
        self.g_loss.update_state(g_loss)
        self.g_acc.update_state(g_acc)
        return {
            "d_loss": self.d_loss.result(),
            "d_acc": self.d_acc.result(),
            "g_loss": self.g_loss.result(),
            "g_acc": self.g_acc.result()
        }


# ### 回调函数

# 这里我们来学习如何定义训练中的回调函数（Keras特有的写法）
# 
# 1. 我们希望每个epoch结束模型能够生成一些人脸图像让我们看到
# 2. 我们需要能模型表现最佳的时候的权重
# 3. 记录每个epoch模型的评估直播

# In[15]:


# 图片生成，每个epoch生成一些人脸图像
class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors) # 调用生成器进行图像生成
        generated_images *= 255 # 还原像素值到区间[0,255]
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save("./dcgan_gen_images/generated_img_%03d_%d.png" % (epoch, i))
gan_gen_callback = GANMonitor(10, latent_dim)

# 模型权重保存
# 只保存生成器loss最小的网络权重
Path("models/dcgan").mkdir(parents=True, exist_ok=True)
ckpt_callback = keras.callbacks.ModelCheckpoint(
    filepath=Path("models/dcgan") / "model",
    save_weights_only=True,
    monitor='g_loss',
    mode='min',
    save_best_only=True)

# 日志记录，记录每个epoch的评估指标
timestamp = datetime.datetime.now().strftime("%m%d%H%M")
csv_logger = keras.callbacks.CSVLogger(f"log/dcgan_training_{timestamp}.log")


# ### 模型训练

# In[16]:


# 模型实例化
latent_dim = 128

dcgan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
# 定义优化器，loss函数，评估函数
dcgan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0002),
    loss_fn=keras.losses.BinaryCrossentropy(),
    metric_fn=keras.metrics.BinaryAccuracy()
)


# In[17]:


# 实际训练时，通常需要100个epoch
# 这里为了演示方便，设置为1
epochs = 1

callbacks = [ckpt_callback, gan_gen_callback, csv_logger]

dcgan.fit(dataset, epochs=epochs, callbacks=callbacks)


# GAN不同于其他神经网络，生成器或判别器的损失突然增加或者减少，是正常情况。如果遇到训练不稳定的适合，建议多训练一会，在训练过程中注意生成的图像的质量是否在变好

# ### 结果展示

# In[18]:


def show_images(epoch):
    imagesList = "".join( ["<img style='width: 120px; margin: 5px; float: left; border: 1px solid black;' src='%s' />" % str(s) 
                     for s in sorted(Path("dcgan_gen_images").glob(f"generated_img_{epoch:03d}_*.png")) ])
    display(HTML(imagesList))


# Epoch 0 生成的人脸图像

# In[19]:


show_images(0)


# Epoch 50 生成的人脸图像

# In[40]:


show_images(49)


# Epoch 100 生成的人脸图像

# In[41]:


show_images(99)


# 使用训练过程中生成的图片通过 imageio 生成动态 gif

# In[44]:


anim_file = 'dcgan_gen_images/celeba.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = Path("dcgan_gen_images").glob(f"generated_img_*.png")
    filenames = sorted(filenames)
    last = -1
    for i,filename in enumerate(filenames):
        frame = 2 * (i ** 0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

if IPython.version_info > (6,2,0,''):
    Image(filename=anim_file)


# ![celeba](./dcgan_gen_images/celeba.gif)

# ### 总结

# GAN来生成图片是GAN最基本，也是最主流的使用方法，这个DCGAN是一个开端。
# 
# 但是 DCGAN仍然还有很多问题，比如生成图片质量不算高，图片再大一点就很难train出好效果，图片大的话，训练稳定性也有问题。

# 一些改进方案：

# * [LSGAN](https://arxiv.org/pdf/1611.04076.pdf)：损失替换为Least Squares，让梯度传递更顺畅
# * [WGAN](https://arxiv.org/pdf/1701.07875.pdf)：使用EM推土距离来维持GAN的对抗平衡状态
# * [SAGAN](https://arxiv.org/pdf/1805.08318.pdf): Self-Attention + GAN，围绕重点区域生成图像
