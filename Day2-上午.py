#!/usr/bin/env python
# coding: utf-8

# ## 迁移学习
# - 通过经典网络已经在大数据量下训练好的模型迁移到相似小数据量任务上

# ![caption](img/1.PNG)

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

# 照片数据加载小妙招
from tensorflow.keras.preprocessing import image_dataset_from_directory


# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu


# In[1]:


# 解压数据


# In[ ]:


# ! unzip ./data.zip -d ./
# ! unzip ./dataset.zip -d ./
# ! unzip ./original_dataset.zip -d ./


# ### 加载数据

# In[3]:


# 数据所在文件夹
base_dir = './data/cats_and_dogs'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# 加载训练集
train_dataset = image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)


# In[7]:


# 加载验证集
validation_dataset = image_dataset_from_directory(
    validation_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)


# In[8]:


class_names = train_dataset.class_names

# 准备画图
plt.figure(figsize=(10, 10))

# 查看数据
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


# ### 从验证集移除20%的数据到测试集中

# In[9]:


# 分离出验证集batch的个数
val_batches = tf.data.experimental.cardinality(validation_dataset)

# 取出5份，即验证集的20%作为测试集
test_dataset = validation_dataset.take(val_batches // 5)

# 验证集保留80%
validation_dataset = validation_dataset.skip(val_batches // 5)


# In[10]:


print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))


# ### 高性能读取数据的方法

# In[8]:


# 从数据集中预先取出数据，取出的数据量进行动态调整
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


# ### 简单的数据增强

# In[9]:


data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'), # 翻转
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2), # 旋转
])


# In[10]:


# 展示数据增强的结果
for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')


# ### MobileNet 模型实现
# - 论文地址 https://arxiv.org/abs/1704.04861
# - 轻量级网络，虽然准确率低一些，但是参数真的很少很少

# ![caption](img/2.png)

# ### 获取预训练模型对输入的预处理方法

# In[1]:


preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


# ### 数据标准化

# In[2]:


rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1) # 范围变成[-1, 1]


# ### 创建预训练模型

# In[13]:


# 输入维度为（160，160，3）
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False, # 是否包含顶层的全连接层，不包含，我们想自己编写
    weights='imagenet'
) # 加载权重


# In[14]:


base_model.summary()


# ### 查看数据 特征提取器 160x160x3 image 转化成 5x5x1280 

# In[15]:


image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)


# ### 冻结参数

# In[16]:


base_model.trainable = False


# ### 查看模型

# In[17]:


base_model.summary()


# ### 利用平均池化做特征提取
# - 没有pool_size, strides
# - GlobalAveragePooling2D最后返回的tensor是（batch_size, channels）两个维度的。

# In[18]:


global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)


# ### 输出一个维度 0~1做逻辑判断

# In[19]:


prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)


# ### 迁移学习主流程代码，开始利用预训练的MobileNet创建模型

# In[20]:


# 输入层
inputs = tf.keras.Input(shape=(160, 160, 3))
# 数据增强
x = data_augmentation(inputs)
# 数据预处理
x = preprocess_input(x)
# 模型
x = base_model(x, training=False)
# 全局池化
x = global_average_layer(x)
# Dropout
x = tf.keras.layers.Dropout(0.2)(x)
# 输出层
outputs = prediction_layer(x)
# 整体封装
model = tf.keras.Model(inputs, outputs)


# ### 编译模型，添加学习率，优化器，损失函数，评价方式

# In[21]:


base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # 输出没有经过激活函数，计算logit值
              metrics=['accuracy'])


# In[22]:


model.summary()


# ### 需要训练的参数 w(1) + b(1)

# In[23]:


len(model.trainable_variables)


# ### 配置epochs次数和验证方式

# In[24]:


initial_epochs = 10

loss0, accuracy0 = model.evaluate(validation_dataset)


# In[25]:


print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))


# ### 开始训练

# In[26]:


history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)


# ### 拟合情况展示图形展示

# In[27]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# In[28]:


loss1, accuracy1 = model.evaluate(test_dataset)
print("test loss: {:.2f}".format(loss1))
print("test accuracy: {:.2f}".format(accuracy1))


# ### Fine tuning 微调

# In[29]:


# 先全部置True
base_model.trainable = True


# ### 模型一共154层，对最后54层进行微调

# In[30]:


print("Number of layers in the base model: ", len(base_model.layers))

fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    # 再把前100置False
    layer.trainable =  False


# In[31]:


model.summary()


# In[32]:


model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])


# In[33]:


len(model.trainable_variables)


# ### 迭代式微调训练

# In[34]:


# 为了增加训练速度，刚训练只训练后面部分的网络，一般都是从后向前微调
fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1], # 基于之前的模型，从第10轮开始训练
                         validation_data=validation_dataset)


# In[35]:


acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']


# ### 画图观测两次收敛状态

# In[36]:


plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.8, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# ### 查看测试集结果

# In[37]:


loss, accuracy = model.evaluate(test_dataset)
print("test loss: {:.2f}".format(loss))
print("test accuracy: {:.2f}".format(accuracy))


# ### 最后验证下实际效果

# In[38]:


image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[predictions[i]])
    plt.axis("off")


# ## 反卷积

# - 反向金字塔
# <img src="./img/10.png" alt="FAO" width="600">

# ## 可分离卷积

# - 正常卷积  一个卷积核
# <img src="./img/4.png" alt="FAO" width="600">
# - 正常卷积  256个卷积核
# <img src="./img/5.png" alt="FAO" width="600">

# ##### 三个过程
# - 1、对每个颜色通道进行 分离 然后实现卷积
# <img src="./img/6.png" alt="FAO" width="600">
# - 2、将分离后的卷积后的结果 合并在一起 再进行卷积 
# <img src="./img/7.png" alt="FAO" width="600">
# - 3、有256个这样的卷积核
# <img src="./img/8.png" alt="FAO" width="600">
# 

# # 反卷积和可分离卷积实现

# In[39]:


from tensorflow import keras


# In[40]:


model = keras.models.Sequential()

# 第一层可分离卷积层
model.add(
    keras.layers.SeparableConv2D(
        filter = 32, # 卷积核数量
        kernel_size = 3, # 卷积核尺寸
        padding = 'same', # padding补齐，让卷积之前与之后的大小相同
        activation = 'relu', # 激活函数relu
        input_shape = (28,28,1) # 输入维度是1通道的28*28
    )
)


# 第二层可分离卷积层
model.add(
    keras.layers.SeparableConv2D(
        filters = 32,
        padding = 'same',
        activation = 'relu'
    )
)
                              
# 最大池化层
model.add(
    keras.layers.MaxPool2D(
        pool_size = 2
    )
)


# 第三层卷积层
model.add(keras.layers.SeparableConv2D(filters = 64,           
                              kernel_size = 3,          
                              padding = 'same',         
                              activation = 'relu'))      
          
# 第四层卷积层
model.add(keras.layers.SeparableConv2D(filters=128, strides=(2, 2),             
                              kernel_size = 3,          
                              padding = 'same',        
                              activation = 'relu'))     
 
# 最大池化层
model.add(keras.layers.MaxPool2D(pool_size = 2))


# 反卷积层
model.add(keras.layers.Conv2DTranspose(filters=128, strides=(2, 2),       
                              kernel_size = 3,          
                              padding = 'same',       
                              activation = 'relu')) 

# 反卷积层
model.add(keras.layers.Conv2DTranspose(filters=128, strides=(2, 2),                 
                              kernel_size = 3,          
                              padding = 'same',         
                              activation = 'relu')) 

# 全连接层
model.add(keras.layers.Flatten())  # 展平输出
model.add(keras.layers.Dense(128, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = "softmax")) # 输出为 10的全连接层


# In[41]:


model.summary()


# In[ ]:




