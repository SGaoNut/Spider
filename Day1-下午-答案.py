#!/usr/bin/env python
# coding: utf-8

# # CNN 项目实战

# ## 猫狗识别
# 
# - 数据预处理：图像数据处理，准备训练和验证数据集
# - 卷积网络模型：构建网络架构
# - 过拟合问题：观察训练和验证效果，针对过拟合问题提出解决方法
# - 数据增强：图像数据增强方法与效果
# 
# <img src="./img/11.png" alt="FAO" width="990">
# 

# ### 导入工具包

# In[1]:


import os
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf

# 导入adam优化函数
from tensorflow.keras.optimizers import Adam

# 导入图像生成器
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[2]:


import os

# 将gpu禁用
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu


# ### 使用GPU

# In[3]:


# gpus = tf.config.list_physical_devices("GPU")
# print(gpus)
# if gpus:
#     gpu0 = gpus[0] #如果有多个GPU，仅使用第0个GPU
#     tf.config.experimental.set_memory_growth(gpu0, True) #设置GPU显存用量按需使用
#
#     tf.config.set_visible_devices([gpu0],"GPU")


# ### 执行本地命令解压

# In[4]:


# ! unzip ./data.zip -d ./


# ### 指定好数据路径（训练和验证）

# In[5]:


# 数据所在文件夹
base_dir = './data/cats_and_dogs'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# 训练集
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

# 验证集
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')


# ### 构建卷积神经网络模型
# - 几层都可以，大家可以随意玩
# - 如果用CPU训练，可以把输入设置的更小一些，一般输入大小更主要的决定了训练速度

# - https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Conv2D

# ### MaxPooling2D 单独测试
# - tf 接口独立测试方法
# ```
# import numpy as np
# x = np.random.randn(2,2,2,2)
# max_pool = tf.keras.layers.MaxPooling2D(2, 2)
# max_pool = tf.keras.layers.MaxPooling2D(2, 2)
# print('原始数据:', x)
# print()
# print('Maxpooling2D后数据:', max_pool(x))
# ```

# In[6]:


model = tf.keras.models.Sequential([
    #如果训练慢，可以把数据设置的更小一些
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    
    # MaxPooling2D 第一个参数pool_size，第二个参数是strides， [4,2,3,1] --> [[4, 2], [3, 1]] --> [4, 3]
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    #为全连接层准备 将所有数据展平
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(512, activation='relu'),
    
    # 二分类sigmoid就够了
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()


# In[7]:


# 参数计算
3 * 3 * 3 * 32 + 32


# ### 配置训练器
# - binary_crossentropy 二分类损失函数
# 
# #### acc 准确率
# - TP：正例预测正确的个数
# - FP：负例预测错误的个数
# - TN：负例预测正确的个数
# - FN：正例预测错误的个数
# - TP + TN / TP + FP + TN + FN

# In[8]:


model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=1e-4),
    metrics=['acc']
)


# ### 数据预处理
# 
# - 读进来的数据会被自动转换成tensor(float32)格式，分别准备训练和验证
# - 图像数据归一化（0-1）区间

# In[9]:


# ImageDataGenerator 数据生成器
# 将0~255的数据生成到0~1之间
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,  # 文件夹路径
        target_size=(64, 64),  # 指定resize成的大小
        batch_size=20, # 训练一批需要多少数据
        # 如果one-hot就是categorical，二分类用binary就可以
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(64, 64),
        batch_size=20,
        class_mode='binary')


# ### 训练网络模型
# - 直接fit也可以，但是通常咱们不能把所有数据全部放入内存，fit_generator相当于一个生成器，动态产生所需的batch数据
# - steps_per_epoch相当给定一个停止条件，因为生成器会不断产生batch数据，说白了就是它不知道一个epoch里需要执行多少个step
# 

# In[10]:


# history 获取所有历史训练的结果
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,  # 2000 images = batch_size * steps
      epochs=20,
      validation_data=validation_generator,
      validation_steps=50,  # 1000 images = batch_size * steps
      verbose=2)


# ### 效果展示

# In[11]:


import matplotlib.pyplot as plt

# 获取所有历史训练结果
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# 为了将横坐标变成训练迭代次数
epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# - 看起来有点过拟合了，如何解决呢？

# - 模型能力不足，欠拟合
# <img src="./img/12.PNG" alt="FAO" width="490">
# 

# - 模型尚未收敛，欠拟合
# <img src="./img/13.PNG" alt="FAO" width="490">
# 

# - 训练集逐步下降，验证集已经长时间收敛， 过拟合
# <img src="./img/14.PNG" alt="FAO" width="490">
# 

# ## 数据增强

# ![caption](img/15.png)

# #### 深度学习中最依赖的就是数据量了，同样一只猫轻松一变数据量就翻倍了

# In[12]:


import matplotlib.pyplot as plt
from PIL import Image
get_ipython().run_line_magic('matplotlib', 'inline')
from tensorflow.keras.preprocessing import image
# import tf.keras.backend as K
import os
import glob
import numpy as np


# In[13]:


# 展示输入数据
def print_result(path):
    name_list = glob.glob(path)
    fig = plt.figure(figsize=(12,16))
    for i in range(3):
        img = Image.open(name_list[i])
        sub_img = fig.add_subplot(131+i)
        sub_img.imshow(img)


# In[14]:


img_path = './img/cats/*'
in_path = './img/'
out_path = './output/'
name_list = glob.glob(img_path)
name_list


# - 获取指定目录下的所有图片
# - 加上r让字符串不转义

# In[15]:


print(glob.glob(img_path),"\n")


# In[17]:


print_result(img_path)


# ### 指定转换后所有图像都变为相同大小
# - in_path 源路径
# - batch_size 批数量
# - shuffle 是否随机化 
# - save_to_dir 存储路径位置
# - save_prefix 存储文件增加前缀
# - target_size 转换后的统一大小

# In[25]:


datagen = image.ImageDataGenerator()

# 变成一个可迭代数据
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False,
                                       save_to_dir = out_path+'resize',
                                  save_prefix='gen', target_size=(224, 224))


# ### 输出到指定ouput目录

# In[26]:


for i in range(3):
    gen_data.next()


# In[27]:


print_result(out_path+'resize/*')


# ### 角度旋转

# In[31]:


# 删除之前输出文件剩余的文件
del_list = os.listdir(out_path+'rotation_range')
for f in del_list:
    file_path = os.path.join(out_path+'rotation_range', f)
    if os.path.isfile(file_path):
        os.remove(file_path)

# 图像旋转45度
datagen = image.ImageDataGenerator(rotation_range=45)

# 获取可迭代数据
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
np_data = np.concatenate([data.next() for i in range(data.n)])

# 数据转换
datagen.fit(np_data)
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, 
                                       save_to_dir=out_path+'rotation_range',save_prefix='gen', target_size=(224, 224))

# 生成角度旋转后的数据
for i in range(3):
    gen_data.next()
print_result(out_path+'rotation_range/*')


# ### 平移变换

# In[56]:


# 数据高度平移，数据宽度平移
datagen = image.ImageDataGenerator(width_shift_range=0.3,height_shift_range=0.3)

# 预处理数据
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
np_data = np.concatenate([data.next() for i in range(data.n)])
datagen.fit(np_data)

# 生成数据
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path+'shift',save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
print_result(out_path+'shift/*')


# ### 平移变换2

# In[57]:


datagen = image.ImageDataGenerator(width_shift_range=-0.3, height_shift_range=0.3)
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
np_data = np.concatenate([data.next() for i in range(data.n)])
datagen.fit(np_data)
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path+'shift2',save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
print_result(out_path+'shift2/*')


# ### 缩放

# In[58]:


# 随机缩放幅度 （图像的部分区域）
datagen = image.ImageDataGenerator(zoom_range=0.5)

# 预处理数据
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
np_data = np.concatenate([data.next() for i in range(data.n)])

datagen.fit(np_data)
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path+'zoom',save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
print_result(out_path+'zoom/*')


# ### 像素通道平移

# In[59]:


# 随机通道偏移的幅度
datagen = image.ImageDataGenerator(channel_shift_range=15)

# 预处理数据
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
np_data = np.concatenate([data.next() for i in range(data.n)])

# 生成数据
datagen.fit(np_data)
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path+'channel',save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
print_result(out_path+'channel/*')


# ### 翻转

# In[60]:


# 进行水平翻转
datagen = image.ImageDataGenerator(horizontal_flip=True)

# 预处理数据
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(
    in_path,
    batch_size=1,
    class_mode=None,
    shuffle=True,
    target_size=(224, 224)
)
np_data = np.concatenate([data.next() for i in range(data.n)])

# 生成数据
datagen.fit(np_data)
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path+'horizontal',save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
print_result(out_path+'horizontal/*')


# ### 查看数据

# In[61]:


gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path+'horizontal',save_prefix='gen', target_size=(224, 224))
gen_data.next()


# ### rescale

# In[62]:


# 归一化到 0~1的数据返回
datagen = image.ImageDataGenerator(rescale= 1/255)

# 预处理数据
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
np_data = np.concatenate([data.next() for i in range(data.n)])
datagen.fit(np_data)

# 生成数据
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path+'rescale',save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
print_result(out_path+'rescale/*')


# In[63]:


gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path+'rescale',save_prefix='gen', target_size=(224, 224))
gen_data.next()


# ### 填充方法
# - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
# - 'nearest': aaaaaaaa|abcd|dddddddd
# - 'reflect': abcddcba|abcd|dcbaabcd
# - 'wrap': abcdabcd|abcd|abcdabcd
# 
# 

# In[64]:


datagen = image.ImageDataGenerator(fill_mode='wrap', zoom_range=[4, 4])

gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
np_data = np.concatenate([data.next() for i in range(data.n)])

datagen.fit(np_data)
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path+'fill_mode',save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
print_result(out_path+'fill_mode/*')


# In[65]:


datagen = image.ImageDataGenerator(fill_mode='nearest', zoom_range=[4, 4])
gen = image.ImageDataGenerator()
data = gen.flow_from_directory(in_path, batch_size=1, class_mode=None, shuffle=True, target_size=(224, 224))
np_data = np.concatenate([data.next() for i in range(data.n)])
datagen.fit(np_data)
gen_data = datagen.flow_from_directory(in_path, batch_size=1, shuffle=False, save_to_dir=out_path+'nearest',save_prefix='gen', target_size=(224, 224))
for i in range(3):
    gen_data.next()
print_result(out_path+'nearest/*')


# ### 增强后重新查看结果

# In[66]:


import os
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[67]:


base_dir = './data/cats_and_dogs'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')


# In[68]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

#     tf.keras.layers.Dropout(0.5), 防止过拟合 随机部分神经元权重为0
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')
])


# In[69]:


model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=1e-4),
              metrics=['acc'])


# In[70]:


train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,  
        target_size=(64, 64),  
        batch_size=20,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(64, 64),
        batch_size=20,
        class_mode='binary')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,  # 2000 images = batch_size * steps
      epochs=20,
      validation_data=validation_generator,
      validation_steps=50,  # 1000 images = batch_size * steps
      verbose=2)


# In[71]:


import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# ### 安装 tensorboardX

# - https://pypi.org/project/tensorboardX/#files
# - https://github.com/lanpa/tensorboardX
# 

# In[72]:


from tensorboardX import SummaryWriter


# In[73]:


# 结果存放的位置
writer = SummaryWriter(log_dir='./board')


# In[74]:


# 以key value n 模式存放结果
for n_iter, _ in enumerate(zip(acc, val_acc, loss, val_loss)):
    writer.add_scalar('train/acc', _[0], n_iter)
    writer.add_scalar('val/acc', _[1], n_iter)
    writer.add_scalar('train/loss', _[2], n_iter)
    writer.add_scalar('val/loss', _[3], n_iter)


# In[40]:


writer.close()


# ### 查看tensorboard

# - tensorboard --logdir=board --port 6006
# - ssh -p 22 -L 16006:127.0.0.1:6006 root@47.94.92.106
# - http://127.0.0.1:16006/#scalars

# <img src="./img/16.PNG" alt="FAO" width="990">

# ### CNN文本分类实战

# <img src="./img/10.PNG" alt="FAO" width="500">

# - 1：文本数据预处理，必须都是相同长度，相同向量维度
# - 2：构建卷积模型，注意卷积核大小的设计
# - 3：将卷积后的特征图池化成一个特征
# - 4：将多种特征拼接在一起，准备完成分类任务

# In[41]:


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# 将句子自动对齐
from tensorflow.keras.preprocessing.sequence import pad_sequences

num_features = 3000
sequence_length = 300
embedding_dimension = 100

# 获取电影评论数据
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_features)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### 将句子对齐

# In[42]:


x_train = pad_sequences(x_train, maxlen=sequence_length)
x_test = pad_sequences(x_test, maxlen=sequence_length)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[43]:


# 多种卷积核，相当于单词数
filter_sizes=[3,4,5]
def convolution():
    inn = layers.Input(shape=(sequence_length, embedding_dimension, 1))#3维的
    cnns = []
    for size in filter_sizes:
#         conv = layers.Conv1D(filters=64, kernel_size=size,padding='valid', activation='relu')(inn)

        conv = layers.Conv2D(filters=64, kernel_size=(size, embedding_dimension),
                            strides=1, padding='valid', activation='relu')(inn)

        #需要将多种卷积后的特征图池化成一个特征
        pool = layers.MaxPool2D(pool_size=(sequence_length-size+1, 1), padding='valid')(conv)
        cnns.append(pool)
    # 将得到的特征拼接在一起
    outt = layers.concatenate(cnns)

    model = keras.Model(inputs=inn, outputs=outt)
    return model

def cnn_mulfilter():
    model = keras.Sequential([
        layers.Embedding(input_dim=num_features, output_dim=embedding_dimension,
                        input_length=sequence_length),
        layers.Reshape((sequence_length, embedding_dimension, 1)),
        
        convolution(),
        
        # 将数据展平
        layers.Flatten(),
        layers.Dense(10, activation='relu'),
#         layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')

    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])
    return model

model = cnn_mulfilter()
model.summary()


# In[44]:


history = model.fit(
    x = x_train,
    y = y_train,
    batch_size=64,
    epochs=10,
    validation_split=0.1
)


# In[45]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valiation'], loc='upper left')
plt.show()


# In[ ]:




