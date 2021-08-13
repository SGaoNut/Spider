#!/usr/bin/env python
# coding: utf-8

# ## Resnet 详解

# ### 论文理解
# - 论文地址 https://arxiv.org/pdf/1512.03385.pdf

# - 实验中发现的不足
# ![title](./img/11.PNG)

# - 思想的启发
# ![title](./img/12.PNG)

# - 网络对比
# ![title](./img/13.PNG)

# - 结果
# ![title](./img/14.PNG)
# - 遗留问题 （虚线是训练集， 实线是预测集）
# ![title](./img/15.PNG)

# ### 模型结构梳理

# ![title](./img/17.PNG)
# ![title](./img/18.PNG)

# ### 基础参数配置

# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu


# In[1]:


# 最大迭代次数
EPOCHS = 10

# 每次批处理数量
BATCH_SIZE = 32

# 多少分类
NUM_CLASSES = 3

# 照片高度
image_height = 32

# 照片宽度
image_width = 32

# 颜色通道数量
channels = 3

# 保存网络模型地址
save_model_dir = "saved_model/model"

# 数据地址
dataset_dir = "dataset/"
train_dir = dataset_dir + "train"
valid_dir = dataset_dir + "valid"
test_dir = dataset_dir + "test"


# ### 加载一些模块

# In[1]:


import tensorflow as tf
import pathlib
import math
import warnings
warnings.filterwarnings("ignore")


# ### GPU设置

# In[5]:


# gpus = tf.config.list_physical_devices("GPU")
# print(gpus)
# if gpus:
#     gpu0 = gpus[0] #如果有多个GPU，仅使用第0个GPU
#     tf.config.experimental.set_memory_growth(gpu0, True) #设置GPU显存用量按需使用
#
#     tf.config.set_visible_devices([gpu0],"GPU")


# ### 数据切分

# In[6]:


import os
import random

# 数据随机化
import shutil


class SplitDataset():
    def __init__(self, dataset_dir, saved_dataset_dir, train_ratio=0.6, test_ratio=0.2, show_progress=False):
        self.dataset_dir = dataset_dir
        self.saved_dataset_dir = saved_dataset_dir
        self.saved_train_dir = saved_dataset_dir + "/train/"
        self.saved_valid_dir = saved_dataset_dir + "/valid/"
        self.saved_test_dir = saved_dataset_dir + "/test/"


        self.train_ratio = train_ratio
        self.test_radio = test_ratio
        self.valid_ratio = 1 - train_ratio - test_ratio

        self.train_file_path = []
        self.valid_file_path = []
        self.test_file_path = []

        # 索引和标签的映射表
        self.index_label_dict = {}

        # 是否展现过程
        self.show_progress = show_progress

        # 定义数据存放位置
        if not os.path.exists(self.saved_train_dir):
            os.mkdir(self.saved_train_dir)
        if not os.path.exists(self.saved_test_dir):
            os.mkdir(self.saved_test_dir)
        if not os.path.exists(self.saved_valid_dir):
            os.mkdir(self.saved_valid_dir)


    def _get_label_names(self):
        # 把所有目录作为标签名称
        label_names = []
        for item in os.listdir(self.dataset_dir):
            item_path = os.path.join(self.dataset_dir, item)
            if os.path.isdir(item_path):
                label_names.append(item)
        return label_names

    def _get_all_file_path(self):
        
        all_file_path = []
        
        #得到各个类别
        for index, file_type in enumerate(self._get_label_names()):
            
            #给每个类别一个索引
            self.index_label_dict[index] = file_type
            
            #当前类别路径
            type_file_path = os.path.join(self.dataset_dir, file_type)
            
            file_path = []
            
            #遍历当前类别里的所有样本
            for file in os.listdir(type_file_path):
                single_file_path = os.path.join(type_file_path, file)
                file_path.append(single_file_path)
            
            # 获取一个二维 [[]] 列表，第一维列表的索引代表 第二维度列表的文件路径的类别
            all_file_path.append(file_path)
            
        return all_file_path


    def _split_dataset(self):
        all_file_paths = self._get_all_file_path()
        
        #每个类别分别做划分
        for index in range(len(all_file_paths)):
            
            #当前类别所有样本路径
            file_path_list = all_file_paths[index]
            file_path_list_length = len(file_path_list)
            random.shuffle(file_path_list)
            
            #随机选择训练和测试样本
            train_num = int(file_path_list_length * self.train_ratio)
            test_num = int(file_path_list_length * self.test_radio)

            self.train_file_path.append([self.index_label_dict[index], file_path_list[: train_num]])
            
            self.test_file_path.append([self.index_label_dict[index], file_path_list[train_num:train_num + test_num]])
            
            self.valid_file_path.append([self.index_label_dict[index], file_path_list[train_num + test_num:]])

    def _copy_files(self, type_path, type_saved_dir):
        for item in type_path:
            src_path_list = item[1]
            dst_path = type_saved_dir + "%s/" % (item[0])
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            for src_path in src_path_list:
                shutil.copy(src_path, dst_path)
                if self.show_progress:
                    print("Copying file "+src_path+" to "+dst_path)
                    
    def start_splitting(self):
        # 主函数入口
        self._split_dataset()
        self._copy_files(type_path=self.train_file_path, type_saved_dir=self.saved_train_dir)
        self._copy_files(type_path=self.valid_file_path, type_saved_dir=self.saved_valid_dir)
        self._copy_files(type_path=self.test_file_path, type_saved_dir=self.saved_test_dir)


# In[7]:


split_dataset = SplitDataset(
    dataset_dir="original_dataset",
    saved_dataset_dir="dataset"
    # show_progress=True
                             )
split_dataset.start_splitting()


# ### 生成训练数据

# In[5]:


def load_and_preprocess_image(img_path):
    # 读取照片
    img_raw = tf.io.read_file(img_path)
    # 解码照片
    img_tensor = tf.image.decode_jpeg(
        img_raw,
        channels=channels
    )
    # 照片resize
    img_tensor = tf.image.resize(
        img_tensor,
        [
            image_height,
            image_width
        ]
    )
    img_tensor = tf.cast(
        img_tensor,
        tf.float32
    )
    # 归一化
    img = img_tensor / 255.0
    return img

def get_images_and_labels(data_root_dir):
    # 得到所有图像路径
    data_root = pathlib.Path(data_root_dir)

    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    # 得到标签名字
    label_names = sorted(item.name for item in data_root.glob('*/'))
    # 例如：{'cats': 0, 'dogs': 1, 'panda': 2}
    label_to_index = dict((index, label) for label, index in enumerate(label_names))
    # 每一个图像对应的标签
    all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in all_image_path]

    return all_image_path, all_image_label


def get_dataset(dataset_root_dir):
    
    # 获取每个照片对应的标签
    all_image_path, all_image_label = get_images_and_labels(
        data_root_dir=dataset_root_dir
    )
    
    # 通过自定义的load_and_preprocess_image函数，从照片路径中读取照片做为数据集
    image_dataset = tf.data.Dataset.from_tensor_slices(all_image_path).map(load_and_preprocess_image)
    
    # 获取标签数据集
    label_dataset = tf.data.Dataset.from_tensor_slices(all_image_label)
    
    # 让照片和标签一一对应
    dataset = tf.data.Dataset.zip(
        (
            image_dataset,
            label_dataset
        )
    )
    image_count = len(all_image_path)

    return dataset, image_count


def generate_datasets():
    
    # 生成数据
    train_dataset, train_count = get_dataset(dataset_root_dir=train_dir)
    valid_dataset, valid_count = get_dataset(dataset_root_dir=valid_dir)
    test_dataset, test_count = get_dataset(dataset_root_dir=test_dir)


    # 以BATCH_SIZE大小随机读取数据
    train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size=BATCH_SIZE)
    valid_dataset = valid_dataset.batch(batch_size=BATCH_SIZE)
    test_dataset = test_dataset.batch(batch_size=BATCH_SIZE)

    return train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count


# In[9]:


train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()


# ### 模型编写

# <img src="./img/17.PNG" alt="FAO" width="600">
# <img src="./img/18.PNG" alt="FAO" width="600">

# ### Block 核心块代码编写

# In[10]:


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1, with_other=True):
        super(BottleNeck, self).__init__()
        self.with_other = with_other

        # 2D卷积神经网络
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filter_num,
            kernel_size=(1, 1),
            strides=1,
            padding='same'
        )

        # 列归一化
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filter_num,
            kernel_size=(3, 3),
            strides=stride,
            padding='same'
        )

        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(
            # 考虑这里为什么需要乘以4
            filters=filter_num * 4,
            kernel_size=(1, 1),
            strides=1,
            padding='same'
        )

        self.bn3 = tf.keras.layers.BatchNormalization()
        # other就是一个完整的模型，
        self.other = tf.keras.Sequential()
        self.other.add(
            tf.keras.layers.Conv2D(
                filters=filter_num * 4,
                kernel_size=(1, 1),
                strides=stride
            )
        )
        self.other.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None):

        covn_ = self.other(inputs)

        conv1 = self.conv1(inputs)

        # 训练时 需要batch，预测试时关闭，预测时候是全局的
        bn1 = self.bn1(conv1, training=training)
        relu1 = tf.nn.relu(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2, training=training)
        relu2 = tf.nn.relu(bn2)
        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3, training=training)

        if self.with_other == True:
            # 构建identity网络，加入了1*1 卷积合并特征
            output = tf.nn.relu(tf.keras.layers.add([covn_, bn3]))
        else:
            output = tf.nn.relu(tf.keras.layers.add([inputs, bn3]))

        return output


def build_res_block(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    # 需要增加虚线（with_other = True）
    res_block.add(BottleNeck(filter_num, stride=stride, with_other=True))

    # 通过other 变量控制增加额外卷积，特征数量一致时无需增加
    for _ in range(1, blocks):
        res_block.add(BottleNeck(filter_num, stride=1, with_other=False))

    return res_block


# ### ResNet50主模型编写 （混合模型编写）

# In[11]:


class ResNet50(tf.keras.Model):
    def __init__(self, num_classes=NUM_CLASSES):
        super(ResNet50, self).__init__()

        # stage1 对应7*7的卷积核
        self.pre1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=2,
            padding='same'
        )
        self.pre2 = tf.keras.layers.BatchNormalization()
        self.pre3 = tf.keras.layers.Activation(tf.keras.activations.relu)
        self.pre4 = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3),
            strides=2
        )

        # stage2~stage5 特征数量和block数据发生了变化，都是以实验结果为导向
        # stage2 构建一个block
        self.layer1 = build_res_block(
            filter_num=64,
            blocks=3
        )
        
        # stage3
        self.layer2 = build_res_block(
            filter_num=128,
            blocks=4,
            stride=2
        )

        # stage4
        self.layer3 = build_res_block(
            filter_num=256,
            blocks=6,
            stride=2
        )
        
        # stage5
        self.layer4 = build_res_block(
            filter_num=512,
            blocks=3,
            stride=2
        )

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        
        # 激活函数relu
        self.fc1 = tf.keras.layers.Dense(
            units=1000,
            activation=tf.keras.activations.relu
        )
        self.drop_out = tf.keras.layers.Dropout(rate=0.5)
        
        # 激活函数 softmax 主要用于多分类，进行1000分类
        self.fc2 = tf.keras.layers.Dense(
            units=num_classes,
            activation=tf.keras.activations.softmax
        )

    def call(self, inputs, training=None, mask=None):
        # 主模型流程
        
        # stage1
        pre1 = self.pre1(inputs)
        # 只是针对training是游泳的
        pre2 = self.pre2(pre1, training=training)
        pre3 = self.pre3(pre2)
        pre4 = self.pre4(pre3)
        
        # stage2
        l1 = self.layer1(pre4, training=training)
        
        # stage3
        l2 = self.layer2(l1, training=training)
        
        # stage4
        l3 = self.layer3(l2, training=training)
        
        # stage5
        l4 = self.layer4(l3, training=training)
        
        # 最后输出
        avgpool = self.avgpool(l4)
        fc1 = self.fc1(avgpool)
        drop = self.drop_out(fc1)
        out = self.fc2(drop)

        return out


# In[12]:


model = ResNet50()
model.build(
    input_shape=(None, image_height, image_width, channels)
)
model.summary()


# ### 定义loss 优化器 评价函数
# - SparseCategoricalCrossentropy 对稀疏的多分类损失函数（类型特别多）
# - SparseCategoricalAccuracy 对系数的多分类准确率
# - Sparse 输出是整数，对应label的index 
# - CategoricalCrossentropy 输出的one hot 对应的是[1,0,0]

# In[13]:


# 设置loss 目标
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

# 设置优化器
optimizer = tf.keras.optimizers.Adam(lr=0.001)

# 设置train loss
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy'
)

# 设置valid loss
valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='valid_accuracy'
)


# ### 分离式自定义求梯度， loss， 准确率，等

# In[14]:


@tf.function
def train_step(images, labels):
    # 自定义梯度
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        
        # 使用自定义loss
        loss = loss_object(y_true=labels, y_pred=predictions)
        
    # 求出梯度结果
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # 将梯度传到训练的所有参数变量中
    optimizer.apply_gradients(
        grads_and_vars=zip(gradients, model.trainable_variables)
    )

    # 获取自定义的评价结果
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def valid_step(images, labels):
    # 预测时候改变参数
    predictions = model(images, training=False)
    v_loss = loss_object(labels, predictions)

    # 获取自定义的评价结果
    valid_loss(v_loss)
    valid_accuracy(labels, predictions)


# ### 开始训练

# In[15]:


for epoch in range(EPOCHS):
    
    # 重置所有梯度值
    train_loss.reset_states()
    train_accuracy.reset_states()
    valid_loss.reset_states()
    valid_accuracy.reset_states()
    
    step = 0
    
    # 开始训练
    for images, labels in train_dataset:
        step += 1
        train_step(images, labels)
        print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(
            epoch + 1, EPOCHS,
            step,
            math.ceil(train_count / BATCH_SIZE),
            train_loss.result(),
            train_accuracy.result()
        )
        )


# ### 储存模型

# In[16]:


model.save_weights(filepath=save_model_dir, save_format='tf')


# ### 重新加载模型

# In[17]:


model = ResNet50()
model.build(input_shape=(None, image_height, image_width, channels))
model.load_weights(filepath=save_model_dir)


# ### 在测试数据集测试结果

# In[18]:


# 在测试数据重复之前的操作
loss_object = tf.keras.metrics.SparseCategoricalCrossentropy()
test_loss = tf.keras.metrics.Mean()
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

for test_images, test_labels in test_dataset:
    test_step(test_images, test_labels)
#     print("loss: {:.5f}, test accuracy: {:.5f}".format(test_loss.result(),
#                                                        test_accuracy.result()))

print("The accuracy on test set is: {:.3f}%".format(test_accuracy.result()*100))


# In[ ]:




