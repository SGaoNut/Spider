#!/usr/bin/env python
# coding: utf-8

# ## RNN 文本分类任务实战
# 
# - 数据集构建：影评数据集进行情感分析（分类任务）
# - 词向量模型：加载训练好的词向量或者自己训练都可以
# - 序列网络模型：训练RNN模型进行识别

# ![title](./img/24.png)

# ### RNN模型所需数据解读：
# 
# ![title](./img/31.png)

# ### 隐藏层计算方式
# - g 代表激活函数
# - h (t-1) 代表上一个时刻的隐藏层神经元计算结果
# - x (t) 代表当前时刻的输入参数
# - w 均表示一个需要被学习的参数
# ![title](./img/34.PNG)

# ### 压缩文件

# In[1]:


get_ipython().system(' unzip ./data.zip -d ./')


# In[2]:


import os
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np

import pprint
import logging
import time

# python 内置用作统计的模块
from collections import Counter

# python 内置构建路径的方法
from pathlib import Path

# 获取进度的方法
from tqdm import tqdm


# In[3]:


import tensorflow


# ### 加载影评数据集

# In[4]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()


# In[5]:


x_train.shape


# ### 查看数据

# In[6]:


x_train[0]


# ### 建立词和ID的映射表，增加3个特殊字符
# - word2idx['pad'] = 0 表示填充字符
# - word2idx['start'] = 1 表示开始字符
# - word2idx['unk'] = 2 表示不在词表范围内的数据

# In[7]:


# 获取电影评论 word 到 index的关系对应表
_word2idx = tf.keras.datasets.imdb.get_word_index()

# 将所有索引往后加三
word2idx = {w: i+3 for w, i in _word2idx.items()}

# 填充三个特定的字符
word2idx['<pad>'] = 0
word2idx['<start>'] = 1
word2idx['<unk>'] = 2

# 获取index 对word是词
idx2word = {i: w for w, i in word2idx.items()}


# ### 按文本长度大小进行排序

# In[8]:


def sort_by_len(x, y):
    x, y = np.asarray(x), np.asarray(y)
    idx = sorted(range(len(x)), key=lambda i: len(x[i]))
    return x[idx], y[idx]


# ### 将文本数据结果保存到本地

# In[9]:


x_train, y_train = sort_by_len(x_train, y_train)
x_test, y_test = sort_by_len(x_test, y_test)

def write_file(f_path, xs, ys):
    with open(f_path, 'w',encoding='utf-8') as f:
        for x, y in zip(xs, ys):
            f.write(str(y)+'\t'+' '.join([idx2word[i] for i in x][1:])+'\n')

# 写入本地文件
write_file('./data/train.txt', x_train, y_train)
write_file('./data/test.txt', x_test, y_test)


# ### 构建语料表，基于词频来进行统计

# In[10]:


counter = Counter()

with open('./data/train.txt',encoding='utf-8') as f:
    for line in f:
        line = line.rstrip()
        label, words = line.split('\t')
        words = words.split(' ')
        
        # 自动对word 进行计数
        counter.update(words)

# 构建二级词表，这个词表是用来获取 出现次数大于等于10的所有单词 加上一个额外用来填充的字符
words = ['<pad>'] + [w for w, freq in counter.most_common() if freq >= 10]

print('Vocab Size:', len(words))

# 制作词表目录
Path('./vocab').mkdir(exist_ok=True)

with open('./vocab/word.txt', 'w',encoding='utf-8') as f:
    for w in words:
        f.write(w+'\n')


# ### 得到新的词表

# In[11]:


word2idx = {}

# 保存到词表的新词库
with open('./vocab/word.txt',encoding='utf-8') as f:
    for i, line in enumerate(f):
        line = line.rstrip()
        word2idx[line] = i


# ### embedding层
# - 可以基于网络来训练，也可以直接加载别人训练好的，一般都是加载预训练模型
# - 这里有一些常用的：https://nlp.stanford.edu/projects/glove/

# ![title](./img/25.png)

# ### 构建词嵌入矩阵（迁移学习）【20599*50】

# In[12]:


# 构建初始矩阵
embedding = np.zeros((len(word2idx)+1, 50)) # + 1 表示如果不在语料表中，就都是unknow

with open('./rnn_data/glove.6B.50d.txt',encoding='utf-8') as f: #下载好的词向量矩阵数据
    count = 0
    for i, line in enumerate(f):
        if i % 100000 == 0:
            print('- line {}'.format(i)) #打印处理了多少数据
        
        # 删除前后的空格
        line = line.rstrip()
        
        sp = line.split(' ')
        
        #默认下载的向量矩阵是第一个是单词， 其余的是向量
        word, vec = sp[0], sp[1:]
        if word in word2idx:
            count += 1
            
            #将词转换成对应的向量
            embedding[word2idx[word]] = np.asarray(vec, dtype='float32') 


# ### 存储词向量文件

# In[13]:


print("[%d / %d] words have found pre-trained values"%(count, len(word2idx)))
np.save('./vocab/word.npy', embedding)
print('Saved ./vocab/word.npy')


# ### 构建训练数据
# 
# - 注意所有的输入样本必须都是相同shape（文本长度，词向量维度等）
# 
# - tf.data.Dataset.from_tensor_slices(tensor)：将tensor沿其第一个维度切片，返回一个含有N个样本的数据集，这样做的问题就是需要将整个数据集整体传入，然后切片建立数据集类对象，比较占内存。
# 
# - tf.data.Dataset.from_generator(data_generator,output_data_type,output_data_shape)：从一个生成器中不断读取样本

# In[14]:


def data_generator(f_path, params):
    with open(f_path,encoding='utf-8') as f:
        print('Reading', f_path)
        for line in f:
            line = line.rstrip()
            label, text = line.split('\t')
            text = text.split(' ')
            
            # 得到当前词所对应的ID
            x = [params['word2idx'].get(w, len(word2idx)) for w in text] 
            
            # 截断操作
            if len(x) >= params['max_len']: 
                x = x[:params['max_len']]
            else:
                # 补齐操作
                x += [0] * (params['max_len'] - len(x)) 
            y = int(label)
            yield x, y


# ### 自定义构建dataset

# In[15]:


def dataset(is_training, params):
    _shapes = ([params['max_len']], ())
    _types = (tf.int32, tf.int32)
  
    if is_training:
        
        # 非常实用
        ds = tf.data.Dataset.from_generator(
            lambda: data_generator(params['train_path'], params),
            output_shapes = _shapes,
            output_types = _types,)
        
        
        # 对数据进行打乱
        ds = ds.shuffle(params['num_samples'])
        
        # 对数据批处理化
        ds = ds.batch(params['batch_size'])
        
        #设置缓存序列，根据可用的CPU动态设置并行调用的数量，为了怎加训练速度
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_generator(
            lambda: data_generator(params['test_path'], params),
            output_shapes = _shapes,
            output_types = _types,)
        ds = ds.batch(params['batch_size'])
        
        # 高性能预获取数据的方法
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  
    return ds


# ### 自定义网络模型
# - 定义好都有哪些层
# - 前向传播走一遍就行了

# embedding_lookup的作用：

# ![title](./img/26.png)

# ![title](./img/32.png)

# ### LSTM 模型概要
# <img src="./img/35.PNG" alt="FAO" width="700">
# <img src="./img/36.PNG" alt="FAO" width="300">
# <img src="./img/37.PNG" alt="FAO" width="300">
# <img src="./img/38.PNG" alt="FAO" width="300">
# <img src="./img/39.PNG" alt="FAO" width="300">
# <img src="./img/40.PNG" alt="FAO" width="300">
# <img src="./img/41.PNG" alt="FAO" width="300">

# In[16]:


class Model(tf.keras.Model):
    def __init__(self, params):
        super().__init__()
    
        # 嵌入层
        self.embedding = tf.Variable(np.load('./vocab/word.npy'),
                                     dtype=tf.float32,
                                     name='pretrained_embedding',
                                     trainable=False,)

        # dropout 层
        self.drop1 = tf.keras.layers.Dropout(params['dropout_rate'])
        self.drop2 = tf.keras.layers.Dropout(params['dropout_rate'])
        self.drop3 = tf.keras.layers.Dropout(params['dropout_rate'])

        # LSTM 层 return_sequences 是否返回序列
        self.rnn1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['rnn_units'], return_sequences=True))
        self.rnn2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['rnn_units'], return_sequences=True))
        
        # 返回全局信息
        self.rnn3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['rnn_units'], return_sequences=False))

        self.drop_fc = tf.keras.layers.Dropout(params['dropout_rate'])
        
        # 全连接结果
        self.fc = tf.keras.layers.Dense(2*params['rnn_units'], tf.nn.relu)

        # 输出二分类
        self.out_linear = tf.keras.layers.Dense(2)

  
    def call(self, inputs, training=False):
        if inputs.dtype != tf.int32:
            inputs = tf.cast(inputs, tf.int32)
    
        batch_sz = tf.shape(inputs)[0]
        rnn_units = 2*params['rnn_units']

        # 嵌入层引用
        x = tf.nn.embedding_lookup(self.embedding, inputs)
        
        # 训练时候dropout使用，预测时候没有dropout
        x = self.drop1(x, training=training)
        x = self.rnn1(x)

        x = self.drop2(x, training=training)
        x = self.rnn2(x)

        x = self.drop3(x, training=training)
        x = self.rnn3(x)

        x = self.drop_fc(x, training=training)
        x = self.fc(x)

        x = self.out_linear(x)

        return x


# ### 速度会更快

# In[17]:


class Model(tf.keras.Model):
    def __init__(self, params):
        super().__init__()
    
        self.embedding = tf.Variable(np.load('./vocab/word.npy'),
                                     dtype=tf.float32,
                                     name='pretrained_embedding',
                                     trainable=False,)

        self.drop1 = tf.keras.layers.Dropout(params['dropout_rate'])
        self.drop2 = tf.keras.layers.Dropout(params['dropout_rate'])
        self.drop3 = tf.keras.layers.Dropout(params['dropout_rate'])

        self.rnn1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['rnn_units'], return_sequences=True))
        self.rnn2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['rnn_units'], return_sequences=True))
        
        # 与第一个模型的区别
        self.rnn3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params['rnn_units'], return_sequences=True))

        self.drop_fc = tf.keras.layers.Dropout(params['dropout_rate'])
        self.fc = tf.keras.layers.Dense(2*params['rnn_units'], tf.nn.elu)

        self.out_linear = tf.keras.layers.Dense(2)

  
    def call(self, inputs, training=False):
        if inputs.dtype != tf.int32:
            inputs = tf.cast(inputs, tf.int32)
    
        batch_sz = tf.shape(inputs)[0]
        rnn_units = 2*params['rnn_units']

        x = tf.nn.embedding_lookup(self.embedding, inputs)
        
        # 维度转换
        # 最大长度参数为1000 * 1 转换为 10*10 * 10 
        x = tf.reshape(x, (batch_sz*10*10, 10, 50))
        
        x = self.drop1(x, training=training)
        x = self.rnn1(x)
        
        # 压缩成 1/10
        x = tf.reduce_max(x, 1)

        x = tf.reshape(x, (batch_sz*10, 10, rnn_units))
        
        x = self.drop2(x, training=training)
        x = self.rnn2(x)
        x = tf.reduce_max(x, 1)

        x = tf.reshape(x, (batch_sz, 10, rnn_units))
        x = self.drop3(x, training=training)
        
        x = self.rnn3(x)
        x = tf.reduce_max(x, 1)

        x = self.drop_fc(x, training=training)
        x = self.fc(x)

        x = self.out_linear(x)

        return x


# ### 设置参数

# In[18]:


params = {
  'vocab_path': './vocab/word.txt',
  'train_path': './data/train.txt',
  'test_path': './data/test.txt',
  'num_samples': 25000,
  'num_labels': 2,
  'batch_size': 32,
  'max_len': 1000,
  'rnn_units': 200, # rnn特征数量
  'dropout_rate': 0.2,
  'clip_norm': 10., # 梯度截断，为了防止过拟合，在RNN中尤其明显
  'num_patience': 3, # early stoping，三次为提升就直接停止训练
  'lr': 3e-4,
}


# ### 用来判断进行提前停止
# - early stoping

# In[19]:


def is_descending(history: list):
    history = history[-(params['num_patience']+1):]
    for i in range(1, len(history)):
        if history[i-1] <= history[i]:
            return False
    return True  


# ### 读取语料数据表

# In[20]:


word2idx = {}
with open(params['vocab_path'],encoding='utf-8') as f:
    for i, line in enumerate(f):
        line = line.rstrip()
        word2idx[line] = i
params['word2idx'] = word2idx
params['vocab_size'] = len(word2idx) + 1


# ### 构建模型

# In[21]:


model = Model(params)
model.build(input_shape=(None, None))#设置输入的大小，或者fit时候也能自动找到


# ### 学习率自动下降
# - 链接：https://tensorflow.google.cn/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay?version=stable
# - return initial_learning_rate * decay_rate ^ (step / decay_steps)

# In[22]:



decay_lr = tf.optimizers.schedules.ExponentialDecay(params['lr'], 1000, 0.95)#相当于加了一个指数衰减函数
optim = tf.optimizers.Adam(params['lr'])
global_step = 0

history_acc = []
best_acc = .0

t0 = time.time()
logger = logging.getLogger('tensorflow')
logger.setLevel(logging.INFO)


# In[ ]:


while True:
    # 训练模型
    for texts, labels in dataset(is_training=True, params=params):
        
        #梯度带，记录所有在上下文中的操作，并且通过调用.gradient()获得任何上下文中计算得出的张量的梯度
        with tf.GradientTape() as tape:
            logits = model(texts, training=True)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            loss = tf.reduce_mean(loss)
  
        optim.lr.assign(decay_lr(global_step))
        grads = tape.gradient(loss, model.trainable_variables)
        
        #将梯度限制一下，有的时候回更新太猛，防止过拟合
        grads, _ = tf.clip_by_global_norm(grads, params['clip_norm']) 
        
        #更新梯度
        optim.apply_gradients(zip(grads, model.trainable_variables))

        if global_step % 50 == 0:
            logger.info("Step {} | Loss: {:.4f} | Spent: {:.1f} secs | LR: {:.6f}".format(
                global_step, loss.numpy().item(), time.time()-t0, optim.lr.numpy().item()))
            t0 = time.time()
        global_step += 1

    # 验证集效果
    m = tf.keras.metrics.Accuracy()

    for texts, labels in dataset(is_training=False, params=params):
        logits = model(texts, training=False)
        y_pred = tf.argmax(logits, axis=-1)
        m.update_state(y_true=labels, y_pred=y_pred)
    
    acc = m.result().numpy()
    logger.info("Evaluation: Testing Accuracy: {:.3f}".format(acc))
    history_acc.append(acc)
  
    if acc > best_acc:
        best_acc = acc
    logger.info("Best Accuracy: {:.3f}".format(best_acc))
  
    # eaaly stoping 提前停止训练
    if len(history_acc) > params['num_patience'] and is_descending(history_acc):
        logger.info("Testing Accuracy not improved over {} epochs, Early Stop".format(params['num_patience']))
        break


# In[ ]:




