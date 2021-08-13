#!/usr/bin/env python
# coding: utf-8

# # 完整问答系统实现

# ### 导入一些包

# In[29]:


import os
import jieba
from zhon.hanzi import punctuation # 中文的一些符号
import re
import sys
import time
import tensorflow as tf
import io

import warnings
warnings.filterwarnings("ignore")


# In[30]:


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu


# - punctuation:
# - ＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。

# ### 读取数据

# In[31]:


conv_path = './data/xiaohuangji50w_nofenci.conv'


# In[11]:


# 用于存储对话的列表
convs = []  

with open(conv_path, encoding='utf-8') as f:
    
    # 存储一次完整对话
    one_conv = []  
    for line in f:
        
        # 去除换行符，并将原文件中已经分词的标记去掉
        line = line.strip('\n').replace('?', '')  
        line = re.sub(r"[%s]+" % punctuation, "", line)
        
        if line == '':
            continue
        if line[0] == 'E':
            if one_conv:
                convs.append(one_conv)
            one_conv = []
        elif line[0] == 'M':
            
            # 将一次完整的对话存储下来
            one_conv.append(line.split(' ')[1])  


# ### 查看下数据

# In[12]:


convs[:5]


# ### 中文分词

# In[13]:


# 把对话分成问与答两个部分
seq = []

for conv in convs:
    
    if len(conv) == 1:
        continue
        
    if len(conv) % 2 != 0: 
        # 因为默认是一问一答的，所以需要进行数据的粗裁剪，对话行数要是偶数的
        conv = conv[:-1]
        
    for i in range(len(conv)):
        if i % 2 == 0:
            
            # 使用jieba分词器进行分词
            conv[i] = " ".join(jieba.cut(conv[i]))  
            conv[i + 1] = " ".join(jieba.cut(conv[i + 1]))
            
            # 因为i是从0开始的，因此偶数行为发问的语句，奇数行为回答的语句
            seq.append(conv[i] + '\t' + conv[i + 1])  


# ### 查看分完词后的QA 对应语料数据

# In[14]:


seq[:10]


# ### 存储结果

# In[15]:


seq_train = open('train_data/seq.data', 'w', encoding='utf-8')

for i in range(len(seq)):
    seq_train.write(seq[i] + '\n')

    if i % 1000 == 0:
        print(len(range(len(seq))), '处理进度：', i)

seq_train.close()


# In[16]:


len(seq)


# In[17]:


train_src = 'train_data/seq.data'
max_train_data_size = 50000
vocab_inp_size = 20000
enc_vocab_size = 20000
vocab_tar_size = 20000
embedding_dim = 128
units = 256
BATCH_SIZE = 32
max_length_inp, max_length_tar = 20, 20


# 预处理所有数据，给所有数据补全开头和结尾，主要目的在序列生成的时候有个统一的向量开始，也有一个统一的向量结束
def preprocess_sentence(w):
    w = 'start ' + w + ' end'
    return w

# 创建数据集合
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    
    # 对所有数据进行创建
    word_pairs = [
        [
            preprocess_sentence(w) 
            for w in l.split('\t')] 
            for l in lines[:num_examples]
            ]

    return zip(*word_pairs)


# 获取最大的长度
def max_length(tensor):
    return max(len(t) for t in tensor)


# 读取所有sample数据
def read_data(path, num_examples):
    input_lang, target_lang = create_dataset(path, num_examples)

    # 组建输入词表
    input_tensor, input_token = tokenize(input_lang)
    
    # 组建输出词表
    target_tensor, target_token = tokenize(target_lang)

    return input_tensor, input_token, target_tensor, target_token


def tokenize(lang):
    # 用来自动构建词表
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=enc_vocab_size, oov_token='unk')
    lang_tokenizer.fit_on_texts(lang)

    # 将构建的词表映射到所有的词上面转化成序列
    tensor = lang_tokenizer.texts_to_sequences(lang)

    # 向后自动补全
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=max_length_inp, padding='post')

    return tensor, lang_tokenizer


input_tensor, input_token, target_tensor, target_token = read_data(train_src, max_train_data_size)


# In[20]:


input_token


# In[11]:


# max_length_inp


# In[21]:


target_token.index_word


# In[22]:


input_token.index_word


# In[23]:


input_token.word_index.get('end')


# In[24]:


input_token.word_index.get('start', 3)


# In[16]:


input_tensor[0]


# In[25]:


input_token.word_index.get('你', 3)


# In[26]:


input_token.word_index.get('你好', 3)


# ### GRU模型概要
# <img src="./img/42.PNG" alt="FAO" width="500">
# <img src="./img/43.PNG" alt="FAO" width="300">

# - h t 表示 t 时刻的 hidden state
# - y t 表示 t 时刻的 encode ouput

# ### 构建encoder模型

# In[27]:


# 继承tf模型
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        
        # 隐藏层数量
        self.enc_units = enc_units
        
        # 根据词表数量和维度建立嵌入层
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        
        # 设置GRU模型  
        # Glorot均匀分布初始化方法, 参数从[-limit, limit]的均匀分布产生，其中limit为sqrt(6 / (fan_in + fan_out))。
        # return_sequences 为 True 输出为B，x length x units， 为False 输出为B * units
        # return_state = True 为输出两个 B * length * units，B* units
        # fan_in为权值张量的输入单元数，fan_out是权重张量的输出单元数
        self.gru = tf.keras.layers.GRU(
            self.enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def call(self, x, hidden):
        
        # 正向传播的embedding
        x = self.embedding(x)
        
        # 输出结果， 输出hidden state
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    # 初始化权重
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


# ## 实例化encoder

# In[28]:


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)


# ### 构建attention 模型

# In[21]:


class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        
        # 权重矩阵 和 hidden 维度必须保持一致
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        
        # V矩阵为了输出一个值
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size=32, 1, hidden size=256)
        # 将向量增加一个维度 
        
        # query --> hidden
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size=32, max_length=20, 1)
        # values --> enc_output (batch_size=32, max_length=20, hidden_size=256)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))
        
        # attention_weights shape == (batch_size=32, max_length=20, 1)
        # 对每个向量产生一个权重值，总和为1
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size=32, hidden_size=256)
        # 通过概率权重更新我们对输入的理解
        context_vector = attention_weights * values
        # 求和
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# In[22]:


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        
        # decode维度数量
        self.dec_units = dec_units
        
        # 建立词向量
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        
        # 设置GRU为解码模型
        self.gru = tf.keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        
        # 创建全连接
        self.fc = tf.keras.layers.Dense(vocab_size)

        # 创建Attention
        self.attention = Attention(self.dec_units)

    def call(self, x, hidden, enc_output):
        
        # x --> dec_input
        # hidden --> dec_hidden
        # enc_output --> enc_output
        # context_vector (batch_size=32, hidden_size=256), attention_weights (batch_size=32, max_length=20, 1)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape (batch_size=32, 1, hidden_size=128)
        x = self.embedding(x)
        
        # x shape (batch_size=32, 1, hidden_size= 384 =128 + 256) context + start
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        
        # output shape (32, 1, 256)
        # output state (32, 1, 256)
        output, state = self.gru(x)

        # output shape (32, 256)
        output = tf.reshape(output, (-1, output.shape[2]))

        # x shape (32, 20000)
        x = self.fc(output)
        
        # 对应结果
        # x -> predictions, 
        # state -> dec_hidden
        # attention_weights -> _
        return x, state, attention_weights


# ### 实例化Decoder

# In[23]:


decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)


# ### 优化器和loss函数定义

# In[24]:


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# ### 自定义loss

# In[25]:


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


# In[26]:


checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)


@tf.function
def train_step(inp, targ, targ_lang, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targ_lang.word_index['start']] * BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


# In[27]:


checkpoint_dir = 'checkpoint'


# ### 构建训练函数

# In[28]:


def train(save_dir):
    checkpoint_dir = save_dir
    print("Preparing data in %s" % train_src)
    steps_per_epoch = len(input_tensor) // BATCH_SIZE
    print(steps_per_epoch)
    
    # enc_hidden 一个初始化权重
    enc_hidden = encoder.initialize_hidden_state()
    
    # 获取最新的一次结果
    ckpt = tf.io.gfile.listdir(checkpoint_dir)
    if ckpt:
        # 如果存在则获取最新预训练结果
        print("reload pretrained model")
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        
    # 缓存大小
    BUFFER_SIZE = len(input_tensor)
    
    # 以最小为BUFFER_SIZE的方式打乱数据
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
    
    # 设置 BATCH_SIZE大小
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    
    # 设置 checkpoint_dir 文件目录
    checkpoint_dir = save_dir
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    start_time = time.time()

    while True:
        start_time_epoch = time.time()
        
        # 设置全局loss
        total_loss = 0
        
        
        # 按batch 获取数据
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            
            # 获取一个batch的loss
            batch_loss = train_step(inp, targ, target_token, enc_hidden)
            
            # 获取全部的loss
            total_loss += batch_loss
            
#             print('每次batch 的loss:', batch_loss.numpy())

        # 获取最新每步耗时
        step_time_epoch = (time.time() - start_time_epoch) / steps_per_epoch
        
        # 获取每步的loss
        step_loss = total_loss / steps_per_epoch
        
        # 获取最近的epoch
        current_steps =+ steps_per_epoch
        
        # 获取每步耗时
        step_time_total = (time.time() - start_time) / current_steps

        # 为了观测数据
        print('训练总步数: {} 每步耗时: {}  最新每步耗时: {} 最新每步loss {:.4f}'.format(current_steps, step_time_total, step_time_epoch,
                                                                      step_loss.numpy()))
        
        # 存下最新的checkpoint
        checkpoint.save(file_prefix=checkpoint_prefix)

        sys.stdout.flush()


# In[29]:


input_token.word_index


# ### 预测函数

# In[30]:


tf.expand_dims([target_token.word_index['start']], 1)


# In[31]:


def predict(sentence, model_path):
    
    # 获取模型checkpoint_dir目录
    checkpoint_dir = model_path
    
    # 获取最后一次训练的模型
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # 预处理数据
    sentence = preprocess_sentence(sentence)
    
    # 将所有的文字转换成数值, 未登入词设置为1
    inputs = [input_token.word_index.get(i, 1) for i in sentence.split(' ')]

    # 将所有词自动往后最大补全
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    
    # 转换成tf的向量
    inputs = tf.convert_to_tensor(inputs)

    # 设置默认结果为空
    result = ''

    # 初始化hidden向量
    hidden = [tf.zeros((1, units))]
    
    # 获取encoder后的结果, 之后会全局使用
    enc_out, enc_hidden = encoder(inputs, hidden)

    # 讲enc_hidden 结果传递给 dec_hidden
    dec_hidden = enc_hidden
    
    # 初始化一个input值 [[2]], 0表示维度坐标系，获取start后面开始
    dec_input = tf.expand_dims([target_token.word_index['start']], 0)

    # 循环最大长度次，有个最大长度的限制
    for t in range(max_length_tar):
        
        # 第一次 dec_input = 'start'
        # dec_input 第二次开始就是自己预测的结果做为 dec_input就是自己预测的结果t时刻输出的 dec_input做为t+1输入的dec_input
        
        # 第一次 dec_hidden = enc_hidden
        # dec_hidden 第二次开始就是自己预测的结果做为 dec_hidden，作为t+1输入的dec_hidden
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        # predictions 为输出的概率序列, predicted_id是序列中概率最大的索引结果
        predicted_id = tf.argmax(predictions[0]).numpy()

        # 如果预测的结果为 end 那么就停止
        if target_token.index_word[predicted_id] == 'end':
            break
            
        # 如果没有end，输出结果从词表中不断拼接
        result += target_token.index_word[predicted_id] + ' '

        # 结果重置为dec_input
        dec_input = tf.expand_dims([predicted_id], 0)

    return result


# ### 开始训练

# In[ ]:


train(checkpoint_dir)


# ### 运行app
# - Windows 可以直接运行
# - Linux jupyter 执行 ssh -p 22 -L 16006:127.0.0.1:8809 root@47.95.198.64
# - http://127.0.0.1:16006

# In[5]:


from flask import Flask, render_template, request, jsonify
import execute
import time
import threading
import jieba

"""
定义心跳检测函数
"""


def heartbeat():
    print(time.strftime('%Y-%m-%d %H:%M:%S - heartbeat', time.localtime(time.time())))
    timer = threading.Timer(60, heartbeat)
    timer.start()


timer = threading.Timer(60, heartbeat)
timer.start()


app = Flask(__name__, static_url_path="/static")


@app.route('/message', methods=['POST'])
# """定义应答函数，用于获取输入信息并返回相应的答案"""
def reply():
    # 从请求中获取参数信息
    req_msg = request.form['msg']
    # 将语句使用结巴分词进行分词
    req_msg = " ".join(jieba.cut(req_msg))

    # 调用decode_line对生成回答信息
    model_path = 'model_data'
    res_msg = execute.predict(req_msg, model_path)
    # 将unk值的词用微笑符号袋贴
    res_msg = res_msg.replace('_UNK', '^_^')
    res_msg = res_msg.strip()
    print(res_msg)
    # 如果接受到的内容为空，则给出相应的回复
    if res_msg == ' ':
        res_msg = '请与我聊聊天吧'

    return jsonify({'text': res_msg})


"""
jsonify:是用于处理序列化json数据的函数，就是将数据组装成json格式返回

http://flask.pocoo.org/docs/0.12/api/#module-flask.json
"""


@app.route("/")
def index():
    return render_template("index.html")


'''
'''
# 启动APP
if (__name__ == "__main__"):
    app.run(host='127.0.0.1', port=8809)


# In[ ]:





# In[ ]:




