#!/usr/bin/env python
# coding: utf-8

# # 文本分类

# ## 任务介绍

# <table><tr><td><img src='./images/text_classification1.png' style=width:600px;></td><td><img src='./images/text_classification2.png' style=width:600px;></td></tr></table>

# 文本分类是NLP应用领域中最常见也最重要的任务类型。

# 在我们的生活和工作中，很多事情可以转化为一个分类问题来解决，比如“今天要不要做点外卖”、“这个方案好不好”等等可以转化为二分类问题。
# 
# 在自然语言处理领域也是这样，大量的任务可以用文本分类的方式来解决，比如垃圾文本识别、涉黄涉暴文本识别、意图识别、文本匹配等。

# ## 常用模型

# ### 传统方法

# * 词袋模型
# * 朴素贝叶斯
# * SVM

# ### TextCNN

# 首次于2014年被一位韩国小哥提出使用CNN网络应用于文本分类任务上。文章链接：[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)
# 
# 网络结构如下图所示：

# <p align=center><img src="./images/TextCNN.png" alt="TextCNN" style="width:600px;"/></p>

# **输入层**：n*k矩阵，n表示句子长度，k表示词向量维度
# 
# **卷积层**：卷积核宽度与词向量维度一致，只在高度方向移动。一般有m种词窗大小的卷积方式，每种方式下有n个卷积核。每个卷积核的大小为（seq_len - filter_window_size + 1, 1），这样得到m*n个表征向量 
# 
# **池化层**：每个卷积向量通过池化层保留信息含量最多的特征值，一般选用max_pool或者k-max_pool。目标是可以将不同长度的句子通过池化得到一个定长的向量表示
# 
# **输出层**：将池化层的输出映射为n维的logits（n表示类别个数），最后套一个softmax函数得到对不同类别的预测概率

# *为什么有效？*

# 卷积层的作用：不同的kernel可以获取不同范围内词的关系，获得的是纵向的差异信息，即类似于n-gram，也就是在一个句子中不同范围的词出现会带来什么信息。比如可以使用3,4,5个词数分别作为卷积核的大小。每种卷积又有很多不同的卷积核取关注不同的地方，作为信息的互补

# ### Bi-LSTM

# RNN应用文本分类任务首次由复旦大学的邱锡鹏教授的团队于2015年发表，文中详细地阐述了RNN模型用于文本分类任务的各种变体模型。
# 
# 文章链接：[Recurrent Neural Network for Text
# Classification with Multi-Task Learning](https://www.ijcai.org/Proceedings/16/Papers/408.pdf)
# 
# 最简单的RNN用于文本分类如下图所示，示例中的是利用LSTM最后一个时间步的向量表示连接接全连接层。

# <p align=center><img src="./images/rnn.png" alt="rnn" style="width:600px;"/></p>

# 然后我们可以对RNN网络进行扩展，一般RNN网络是对句子进行从左到右的顺序进行编码，我们也可以对句子按照从右到左的顺序进行编码。
# 
# 然后把前向和后向的向量表示拼接在一起，这样每一个token同时包含其周围上下文的信息，token的表征更准确且丰富。
# 
# 简单的网络示意图如下所示：

# <p align=center><img src="./images/birnn.png" alt="birnn" style="width:600px;"/></p>

# 我们现在看到的RNN网络很浅，只有一层。我们也可以堆叠多层的RNN层构造出更深的网络。
# 
# 如下图所示，第i+1层RNN的输入来自于第i层RNN的输出

# <p align=center><img src="./images/mlrnn.png" alt="mlrnn" style="width:600px;"/></p>

# 原始的RNN网络存在梯度消失的问题，而LSTM的单元结构让RNN更容易保存前面时间步的信息缓解梯度消失问题。
# 
# 更多详细的原理介绍可以查看这篇[博文](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)

# ### Bi-LSTM Attention

# 从前面介绍的几种方法，可以自然地得到文本分类的框架，就是先基于上下文对token编码，然后pooling出句子表示再分类。
# 
# 在最终池化时，max-pooling通常表现更好，因为文本分类经常是主题上的分类，从句子中一两个主要的词就可以得到结论，其他大多是噪声，对分类没有意义。
# 
# 而到更细粒度的分析时，max-pooling可能又把有用的特征去掉了，这时便可以用attention进行句子表示的融合。

# <p align=center><img src="./images/bi_lstm_att.jpg" alt="bi_lstm_att" style="width:600px;"/></p>

# <center>$M=tanh(H)$</center>
# <center>$\alpha=softmax(w^TM)$</center>
# <center>$r=H\alpha^T$</center>

# 注意力机制，简单来说，对所有时间步的隐藏状态进行加权，把注意力集中到整段文本中比较重要的信息。
# 
# 注意力机制的计算过程大致可以概况如下：
# 
# 第一步：`query`和`key`进行相似度计算，得到权值。这里定义的相似度计算方式为矩阵相乘
# 
# 第二步：将权值进行归一化，得到直接可用的权重
# 
# 第三步：将权重和`value`进行加权求和

# # 数据处理

# 今天主要涉及的实战项目为今日头条中文新闻（短文本）分类
# 
# 该数据集来自今日头条的新闻版块，共提取了15个类别的新闻，包括旅游，教育，金融，军事等。

# In[1]:


import json
from collections import Counter, defaultdict
from pathlib import Path
from pprint import pprint

import jieba_fast as jieba
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.legacy.data import Dataset, Example, Field, Iterator, BucketIterator
from torchtext.vocab import Vectors

# 手动初始化
jieba.initialize()
plt.rcParams['figure.figsize'] = (15, 8)


# ## 加载数据

# In[2]:


# 读取json数据
def load_data(fn, has_label=True):
    texts, labels = [], []
    for line in open(fn, encoding="utf8"):
        line = json.loads(line.strip())
        texts.append(line["sentence"])
        if has_label:
            labels.append(line["label"])
    if has_label:
        return texts, labels
    else:
        return texts


x_train, y_train = load_data("data/tnews/train.json")
x_dev, y_dev = load_data("data/tnews/dev.json")
x_test = load_data("data/tnews/test.json", False)
print(f"训练集/开发集/测试集 样本数: {len(x_train)}/{len(x_dev)}/{len(x_test)}")

label_map = {}
for line in open("data/tnews/labels.json"):
    line = json.loads(line.strip())
    label_map[line["label"]] = line["label_desc"].split("_")[-1]
print("\n标签集合：")
print(label_map)

print(f"\n文本示例：{x_train[0]} - 类别：{label_map[y_train[0]]}")


# ## 数据探查

# **最大长度**

# In[3]:


# NLP的任务需要关注下句子的长度
all_sentences = x_train + x_dev + x_test
sentence_lengths = list(map(len, all_sentences))
print(f"50%句子长度: {np.mean(sentence_lengths):.1f}")
print(f"95%句子长度: {np.percentile(sentence_lengths, 95):.1f}")
print(f"99%句子长度: {np.percentile(sentence_lengths, 99):.1f}")
print(f"最大句子长度: {np.max(sentence_lengths)}")


# In[4]:


sns.histplot(sentence_lengths)


# 我们可以看到句子长度集中在30-40之间

# 如果构建以字为粒度的输入，长度设置为48是一个不错的选择

# In[5]:


max_seq_len_char = 40


# 超过99%的新闻标题长度不超过39个字符 => 合理设置文本序列的最大长度可以加速模型的训练

# **类目样本数**

# In[6]:


label_cnt = {
    label_map[label]: cnt for label, cnt in Counter(y_train).items()
}
print(sorted(label_cnt.items(), key=lambda x: -x[1]))


# 故事和股票类新闻样本较少，可能会存在样本不均衡的问题

# **词汇表**

# 收集词汇表，为后续使用预训练的词向量做准备

# 这里我们选择使用高效的Jieba分词，当然你也可以尝试替换为其他精度更高的分词工具

# In[7]:


# 这里我们只使用训练集中出现的token构成词汇表
vocabs = list(set([token for sent in x_train for token in jieba.cut(sent)]))
# 这里我们预留几个通用token
# [UNK]表示未登录词，增加模型的鲁棒性
# [PAD]表示后续补全序列用的token
vocabs = ["<pad>", "<unk>"] + vocabs

vocab2idx = {token: idx for idx, token in enumerate(vocabs)}
idx2vocab = {idx: token for token, idx in vocab2idx.items()}

print(f"总共有{len(vocab2idx)}个词")


# ## 数据处理

# ### 文本id化

# 我们首先要将商品标题进行分词，由于机器并不认识语言词汇，我们还需要根据词汇表转化为数字id

# In[8]:


def sent2idx(sentences):
    # 如果词典里面不存在，就换成unk
    return [[vocab2idx.get(token, vocab2idx["<unk>"]) for token in jieba.cut(sent)] for sent in sentences]


x_train_id = sent2idx(x_train)
x_dev_id = sent2idx(x_dev)
x_test_id = sent2idx(x_test)

print("转换前句子：", x_train[1])
print("分词结果：", [idx2vocab[idx] for idx in x_train_id[1]])
print("转换后句子：", x_train_id[1])


# 我们再来看看分词后句子序列的长度

# In[9]:


all_sentences_id = x_train_id + x_dev_id + x_test_id
sentence_lengths = list(map(len, all_sentences_id))
print(f"50%句子长度: {np.mean(sentence_lengths):.1f}")
print(f"95%句子长度: {np.percentile(sentence_lengths, 95):.1f}")
print(f"99%句子长度: {np.percentile(sentence_lengths, 99):.1f}")
print(f"最大句子长度: {np.max(sentence_lengths)}")


# In[10]:


sns.histplot(sentence_lengths)


# 我们可以看到基本上集中在20个字符左右

# In[11]:


max_seq_len = 24


# 设置最大序列长度为24可能是一个不错的选择

# ### 标签id化

# 接着我们还需要对样本标签进行id化

# In[12]:


label2idx = {label: idx for idx, label in enumerate(label_map.keys())}
idx2label = {idx: label for label, idx in label2idx.items()}
sorted_labels = [label_map[label] for label, idx in label2idx.items()]

print(label2idx)
print(sorted_labels)


# In[13]:


y_train = [label2idx[label] for label in y_train]
y_dev = [label2idx[label] for label in y_dev]

print(y_train[0:10])


# ## 预训练词向量

# NLP模型一般会使用预训练的词向量来对每个token对应的向量进行初始化，来提升模型的表现

# 这里我们使用[腾讯词向量](https://ai.tencent.com/ailab/nlp/en/data/Tencent_AILab_ChineseEmbedding.tar.gz)（目前中文效果最好的）。
# 
# 如果你有大量自己领域的文本语料，也可以通过[Gensim](https://radimrehurek.com/gensim/)工具训练该领域的词向量

# 由于这份词向量文件特别大，每次读取它会花费不少时间
# 
# 我们这里定义个`save_small_embeddings`函数，只保存当前数据集所需的词向量

# In[14]:


def save_small_embedding(vocab, embed_file, output_file):
    fin = open(embed_file, "r", encoding="utf-8", newline="\n", errors="ignore")
    try:
        _, d = map(int, fin.readline().split())
    except Exception:
        print("Please make sure the embed file is gensim-formatted")

    def gen():
        for line in fin:
            token = line.rstrip().split(" ", 1)[0]
            if token in vocab:
                yield line

    result = [line for line in gen()]
    rate = 1 - len(result) / len(vocab)
    print("oov rate: {:4.2f}%".format(rate * 100))

    with open(output_file, "w", encoding="utf-8") as fout:
        fout.write(str(len(result)) + " " + str(d) + "\n")
        for line in result:
            fout.write(line)


# In[15]:


embed_dir = "../../work/embeddings/tencent"  # 根据自己的存储路径进行修改
new_embed_file = "data/tnews_jieba_tencent_embeddings.txt"
if not Path(new_embed_file).exists():
    save_small_embedding(vocab2idx, f"{embed_dir}/Tencent_AILab_ChineseEmbedding.txt", new_embed_file)


# 加载缩小后的向量文件，需要注意的是，如果是出现未登录词的话，我们赋予其一个符合均匀分布的随机向量

# In[16]:


def load_vectors(fname, token2id):
    fin = open(fname, "r", encoding="utf-8", errors="ignore")
    _, d = map(int, fin.readline().split())
    embeddings = {}
    for line in fin:
        tokens = line.rstrip().split(" ")
        embeddings[tokens[0]] = np.asarray(tokens[1:], dtype="float32")

    scale = 0.25
    # 对于不出现的词进行平均分布 uniform
    embedding_matrix = np.random.uniform(-scale, scale, [len(token2id), d])
    embedding_matrix[0] = np.zeros(d)  # 赋予<pad>为零向量
    cnt = 0
    for token, i in token2id.items():
        embedding_vector = embeddings.get(token)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            cnt += 1
    print("oov rate: {:4.2f}%".format(1 - cnt / len(token2id)))
    return embedding_matrix


# In[17]:


word_embeddings = load_vectors("data/tnews_jieba_tencent_embeddings.txt", vocab2idx)


# In[18]:


word_embeddings.shape


# ## 数据迭代器

# ### 自定义迭代器

# 我们需要实现这样一个数据迭代器：
# 1. 能够每个batch提供样本和标签
# 2. 能够随机打乱样本顺序给出样本（训练集），也可以按顺序给出（验证集和测试集）
# 3. 这个数据迭代器可以根据不同数据集重构迭代逻辑
# 
# 这是一个通用的深度学习模型的数据迭代器

# In[19]:


class DataGenerator(object):
    def __init__(self, data, batch_size=32, max_len=None, return_length=False):
        self.data = data
        self.batch_size = batch_size  # 每一批的样本个数
        self.max_len = max_len  # 序列最大长度
        self.return_length = return_length  # 是否返回序列长度
        # 数据集的batch个数
        # 若不能被batch_size整除，则将最后所有剩余样本加入一个batch中，batch个数加1
        if hasattr(self.data, "__len__"):
            self.steps = len(self.data) // self.batch_size
            if len(self.data) % self.batch_size != 0:
                self.steps += 1
        else:
            self.steps = None

    def __len__(self):
        return self.steps

    def sample(self, random=False):
        if random:

            # 随机采样每个样本的index，按照打乱的index顺序访问数据样本
            def generator():
                for i in np.random.permutation(len(self.data)):
                    yield self.data[i]

            data = generator()
        else:
            # 按顺序依次访问数据样本
            data = iter(self.data)

        # 不断迭代data这个迭代器
        # 每次返回一个数据集是否结束迭代的标记和一个样本
        d_current = next(data)
        for d_next in data:
            yield False, d_current
            d_current = d_next

        yield True, d_current

    def __iter__(self, random=False):
        raise NotImplementedError

    def forfit(self, random=True):
        while True:
            for d in self.__iter__(random):
                yield d


# 按照自己的需要实现迭代函数
class TextDataGenerator(DataGenerator):
    def __iter__(self, random=False):
        batch_token_ids, batch_labels = [], []
        batch_lengths = []
        for is_end, (token_ids, label) in self.sample(random):
            batch_token_ids.append(token_ids)
            batch_labels.append(label)
            if self.return_length:
                batch_lengths.append(len(token_ids))
            if len(batch_token_ids) == self.batch_size or is_end:
                # 需要保证每个batch的数据是一样长的
                batch_token_ids = self.sequence_padding(batch_token_ids, self.max_len)
                batch_labels = np.array(batch_labels)
                if self.return_length:
                    batch_lengths = np.array(batch_lengths)
                    yield (batch_token_ids, batch_labels, batch_lengths)
                else:
                    yield (batch_token_ids, batch_labels)
                batch_token_ids, batch_labels = [], []
                batch_lengths = []

    def sequence_padding(self, inputs, max_len=None, value=0, mode="post"):
        """将序列padding到同一长度"""
        # 如果没有给定最大长度的话，取这个batch中最长的序列的长度
        if max_len is None:
            max_len = max([len(x) for x in inputs])

        outputs = []
        for x in inputs:
            if len(x) < max_len:
                if mode == "post":
                    # 序列后填充
                    pad_width = (0, max_len - len(x))
                elif mode == "pre":
                    # 序列前填充
                    pad_width = (max_len - len(x), 0)
                else:
                    raise ValueError("`mode`参数必须是`post`或`pre`")
                # 以固定值进行填充
                try:
                    x = np.pad(x, pad_width, "constant", constant_values=value)
                except:
                    print(len(x), pad_width)
                    continue
            else:
                # 截断到最大序列长度
                x = np.array(x[:max_len])
            outputs.append(x)

        return np.array(outputs)


# In[20]:


# 组合文本和标签数据，以符合数据生成器的输入格式
train_data = list(zip(x_train_id, y_train))
print(train_data[0])

train_dataset = TextDataGenerator(train_data, batch_size=32, max_len=48)
batch_tokens, batch_labels = next(iter(train_dataset))
print(batch_tokens.shape, batch_labels.shape)


# ### Torchtext

# [Torchtext](https://pytorch.org/text/stable/index.html)包是由官方维护的，集成NLP数据处理通用工具和常见的NLP数据集，可以使我们实施NLP项目时数据处理更规范

# <p align=center><img src="./images/torchtext.png".png" alt="torchtext" style="width:800px;"/></p>

# torchtext预处理流程：
# 1. 定义Field：声明如何处理数据
# 2. 定义Dataset：得到数据集，此时数据集里每一个样本是一个经过Field声明的预处理后的标准格式
# 3. 建立vocab：在这一步建立词汇表，词向量
# 4. 构造迭代器：构造迭代器，用来分批次训练模型

# 更多详细使用方法请查阅[官方文档](https://torchtext.readthedocs.io/en/latest/index.html)

# **Fields**

# 一个能够加载、预处理和存储文本数据和标签的对象。我们可以用它根据训练数据来建立词表，加载预训练的词向量等等

# 常用的Field参数：
# 
# * sequential：默认为True，是否是序列数据，如果不是就不使用分词器
# * use_vocab：默认为True，是否使用Vocab对象进行id化；若不使用，确保数据已经是数字类型
# * tokenize：默认为`string.split`，即按字符切分，也可以传入自定义的分词器
# * fix_length：默认为None，每个batch使用动态长度。对于某些需要固定长度输入的网络，需要手动设置下
# * lower: 默认为False. 字符串转为小写

# In[21]:


TEXT = Field(
    sequential=True,
    tokenize=lambda x: list(jieba.cut(x)),
    lower=True,
    fix_length=max_seq_len
)

LABEL = Field(
    sequential=False,
    use_vocab=False
)

TEXT_CHAR = Field(
    sequential=True,
    tokenize=lambda x: list(x),
    lower=True,
    fix_length=max_seq_len_char
)


# **Dataset**

# 基于`Dataset`类包装我们自己的数据集类，核心就是使用Field来定义数据组成形式，得到数据集

# In[22]:


class MyDataset(Dataset):
    def __init__(self, fn, text_field, label_field, test=False):
        # 数据处理操作格式
        fields = [("text", text_field), ("label", label_field)]

        # 这里我们可以把数据读取逻辑也写入dataset中
        examples = []
        for line in open(fn, encoding="utf8"):
            line = json.loads(line.strip())
            text = line["sentence"]
            if test:
                examples.append(Example.fromlist([text, None], fields))
            else:
                label = label2idx[line["label"]]
                examples.append(Example.fromlist([text, label], fields))

        # 上面是一些预处理操作，此处调用super来调用父类构造方法，产生标准Dataset实例
        super(MyDataset, self).__init__(examples, fields)


# In[23]:


train_dataset_torchtext = MyDataset("data/tnews/train.json", TEXT, LABEL)
for batch_data in train_dataset_torchtext[:3]:
    print(batch_data.text, batch_data.label)


# In[24]:


test_dataset_torchtext = MyDataset("data/tnews/test.json", TEXT, LABEL, True)
for batch_data in test_dataset_torchtext[:3]:
    print(batch_data.text)


# **Embedding**

# 下面介绍如何在torchtext中使用预训练的词向量，进而传送给神经网络模型进行训练

# In[25]:


# 加载预训练的词向量
vectors = Vectors(name="data/tnews_jieba_tencent_embeddings.txt")
# 指定Vector缺失值的初始化方式，oov词的初始化方式
vectors.unk_init = nn.init.uniform_
# 这里只使用训练集的数据进行词汇表的构建
TEXT.build_vocab(train_dataset_torchtext, vectors=vectors)


# In[26]:


# 统计词频
TEXT.vocab.freqs.most_common(10)


# **迭代器**
# 
# * **Iterator**：保持数据样本顺序不变来构建批数据，适用测试集
# 
# * **BucketIterator**：自动将相似长度的示例批处理在一起，最大程度地减少所需的填充量，适合训练集和验证集
# 
# 参数解释：
# 
# * sort：是否对样本进行长度排序
# * sort_within_batch：是否对每个批次内的样本进行长度排序
# * sort_key：每个小批次内数据的排序方式

# In[27]:


def data_iter(text_field, label_field, train_bs=32, eval_bs=128, is_char_token=False):
    train = MyDataset("data/tnews/train.json", text_field=text_field, label_field=label_field, test=False)
    valid = MyDataset("data/tnews/dev.json", text_field=text_field, label_field=label_field, test=False)
    test = MyDataset("data/tnews/test.json", text_field=text_field, label_field=None, test=True)
    # 如果是char粒度的输入，不存在预训练的词向量
    word_embeddings = None
    if not is_char_token:
        word_embeddings = text_field.vocab.vectors

    # device=-1表示使用cpu进行数据集迭代
    device = 0 if torch.cuda.is_available() else -1
    train_iter = BucketIterator(
        dataset=train, batch_size=train_bs, shuffle=True, train=True,
        sort_key=lambda x: len(x.text), sort=False, device=device)
    val_iter = BucketIterator(valid, eval_bs, train=False, sort_key=lambda x: len(x.text),
                              sort=False, device=device)
    test_iter = Iterator(test, 128, shuffle=False, train=False, sort=False, device=device)

    return train_iter, val_iter, test_iter, word_embeddings


# In[28]:


train_iter, val_iter, test_iter, word_embeddings_torchtext = data_iter(TEXT, LABEL)


# In[29]:


for batch_data in train_iter:
    print(batch_data)
    break


# In[30]:


# assert False, "check data process codes work"


# # 模型构建

# ## TextCNN模型

# 这篇论文 [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1510.03820.pdf) 详细地阐述了关于TextCNN模型的调参心得。
# 
# * 卷积核大小（高度）：1~10
# * 特征图个数：100~600
# * dropout丢弃率：0~0.5
# * 尝试不同的激活函数，ReLU是一个不错的选择
# * 使用最大池化层

# 我们这里要开始搭建CNN网络了

# 模型结构：
# 
# 1. 嵌入层（embedding）：嵌入对于任何NLP相关任务都非常重要，因为它以数字格式表示一个单词。嵌入层创建一个查找表，其中每一行代表一个词的嵌入。嵌入层将整数序列转换为密集向量表示。
#     * vocab_size：词表大小
#     * embedding_dim：嵌入层维度，表示一个词的维度数
# 2. 卷积池化层（TextCNN）：通过对不同粒度的NGRAM窗口进行卷积，得到局部上下文特征
#     * channel_num：通道个数，NLP任务默认为1，根据你输入的词向量种类决定
#     * filter_num: 每一类卷积核个数，来提取词向量不同位置上的特征
#     * filter_sizes: 宽度默认为词向量的维度，卷积核高度（NLP任务重宽度默认为词向量维数）
# 3. 线性层（Linear)：指的是Dense层，全连接层，将CNN网络的输出映射为类别个数的输出维度

# In[31]:


class CNNModelConfig:
    def __init__(self, label_map, vocab=None, embedding_type="non_static",
                 embeddings=None, embed_dim=200, embedding_channels=1):
        if embedding_type == "scratch":
            self.vocab_size = len(vocab)
            self.embedding_dim = embed_dim
            self.embeddings = None  # 使用随机初始化的词向量
        else:
            self.vocab_size = embeddings.shape[0]  # 词汇表大小
            self.embedding_dim = embeddings.shape[1]  # 词向量维度
            self.embeddings = torch.FloatTensor(embeddings)  # numpy.ndarray -> FloatTensor
        self.embedding_type = embedding_type
        self.static = True if embedding_type == "static" else False  # 词向量是否随网络训练进行更新

        self.class_num = len(label_map)  # 样本标签个数
        self.channel_num = 2 if embedding_type == "multichannel" else 1  # 输入通道数
        self.filter_num = 128  # 特征图个数
        self.filter_sizes = [2, 3, 4, 5]  # 过滤器高度（纵向）
        self.drop_rate = 0.2  # 神经元丢弃率


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()

        # 构建词向量层，输入参数分别为词汇表大小和词向量维度
        self.embed_layer = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.embed_layer2 = None
        if config.embeddings is not None:
            # 加载预训练词向量
            if config.channel_num == 1:
                # freeze=True 表示该词向量不随网络训练而更新
                self.embed_layer = self.embed_layer.from_pretrained(config.embeddings, freeze=config.static)
            elif config.channel_num == 2:
                # 输入变为两个通道，其一为词向量更新，其二为词向量不更新
                self.embed_layer = self.embed_layer.from_pretrained(config.embeddings, freeze=False)
                self.embed_layer2 = nn.Embedding(config.vocab_size, config.embedding_dim)
                self.embed_layer2 = self.embed_layer2.from_pretrained(config.embeddings, freeze=True)

        # 根据过滤器不同的尺寸构建多个卷积层，对输入的句子向量通过卷积操作进行Ngram特征抽取
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=config.channel_num,
                    out_channels=config.filter_num,
                    kernel_size=(size, config.embedding_dim)
                )
                for size in config.filter_sizes
            ]
        )
        # 丢弃层，随机丢弃一定比例的特征值，来提升模型的鲁棒性
        self.dropout = nn.Dropout(config.drop_rate)
        # 全连接层，将网络最终的输出变为样本标签个数
        self.fc = nn.Linear(len(config.filter_sizes) * config.filter_num, config.class_num)

    def forward(self, x):
        # x shape: [batch_size, max_len]
        if self.embed_layer2:
            # x shape: [batch_size, channel_num, seq_len, embed_dim]
            x = torch.stack([self.embed_layer(x), self.embed_layer2(x)], dim=1)
        else:
            # x shape: [batch_size, max_len, embed_dim]
            x = self.embed_layer(x)
            # x shape: [batch_size, 1, max_len, embed_dim]
            x = x.unsqueeze(1)
        # conv_out = (seq_len - filter_size + 2 * padding) / stride + 1
        # each of x shape: [batch_size, filter_num, conv_out]
        # 没有参数，也可以直接F.relu
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        # each of x shape: [batch_size, filter_num]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        # x shape: [batch_size, filter_num * len(filter_sizes)]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        # logits shape: [batch_size, class_num]
        logits = self.fc(x)
        return logits


# ## Bi-LSTM模型

# RNN家族的网络以序列顺序建模，符合人类阅读文字的习惯，天生适合构建NLP的学习任务。
# 
# 这里我们来搭建上文提到的双向LSTM网络

# 模型结构：
# 
# 1. 嵌入层：与上文的TextCNN一致
# 2. LSTM层：LSTM是RNN 的一种变体，能够捕获序列内部的长期依赖关系
#     * input_size : 输入维度
#     * hidden_size : 隐藏节点的数量
#     * num_layers : 要堆叠的层数
#     * batch_first ：如果为 True，则输入和输出张量提供为 (batch, seq, feature)
#     * dropout：如果非零，则在除最后一层之外的每个LSTM层的输出上引入一个Dropout 层，dropout概率等于dropout。 默认值：0
#     * 双向：如果为真，则引入双向 LSTM
# 3. 线性层：与上文的TextCNN的设置一致

# **如何处理变长序列**

# 我们在数据迭代器中定义了`sequence_padding`方法，通过填充pad值使得每个batch中的序列长度保持一致。
# 
# 但这样，序列里面填充了很多无效值0，将填充值0喂给RNN进行前向计算，不仅浪费计算资源，最后得到的值可能还会存在误差。
# 
# 将序列送给RNN进行处理之前，需要采用`pack_padded_sequence`进行压缩，压缩掉无效的填充值。
# 序列经过RNN处理之后的输出仍然是压紧的序列，需要采用`pad_packed_sequence`把压紧的序列再填充回来，便于进行后续的处理。

# 更多关于变长序列的处理细节可查看此[文章](https://zhuanlan.zhihu.com/p/342685890)

# In[32]:


class RNNModelConfig(object):
    def __init__(self, label_map, vocab=None, embedding_type="non_static",
                 embeddings=None, embed_dim=200, num_lstm_layers=1, num_directions=2):
        if embedding_type == "scratch":
            self.vocab_size = len(vocab)
            self.embedding_dim = embed_dim
            self.embeddings = None  # 使用随机初始化的词向量
        else:
            self.vocab_size = embeddings.shape[0]  # 词汇表大小
            self.embedding_dim = embeddings.shape[1]  # 词向量维度
            self.embeddings = torch.FloatTensor(embeddings)  # numpy.ndarray -> FloatTensor
        self.embedding_type = embedding_type
        self.static = True if embedding_type == "static" else False  # 词向量是否随网络训练进行更新

        self.class_num = len(label_map)  # 样本标签个数
        self.num_lstm_layers = num_lstm_layers  # lstm网络层数
        self.num_directions = num_directions  # 1表示单向，2表示双向
        self.final_lstm_out_type = "last"  # lstm输出类型，"mean"表示取各个时间步的向量的平均，"last"表示取最后一个时间步的向量
        self.bi_lstm_out_type = "add"  # 双向lstm前后向输出处理方式，"add"表示向量相加，"cat"表示向量拼接
        self.hidden_dim = 128  # lstm隐藏状态大小
        self.lstm_drop_rate = 0 if num_lstm_layers == 1 else 0.1  # lstm内部神经元丢弃率
        self.drop_rate = 0.1
        self.batch_first = True  # 默认lstm的输入第一维为序列长度，开启后变为batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # 获取当前机器是否支持cuda


class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()

        self.embed_layer = nn.Embedding(config.vocab_size, config.embedding_dim)
        if config.embeddings is not None:
            self.embed_layer = self.embed_layer.from_pretrained(config.embeddings, freeze=config.static)

        self.bilstm = nn.LSTM(
            #把所有配置想都配置在config类里面
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_lstm_layers,
            dropout=config.lstm_drop_rate,
            batch_first=config.batch_first,
            bidirectional=(config.num_directions == 2)
        )
        if config.bi_lstm_out_type == "add":
            fc_hidden_dim = config.hidden_dim
        else:
            fc_hidden_dim = config.hidden_dim * config.num_directions
        self.dropout = nn.Dropout(config.drop_rate)
        self.lstm_fc = nn.Linear(fc_hidden_dim, fc_hidden_dim // 2)
        self.label_fc = nn.Linear(fc_hidden_dim // 2, config.class_num)
        self.conf = config

    def init_hidden(self, batch_size):
        # 包含批次中每个元素的初始隐藏状态
        h0 = torch.rand(self.conf.num_directions * self.conf.num_lstm_layers, batch_size, self.conf.hidden_dim)
        # 包含批次中每个元素的初始细胞状态
        c0 = torch.rand(self.conf.num_directions * self.conf.num_lstm_layers, batch_size, self.conf.hidden_dim)
        h0 = h0.to(self.conf.device)
        c0 = c0.to(self.conf.device)
        return (h0, c0)

    def forward(self, x, lengths=None):
        # x shape: [batch_size, max_len]
        batch_size = x.shape[0]
        # embeds shape: [batch_size, max_len, embed_dim]
        embeds = self.embed_layer(x)

        # 初始化隐状态，默认初始隐状态为零向量
        # 如果你有特殊的初始化需求，可以自定义隐状态
        # (h0, c0) = self.init_hidden(batch_size)

        if not self.conf.batch_first:
            # 当lstm层的batch_first=False，需要进行矩阵维度交换，让seq_len维度为第一维
            # embeds shape: [max_len, batch_size, embed_dim]
            embeds = embeds.permute(1, 0, 2)  # 第0维和第1维进行交换

        if lengths is not None:
            # 重组输入，句子序列变为扁平的整个batch的token序列
            # x = [[1,2,3,4,0], [1,2,3,0,0]]
            # lengths = [4,3]
            # packed_tokens = [1,2,3,4,1,2,3]
            # shape: [n_tokens, embed_dim]
            embeds = pack_padded_sequence(embeds, lengths=lengths,
                                          batch_first=self.conf.batch_first, enforce_sorted=False)

        # 若是双向的lstm网络，lstm_out默认将前后向输出的向量拼接在一起
        # lstm_out shape: [seq_len, batch_size, hidden_dim * num_directions]
        # 如果传入lengths，lstm_out shape: [n_tokens, hidden_dim * num_directions]
        lstm_out, (hn, cn) = self.bilstm(embeds)

        if lengths is not None:
            # 为了使得LSTM的输出结果和标签对齐，将压紧的序列重新展开并填充pad值
            # shape: [batch_size, seq_len, hidden_dim * num_directions]
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=self.conf.batch_first)

        if self.conf.num_directions == 2:
            # 将双向lstm的输出拆分为前向输出和后向输出
            (forward_out, backward_out) = torch.chunk(lstm_out, 2, dim=2)
            if self.conf.bi_lstm_out_type == "add":  # 向量相加
                lstm_out = forward_out + backward_out  # [seq_len, batch_size, hidden_size]
            elif self.conf.bi_lstm_out_type == "cat":  # 向量拼接
                lstm_out = torch.cat([forward_out, backward_out], axis=2)  # [seq_len, batch_size, hidden_size * 2]

        if self.conf.final_lstm_out_type == "mean":
            # 对序列长度对应的维数的向量进行均值计算
            if self.conf.batch_fist:
                lstm_out = torch.mean(lstm_out, 1)
            else:
                lstm_out = torch.mean(lstm_out, 0)
        else:
            # 取最后一个时间步的向量作为输出
            if self.conf.batch_first:
                lstm_out = lstm_out[:, -1, :]
            else:
                lstm_out = lstm_out[-1, :, :]

        lstm_out = self.dropout(lstm_out)

        logits = self.label_fc(self.lstm_fc(lstm_out))

        return logits


# ## Bi-LSTM-Attention 模型

# w是上下文向量，表示不同时间步，随机初始化并随着训练更新。最后得到句子表示r，再进行分类

# In[33]:


class BiLSTMAtt(nn.Module):
    def __init__(self, config):
        super(BiLSTMAtt, self).__init__()

        if config.embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embeddings, freeze=config.static)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        self.bilstm = nn.LSTM(config.embedding_dim, config.hidden_dim, config.num_lstm_layers,
                              bidirectional=True, batch_first=True, dropout=config.drop_rate)
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(config.hidden_dim * config.num_directions))
        self.hidden_fc = nn.Linear(config.hidden_dim * config.num_directions, config.hidden_dim)
        self.label_fc = nn.Linear(config.hidden_dim, config.class_num)

    def forward(self, x):
        embeds = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        H, _ = self.bilstm(embeds)  # [batch_size, seq_len, hidden_dim * num_directions]

        M = self.tanh(H)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [batch_size, seq_len]
        r = H * alpha  # [batch_size, seq_len, hidden * num_directions]
        out = torch.sum(r, 1)  # [batch_size, hidden_dim * num_directions]
        out = F.relu(out)
        out = self.hidden_fc(out)  # [batch_size, hidden_dim]
        logits = self.label_fc(out)  # [batch_size, class_num]
        return logits


# # 模型训练

# ## 通用训练过程

# ### 模型评估

# 定义模型的评估过程，主要包含如下内容：
# 1. 模型训练过程中的验证集评估
# 2. 比准确率更详细的分类报告和混淆矩阵

# In[34]:


def evaluation(dataset, model, train_config):
    device = train_config.device
    model.eval()  # 沿用batch normalization的值，并不使用drop out

    corrects, avg_loss = 0, 0
    total_cnt = 0  # 总样本数
    y_trues, y_preds = [], []  # 用于所有的存放正确标签和预测标签

    for batch_data in dataset:
        batch_lengths = None
        if isinstance(dataset, BucketIterator) or isinstance(dataset, Iterator):
            batch_tokens = batch_data.text.permute(1, 0)
            batch_labels = batch_data.label
        else:
            if dataset.return_length:
                (batch_tokens, batch_labels, batch_lengths) = batch_data
            else:
                (batch_tokens, batch_labels) = batch_data
            batch_tokens = torch.from_numpy(batch_tokens)
            batch_labels = torch.from_numpy(batch_labels)
            if batch_lengths is not None:
                batch_lengths = torch.from_numpy(batch_lengths)

        labels = batch_labels.cpu().detach().numpy()  # 将tensor转换为numpy的array格式
        y_trues.extend(list(labels))

        batch_tokens = batch_tokens.to(device)
        batch_labels = batch_labels.to(device)

        # 获取模型预测结果
        if batch_lengths is not None:
            batch_lengths = batch_lengths.to(device)
            logits = model(batch_tokens, batch_lengths)
        else:
            logits = model(batch_tokens)
        probs = F.softmax(logits, dim=0)  # 转换为预测概率
        predictions = torch.argmax(probs, axis=1).cpu().detach().numpy()  # 将tensor转换为numpy的array格式
        y_preds.extend(list(predictions))

        loss = train_config.loss_fn(logits, batch_labels)  # 计算交叉熵损失

        avg_loss += loss.item()
        corrects += sum(predictions == labels)
        total_cnt += batch_tokens.shape[0]

    avg_loss /= len(dataset)
    acc = corrects / total_cnt * 100
    print(f"Evaluation - loss:{avg_loss:.4f} acc:{acc:.2f}")
    return acc, y_trues, y_preds


# 为了查看模型在不同类别上的表现如何，我们将创建一个混淆矩阵，观察每种类别的错分情况

# In[35]:


def plot_confusion_matrix(y_trues, y_preds, all_categories):
    # 调用sklearn的api，获取混淆矩阵
    confusion = confusion_matrix(y_trues, y_preds)

    # 通过将每一行除以其总和来标准化
    confusion = confusion / confusion.sum(axis=1, keepdims=True)

    # 设置图像基本参数
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion)
    fig.colorbar(cax)

    # 设置坐标轴
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # 在每个刻度处强制标签
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


# 使用sklearn.metrics.classification_report来查看更全面的评估结果
# 
# 可以看到不同类目的precision，recall和f1分数

# In[36]:


def show_classification_report(y_trues, y_preds):
    print(classification_report(y_trues, y_preds, target_names=sorted_labels, digits=4))


# ### 模型预测

# In[37]:


def predict(dataset, model, texts, device="cpu"):
    model.eval()

    y_preds = []
    y_probs = []
    for batch_data in dataset:
        # 和模型评估差不多的流程，这里不需要获取label输入
        batch_lengths = None
        if isinstance(dataset, BucketIterator) or isinstance(dataset, Iterator):
            batch_tokens = batch_data.text.permute(1, 0)
        else:
            if dataset.return_length:
                (batch_tokens, batch_lengths) = batch_data
            else:
                batch_tokens = batch_data
            batch_tokens = torch.from_numpy(batch_tokens)
            if batch_lengths is not None:
                batch_lengths = torch.from_numpy(batch_lengths)
        batch_tokens = batch_tokens.to(device)

        # 获取模型预测结果
        if batch_lengths is not None:
            batch_lengths = batch_lengths.to(device)
            logits = model(batch_tokens, batch_lengths)
        else:
            logits = model(batch_tokens)

        probs = F.softmax(logits, dim=0)  # 转换为预测概率
        predictions = torch.argmax(probs, axis=1).cpu().detach().numpy()  # 将tensor转换为numpy的array格式
        max_prob = torch.max(probs).cpu().detach().numpy()  # 取出预测概率值
        y_probs.append(float(max_prob))
        y_preds.extend(list(predictions))

    # 组合预测结果，文本+预测类别文本+预测概率
    final_predictions = []
    for text, y_pred, prob in zip(texts, predictions, y_probs):
        label_text = label_map[idx2label[y_pred]]
        final_predictions.append([text, label_text, round(prob, 4)])

    return final_predictions


# ### 模型训练

# 首先来定义一些模型训练会使用到的参数

# In[38]:


class TrainConfig:
    def __init__(self, model_name, dataset_name, loss_fn=F.cross_entropy, optimizer=torch.optim.Adam):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # 获取当前机器是否支持cuda

        self.log_interval = 25  # 每多少step打印一次训练集评估结果
        self.eval_interval = 100  # 每多少step评估一次模型
        self.early_stopping_steps = 600  # 提前停止训练的step数
        self.num_epoch = 10  # 总迭代数

        self.loss_fn = loss_fn  # 损失函数，默认为交叉熵损失
        self.optimizer = optimizer  # 优化器，默认为Adam
        self.lr = 0.001  # 学习率

        self.model_name = model_name  # 模型名称
        self.dataset_name = dataset_name  # 数据集名称
        self.label_weights = None  # 标签权重


# 定义模型的训练过程，一般有如下环节：
# 1. 两个循环，第一个循环为range(num_epoch)，第二个循环为数据集完整遍历
# 2. 每个batch计算模型的前传结果和损失，并更新梯度
# 3. 记录模型在训练集和验证集上的表现
# 4. 保存最佳表现的模型
# 5. 提前终止模型训练，避免模型出现可能过拟合

# In[39]:


def train(model, train_dataset, eval_dataset, train_config, model_config):
    device = train_config.device
    if device == "cuda":
        model.to(device)  # 将模型放入cuda设备中，来提升训练速度
    model.train()  # 作用是启用batch normalization和drop out

    # 默认为Adam优化器
    optimizer = train_config.optimizer(model.parameters(), lr=train_config.lr)

    steps = 0
    best_acc = 0
    best_step = 0
    train_stop = False

    # 模型保存路径
    model_dir = Path(f"model/{train_config.dataset_name}")
    model_dir.mkdir(exist_ok=True, parents=True)
    save_path = model_dir / f"{train_config.model_name}_{model_config.embedding_type}.pt"

    for i in range(train_config.num_epoch):
        for batch_data in train_dataset:
            batch_lengths = None
            if isinstance(train_dataset, BucketIterator) or isinstance(train_dataset, Iterator):
                batch_tokens = batch_data.text.permute(1, 0)
                batch_labels = batch_data.label
            else:
                if train_dataset.return_length:
                    (batch_tokens, batch_labels, batch_lengths) = batch_data
                else:
                    (batch_tokens, batch_labels) = batch_data
                batch_tokens = torch.from_numpy(batch_tokens)
                batch_labels = torch.from_numpy(batch_labels)

            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()  # 重置模型参数的梯度
            # 获取模型前传结果
            if batch_lengths is not None:
                logits = model(batch_tokens, batch_lengths)
            else:
                logits = model(batch_tokens)

            loss = train_config.loss_fn(logits, batch_labels, train_config.label_weights)  # 损失函数默认为交叉熵
            loss.backward()  # 反向传播计算出损失
            optimizer.step()  # 通过反向传播中收集的梯度来调整参数

            steps += 1

            if steps % train_config.log_interval == 0:
                preds = torch.argmax(F.softmax(logits, dim=0), axis=1)  # 得到训练集当前batch的预测结果
                corrects = (preds.view(batch_labels.size()).data == batch_labels.data).sum()  # 计算与正确标签相同的个数
                train_acc = corrects / train_dataset.batch_size * 100
                print(f"Epoch {i} Step {steps} - loss: {loss.item():.4f} train_acc: {train_acc:.2f}")

            if steps % train_config.eval_interval == 0:
                eval_acc, _, _ = evaluation(eval_dataset, model, train_config)

                # 当前验证集准确率大于上一次的最佳结果时，保存模型
                if eval_acc > best_acc:
                    best_acc = eval_acc
                    best_step = steps
                    torch.save(model.state_dict(), save_path)

                # 当间隔n个step，模型表现已经无提升，则终止训练
                if steps - best_step > train_config.early_stopping_steps:
                    print(f"Early stop at {steps} step, best_acc:{best_acc:.2f}")
                    train_stop = True
                    break

                if "lstm" in train_config.model_name:
                    model.train()

        if train_stop:
            break

    # 重新加载最佳模型的权重
    model.load_state_dict(torch.load(save_path))
    # 再次评估模型在验证集上的表现
    eval_acc, y_trues, y_preds = evaluation(eval_dataset, model, train_config)
    show_classification_report(y_trues, y_preds)
    return eval_acc


# 现在，正式开启模型的训练。首先我们来初始化训练集和验证集

# ## TextCNN

# 初始化数据迭代器

# In[40]:


train_data = list(zip(x_train_id, y_train))
train_dataset = TextDataGenerator(train_data, batch_size=32, max_len=max_seq_len)

eval_data = list(zip(x_dev_id, y_dev))
eval_dataset = TextDataGenerator(eval_data, batch_size=128, max_len=max_seq_len)


# 初始化模型实例

# In[41]:


model_conf = CNNModelConfig(label_map, vocab2idx, "static", word_embeddings)
cnn_model = TextCNN(model_conf)
print(cnn_model)


# 模型训练

# In[42]:


train_conf = TrainConfig("text_cnn", "tnews")

best_acc = train(cnn_model, train_dataset, eval_dataset, train_conf, model_conf)


# 在第二个epoch模型基本收敛，整体准确率为52.7%

# 我们可以可以看到 stock, story类别表现比较差，这和它们的训练样本数少可能有关系

# 由于我们有四种不同的embedding配置选项，所以这里我们一次性训练多个模型，来观察每个配置模型的表现

# In[43]:


cnn_results = defaultdict(list)
for embed_type in ["static", "non_static", "multichannel", "scratch"]:
    embeddings = None if embed_type == "scratch" else word_embeddings
    model_conf = CNNModelConfig(label_map, vocab2idx, embed_type, embeddings)
    m = TextCNN(model_conf)
    # 为保证结果的稳定，这里每组配置进行5次训练，取平均结果作为该配置的结果
    if embed_type == "scratch":
        # 随机初始的向量拟合过程可能需要更多的steps
        train_conf.early_stopping_steps = 1000
    for i in range(5):
        best_acc = train(m, train_dataset, eval_dataset, train_conf, model_conf)
        cnn_results[embed_type].append(best_acc)
for embed_type, result in cnn_results.items():
    print(f"{embed_type}: {np.mean(result):.4f}")


# 随机向量是最差的，准确率不到38%，证明了预训练的词向量的有效性。
# 
# 但静态词向量的表现是最好的，达到了53%的准确率。
# 
# 这与论文的结论有一些出路，论文中实验表明经过随模型微调的词向量效果是更好的。

# **混淆矩阵**

# In[ ]:


eval_acc, y_trues, y_preds = evaluation(eval_dataset, cnn_model, train_conf)
plot_confusion_matrix(y_trues, y_preds, sorted_labels)


# 我们可以看到，颜色越浅表示被分配到这个类别的比例越大
# 
# * 娱乐类（entertainment）新闻标题有15%左右错分到了故事类（story）
# * 金融类（finance）有30%错分到了科技类（tech），有25%错分到了股票类（stock）
# * 科技类（tech）有20%错分到了金融类（finance）
# * 军事类（military）有30%以上被错分到了世界综合类（world）
# * 世界综合类（world）有20%以上被错分到了军事类（military）
# * 股票类（stock）有30%左右被错分到了金融类（finance）
# 
# 综上，有三组标题类别模型容易出现混淆，分别为娱乐类和故事类，金融、科技和股票类，军事和世界综合类，这也符合客观认知

# ## BiLSTM

# 这里我们使用torchtext构建的数据迭代器进行训练

# 初始化数据迭代器

# In[ ]:


train_iter, val_iter, test_iter, word_embeddings_torchtext = data_iter(TEXT, LABEL)


# 初始化模型实例

# In[ ]:


model_conf = RNNModelConfig(label_map, vocab2idx, "static", word_embeddings_torchtext)
bilstm_model = BiLSTM(model_conf)
print(bilstm_model)


# 开始训练

# In[ ]:


train_conf = TrainConfig("bi_lstm", "tnews")
train_conf.early_stopping_steps = 3000

best_acc = train(bilstm_model, train_iter, val_iter, train_conf, model_conf)


# 双向LSTM的结果与TextCNN基本一致，准确率在52%~53%

# 我们对embeding类型、LSTM层数和是否双向等超参进行组合，找到最佳的配置，所以这里我们一次性训练多个模型，来观察每个配置模型的表现

# In[ ]:


rnn_results = defaultdict(list)
train_conf.log_interval = 100
train_conf.eval_interval = 400
for embedding_type in ["static", "non_static"]:
    for num_lstm_layer in [1, 2]:
        for num_direction in [1, 2]:
            model_conf = RNNModelConfig(label_map, vocab2idx, embedding_type,
                                        word_embeddings_torchtext, num_lstm_layers=num_lstm_layer,
                                        num_directions=num_direction)
            for i in range(5):
                m = BiLSTM(model_conf)
                best_acc = train(m, train_iter, val_iter, train_conf, model_conf)
                hp_combination = f"{embedding_type}-{num_lstm_layer}-{num_direction}"
                rnn_results[hp_combination].append(best_acc)

for hp_combination, result in rnn_results.items():
    print(f"{hp_combination}: {np.mean(result):.4f}")


# Embedding类型为静态+两层单向LSTM网络效果最佳，准确率为52.59%

# ## BiLSTM-Att

# In[ ]:


model_conf = RNNModelConfig(label_map, vocab2idx, "non_static", word_embeddings_torchtext)
bilstm_att_model = BiLSTMAtt(model_conf)
print(bilstm_att_model)


# In[ ]:


train_conf = TrainConfig("bi_lstm_att", "tnews")
train_conf.early_stopping_steps = 3000

best_acc = train(bilstm_att_model, train_iter, val_iter, train_conf, model_conf)


# 我们可以看到准确率有一点提升，来到了53.21%

# ## 模型预测

# In[ ]:


print(bilstm_model)
predictions = predict(test_iter, bilstm_model, x_test, train_conf.device)


# In[ ]:


predictions[0:10]


# 我们可以看到在测试集上的预测结果还是不尽如人意的

# # （可选）改进方案

# 不平衡问题（长尾问题）是文本分类任务一个难啃的骨头

# <p align=center><img src="./images/long_tail.png" alt="long_tail" style="width:600px;"/></p>

# 长尾分布：少数类别的样本数目非常多，多数类别的样本数目非常少

# 解决不平衡问题的通常思路有两种：重采样（re-sampling）和重加权（re-weighting）

# ## 重采样——数据增强

# ## 重加权——Loss改进

# 重加权就是改变分类loss。相较于重采样，重加权loss更加灵活和方便。其常用方法有：
# 
# * loss类别加权
# 
# * Focal Loss

# ### 带权重的交叉熵

# 通常根据类别数量进行加权，加权系数与类别数量成反比。

# <p align=center><img src="./images/ce_loss_with_weights.svg" alt="ce_loss_with_weights" style="width:400px;"/></p>

# In[ ]:


label_map_reverse = {y: x for x, y in label_map.items()}
label_id_cnt = {label2idx[label_map_reverse[label_name]]: cnt for label_name, cnt in label_cnt.items()}
label_id_cnt = sorted(label_id_cnt.items(), key=lambda x: x[0])

label_weights = np.array([1 / cnt for label_id, cnt in label_id_cnt])
print(label_weights)


# In[ ]:


train_conf = TrainConfig("text_cnn", "tnews")
train_conf.label_weights = torch.from_numpy(label_weights).float().to(train_conf.device)


# In[ ]:


model_conf = CNNModelConfig(label_map, vocab2idx, "static", word_embeddings)
cnn_model = TextCNN(model_conf)
print(cnn_model)


# In[ ]:


best_acc = train(cnn_model, train_dataset, eval_dataset, train_conf, model_conf)


# ### Focal Loss

# 上述loss类别加权主要关注正负样本数量的不平衡，并没有关注难易不平衡。
# 
# Focal Loss主要关注难易样本的不平衡问题，可根据对高置信度(p)样本进行降权。

# <p align=center><img src="./images/focal_loss.svg" alt="focal_loss" style="width:400px;"/></p>

# In[ ]:


def focal_loss(logits, target, alpha=None, gamma=2):
    # 计算加权的交叉熵项: -alpha * log(pt)
    # 其中alpha已经包含在了nll_loss中了
    log_p = F.log_softmax(logits, dim=-1)
    ce = F.nll_loss(log_p, target, weight=alpha, reduction='none')

    # 取出对应真实标签的那个对数似然值
    all_rows = torch.arange(len(logits))
    log_pt = log_p[all_rows, target]

    # 计算focal项: (1 - pt)^gamma
    pt = log_pt.exp()
    focal_term = (1 - pt) ** gamma

    # 完整的损失: -alpha * ((1 - pt)^gamma) * log(pt)
    loss = focal_term * ce
    return loss.mean()


# In[ ]:


def new_loss(logits, target, alpha=None, K=1):
    log_p = F.log_softmax(logits, dim=-1)
    ce = F.nll_loss(log_p, target, weight=alpha, reduction='none')

    logits_new = -K * logits
    soft_p = F.softmax(logits_new, dim=-1)

    all_rows = torch.arange(len(logits))
    new_term = soft_p[all_rows, target]

    loss = new_term * ce
    return loss.mean()


# In[ ]:


train_conf = TrainConfig("text_cnn", "tnews")
label_weights = np.array([1 / 15] * 15)
train_conf.label_weights = torch.from_numpy(label_weights).float().to(train_conf.device)
train_conf.loss_fn = focal_loss


# In[ ]:


model_conf = CNNModelConfig(label_map, vocab2idx, "static", word_embeddings)
cnn_model = TextCNN(model_conf)
print(cnn_model)


# In[ ]:


best_acc = train(cnn_model, train_dataset, eval_dataset, train_conf, model_conf)


# 准确率有略微的提升，提升至53%

# ## 组合模型

# 为容易混淆的类别单独进行建模，组合多个模型的预测结果

# # 总结

# 实战经验总结：
# 
# 1. TextCNN是很适合中短文本场景的强baseline，但不太适合长文本
# 2. 文本分类是一个相容易的应用场景，数据决定上限，与其构建复杂的模型，不如好好分析下错误案例，做一些数据增强。
# 3. 简单的常景推荐FastText和CNN，稍微复杂一点使用BiLSTM+Attention的结构，或者轻量的BERT模型
# 
# 本次实战学习到了如下内容：
# 
# 1. NLP文本分类任务的训练评估流程
# 2. 经典的TextCNN和BiLSTM网络如何进行构建
# 3. 模型效果可视化展示和误差分析

# ## 扩展

# IFLYTEK 长文本分类任务
# 
# 数据集位于`data/iflytek`目录下，数据集格式与tnews一致
# 
# 该数据集共有1.7万多条关于app应用描述的长文本标注数据，包含和日常生活相关的各类应用主题，共119个类别

# 可能遇到的问题如下：
# 
# 1. 长文本如何进行表征？和短文本的差异在哪里？
# 2. 多类别分类任务可能存在更严重的样本不均衡问题
