#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


#     pandas :1.主要包含Series和DataFrame两个数据结构；2.经常和numpy配合使用
#     Series 带有标签的同构类型数组
#     DataFrame：一个DataFrame中可以包含若干个Series。

# In[3]:


# 获取版本信息
print(pd.__version__)


# In[4]:


path='./'
train = pd.read_csv(path+'input/train_set.csv')
test = pd.read_csv(path+'input/test_set.csv')
print(train.info())
print(test.info())


# In[5]:


print(train.get_dtype_counts())


# In[7]:


print(train['age'].value_counts())


# In[8]:


train.head()


# In[9]:


train.tail()


# 重命名dataframe的特定列

# In[12]:


train=train.rename(columns={'age':'age1'})


# In[13]:


train.head()


# In[17]:


train.columns.values[0] = 'ID1'


# In[25]:


train.columns.values[1] = 'age'


# In[26]:


train.head()


# In[27]:


# 若有缺失值，则为Ture
train.isnull().values.any()


# In[28]:


train.isnull().sum()


# In[29]:


n_missings_each_col = train.apply(lambda x: x.isnull().sum())
n_missings_each_col


# In[ ]:


import pandas as pd
df = pd.DataFrame({'name' : "hello the cruel world".split(),
                   'growth' : [100, 125, 150, 200]},
                   index = "jack tom mike nike".split())
print (df)


# In[4]:


#columns 
print(train.columns)


# In[5]:


l1=list(train.columns)


# In[11]:


print(train.shape)


# In[6]:


print(train.size)


# In[7]:


train.values


# In[12]:


train.dtypes


# In[9]:


train.ndim


# In[10]:


train['ID'].head()


# In[14]:


train['poutcome'].head(10)


# In[ ]:




