#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


#     pandas :1.主要包含Series和DataFrame两个数据结构；2.经常和numpy配合使用
#     Series 带有标签的同构类型数组
#     DataFrame：一个DataFrame中可以包含若干个Series。

# In[2]:


# 获取版本信息
print(pd.__version__)


# In[3]:


path='./'
train = pd.read_csv(path+'input/train_set.csv')
test = pd.read_csv(path+'input/test_set.csv')
print(train.info())
print(test.info())


# In[5]:


print(train.dtypes.value_counts())


# In[6]:


print(train['age'].value_counts())


# In[7]:


train.head()


# In[8]:


train.tail()


# 重命名dataframe的特定列

# In[9]:


train=train.rename(columns={'age':'age1'})


# In[10]:


train.head()


# In[11]:


train.columns.values[0] = 'ID1'


# In[12]:


train.columns.values[1] = 'age'


# In[13]:


train.head()


# In[14]:


# 若有缺失值，则为Ture
train.isnull().values.any()


# In[15]:


train.isnull().sum()


# In[16]:


n_missings_each_col = train.apply(lambda x: x.isnull().sum())
n_missings_each_col


# In[17]:


import pandas as pd
df = pd.DataFrame({'name' : "hello the cruel world".split(),
                   'growth' : [100, 125, 150, 200]},
                   index = "jack tom mike nike".split())
print (df)


# In[18]:


#columns 
print(train.columns)


# In[19]:


l1=list(train.columns)


# In[20]:


print(train.shape)


# In[21]:


print(train.size)


# In[22]:


train.values


# In[23]:


train.dtypes


# In[24]:


train.ndim


# In[25]:


train['ID'].head()


# In[26]:


train['poutcome'].head(10)


# In[ ]:





# In[ ]:




