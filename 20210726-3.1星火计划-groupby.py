#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import pandas as pd
pd.set_option("display.max_rows", 20)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)
import time


# ## pandas.groupby学习

# ### groupby示意
# groupby就是按某个字段分组, 它也确实是用来实现这样功能的.   
# 比如, 将一份数据集按A列进行分组：  

# ![jupyter](./2862169-51af7d4ae64c2f78.webp)

# ### 读取数据

# In[2]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('./input/train_set.csv',nrows=1000)
print(df.head())


# In[3]:


print(set(df['job']))


# In[ ]:





# ### groupby对象
# DataFrame使用groupby()函数返回的结果是DataFrameGroupBy   
# 不是一个DataFrame或者Series  

# In[4]:


groupbyage = df.groupby('age')
print(type(groupbyage))
print(groupbyage)


# groupby分组不仅可以指定一个列名，也可以指定多个列名

# In[5]:


groupbyagemarital = df.groupby(['age','marital'])
print(groupbyagemarital.size())
print(groupbyagemarital)


# ### 可以通过调用get_group()获得按照分组得到的DataFrame对象

# In[ ]:





# In[6]:


groupbyage


# ### groupby常用的一些功能

# In[7]:


df.groupby('job')['age'].sum()


# In[8]:


df.groupby('job')['age'].mean()


# In[9]:


df.groupby('job')['age'].count()


# In[ ]:





# In[ ]:




