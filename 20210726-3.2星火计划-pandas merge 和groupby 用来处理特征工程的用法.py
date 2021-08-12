#!/usr/bin/env python
# coding: utf-8

# ## pandas.merge的用法
# 
#     pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,
#          left_index=False, right_index=False, sort=True,
#          suffixes=('_x', '_y'), copy=True, indicator=False,
#          validate=None)
# 

# left 参与合并的左侧DataFrame   
# right 参与合并的右侧DataFrame    
# how 连接方式：‘inner’（默认）；还有，‘outer’、‘left’、‘right’    
# on 用于连接的列名，必须同时存在于左右两个DataFrame对象中，如果位指定，则以left和right列名的交集作为连接键    
# left_on 左侧DataFarme中用作连接键的列   
# right_on 右侧DataFarme中用作连接键的列   
# left_index 将左侧的行索引用作其连接键    
# right_index 将右侧的行索引用作其连接键    
# sort 根据连接键对合并后的数据进行排序，默认为True。有时在处理大数据集时，禁用该选项可获得更好的性能    
# suffixes 字符串值元组，用于追加到重叠列名的末尾，默认为（‘_x’,‘_y’）.例如，左右两个DataFrame对象都有‘data’，则结果中就会出   现‘data_x’，‘data_y’  

# In[1]:


import pandas as pd
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3']})


right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                 'C': ['C0', 'C1', 'C2', 'C3'],
                 'D': ['D0', 'D1', 'D2', 'D3']})


result = pd.merge(left, right, on='key')


# In[2]:


result.head()


# In[3]:


import pandas as pd
left = pd.DataFrame({'key1': ['K0', 'K1', 'K2', 'K3'],
                      'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3']})


right = pd.DataFrame({'key2': ['K0', 'K1', 'K2', 'K3'],
                     'C': ['C0', 'C1', 'C2', 'C3'],
                     'D': ['D0', 'D1', 'D2', 'D3']})


result = pd.merge(left, right, left_on='key1',right_on='key2')


# In[4]:


result


# ### merge删除重复列

# In[5]:


result = pd.merge(left, right, left_on='key1',right_on='key2').drop('key1',axis=1)


# In[6]:


result


# ### 参数how的用法-inner
# 
# 默认：inner内连接，取交集
# 

# In[7]:


import pandas as pd
left = pd.DataFrame({'key1': ['K0', 'K1', 'K2', 'K3','K4'],
                      'A': ['A0', 'A1', 'A2', 'A3','A4'],
                    'B': ['B0', 'B1', 'B2', 'B3','B4']})


right = pd.DataFrame({'key2': ['K0', 'K1', 'K2', 'K3','K5'],
                     'C': ['C0', 'C1', 'C2', 'C3','C4'],
                     'D': ['D0', 'D1', 'D2', 'D3','D4']})


result = pd.merge(left, right, left_on='key1',right_on='key2',how='inner')


# In[8]:


result 


# ### 参数how的用法-outer
# 
# outer 外连接，取并集，并用nan填充
# 

# In[9]:


import pandas as pd
left = pd.DataFrame({'key1': ['K0', 'K1', 'K2', 'K3','K4'],
                      'A': ['A0', 'A1', 'A2', 'A3','A4'],
                    'B': ['B0', 'B1', 'B2', 'B3','B4']})


right = pd.DataFrame({'key2': ['K0', 'K1', 'K2', 'K3','K5'],
                     'C': ['C0', 'C1', 'C2', 'C3','C4'],
                     'D': ['D0', 'D1', 'D2', 'D3','D4']})


result = pd.merge(left, right, left_on='key1',right_on='key2',how='outer')
result 


# ### 参数how的用法-left
# 
# left 左连接， 左侧取全部，右侧取部分

# In[10]:


import pandas as pd
left = pd.DataFrame({'key1': ['K0', 'K1', 'K2', 'K3','K4'],
                      'A': ['A0', 'A1', 'A2', 'A3','A4'],
                    'B': ['B0', 'B1', 'B2', 'B3','B4']})


right = pd.DataFrame({'key2': ['K0', 'K1', 'K2', 'K3','K5'],
                     'C': ['C0', 'C1', 'C2', 'C3','C4'],
                     'D': ['D0', 'D1', 'D2', 'D3','D4']})


result = pd.merge(left, right, left_on='key1',right_on='key2',how='left')
result 


# ### 参数how的用法-right
# 
# right 有连接，左侧取部分，右侧取全部

# In[11]:


import pandas as pd
left = pd.DataFrame({'key1': ['K0', 'K1', 'K2', 'K3','K4'],
                      'A': ['A0', 'A1', 'A2', 'A3','A4'],
                    'B': ['B0', 'B1', 'B2', 'B3','B4']})


right = pd.DataFrame({'key2': ['K0', 'K1', 'K2', 'K3','K5'],
                     'C': ['C0', 'C1', 'C2', 'C3','C4'],
                     'D': ['D0', 'D1', 'D2', 'D3','D4']})


result = pd.merge(left, right, left_on='key1',right_on='key2',how='right')
result 


# ## pandas.groupby的用法

# ### pandaas.groupby trainform的用法
# transform()方法+自定义方法

# In[12]:


import pandas as pd
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)


# In[13]:


data=pd.read_csv('./input/test_set.csv')


# In[14]:


data.head()


# In[15]:


data['job_age_count']=data.groupby(['job'])['age'].transform(lambda x:x.count())
data['job_age_sum']=data.groupby(['job'])['age'].transform(lambda x:x.sum())
data['job_age_max']=data.groupby(['job'])['age'].transform(lambda x:x.max())
data['job_age_min']=data.groupby(['job'])['age'].transform(lambda x:x.min())
data['job_age_mean']=data.groupby(['job'])['age'].transform(lambda x:x.mean())


# In[16]:


data.head()


# ### pandas.groupby trainform的用法
# transform()方法+python内置方法

# In[17]:


data['job_age_count1']=data.groupby(['job'])['age'].transform('count')
data['job_age_sum1']=data.groupby(['job'])['age'].transform(sum)
data['job_age_max1']=data.groupby(['job'])['age'].transform(max)
data['job_age_min1']=data.groupby(['job'])['age'].transform(min)
data['job_age_mean1']=data.groupby(['job'])['age'].transform('mean')


# In[18]:


data.head()


# ### pandas.groupby apply的用法
# apply()方法+自定义方法

# In[19]:


data['job_age_count']=data.groupby(['job'])['age'].apply(lambda x:x.count())
data['job_age_sum']=data.groupby(['job'])['age'].apply(lambda x:x.sum)
data['job_age_max']=data.groupby(['job'])['age'].apply(lambda x:x.max)
data['job_age_min']=data.groupby(['job'])['age'].apply(lambda x:x.min)
data['job_age_mean']=data.groupby(['job'])['age'].apply(lambda x:x.mean())


# In[20]:


data.head()


# ### pandas.groupby agg的用法
# agg()方法+自定义方法

# In[21]:


data.groupby(['job'])['age'].agg(lambda x:x.count())
data.groupby(['job'])['age'].agg(lambda x:x.sum)
data.groupby(['job'])['age'].agg(lambda x:x.max)
data.groupby(['job'])['age'].agg(lambda x:x.min)
data.groupby(['job'])['age'].agg(lambda x:x.mean())


# In[22]:


data.head()


# In[23]:


data.groupby(['job'])['age'].agg('count')
data.groupby(['job'])['age'].agg(sum)
data.groupby(['job'])['age'].agg(max)
data.groupby(['job'])['age'].agg(min)
data.groupby(['job'])['age'].agg('mean')


# In[27]:


aggcount=data.groupby(['job'])['age'].agg({'count',sum,max,min,'mean'})
aggcount.columns=['job_age_count2','job_age_sum2','job_age_max2','job_age_min2','job_age_mean2']


# In[28]:


aggcount


# In[29]:


data=pd.merge(data,aggcount,on='job',how='left')


# In[30]:


data.head()


# ### pandas.groupby 众数特征

# In[31]:


import scipy.stats as stats
data['job_age_mode']=data.groupby(['job'])['age'].transform(lambda x:stats.mode(x)[0][0])


# In[32]:


data.head()


# In[ ]:




