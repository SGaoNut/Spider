#!/usr/bin/env python
# coding: utf-8

# 原生形式lightgbm(import lightgbm as lgb)

# In[2]:


# 从sklearn 调入所需要的包
from sklearn import datasets
from sklearn.model_selection import train_test_split #数据分隔出训练集和验证集
import lightgbm as lgb
import numpy as np
import pandas as pd
#导入精度和召回
from sklearn.metrics import precision_score, recall_score
#导入鸢尾花数据
iris = datasets.load_iris()
data = iris.data
label = iris.target
print(pd.DataFrame(data).head())
print(pd.DataFrame(label).head())
data1 = pd.DataFrame(data)
## 花萼长宽花瓣长宽
data1.columns = ['sepal_l','sepal_w','petal_l','petal_w']
print(data1.head())
label1 =pd.DataFrame(label)
label1.columns=['label']
print(label1.head())
#注意这里data label顺序是一致的，千万别打乱


# In[3]:


# 划分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(data1, label1, test_size=0.3, random_state=42)
print("训练集长度:", len(train_x))
print("测试集长度:", len(test_x))


# In[4]:


# 转换为DMatrix数据格式
train_data = lgb.Dataset(train_x,train_y)
test_data = lgb.Dataset(test_x,test_y)
# 设置参数
lgb_params = {
   'boosting_type': 'gbdt',  
    'objective': 'multiclass',
    'metric': 'multi_error', 
    'verbose': 1 , # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
     'num_class':3 #lightgbm.basic.LightGBMError: b'Number of classes should be specified and greater than 1 for multiclass training'
    }

# 模型训练
clf = lgb.train(lgb_params,train_data,num_boost_round =10,
                valid_sets = [train_data,test_data], 
                verbose_eval = 1)
# 模型预测
test_pre = clf.predict(test_x, num_iteration=clf.best_iteration)
# print(test_pre)
print(test_pre[:5])

# 选择表示最高概率的列
test_pre_1 = np.asarray([np.argmax(row) for row in test_pre])
print("test的预测结果:",test_pre_1)

# 模型评估
print('验证集精准率：',precision_score(test_y, test_pre_1, average='macro')) 
print('验证集召回率：',recall_score(test_y, test_pre_1, average='macro'))  


# Sklearn接口形式使用lightgbm

# In[5]:


import lightgbm as lgb
lgb_params = {
    'learning_rate':0.1,
    'max_bin':150,
    'num_leaves':32,    
    'max_depth':11,  
    'objective':'multiclass',
    'n_estimators':300
}
model=lgb.LGBMClassifier(**lgb_params)

model.fit(train_x,train_y)
# 预测
#输出预测结果
test_pre2 = model.predict(test_x)
print(test_pre2)
# 模型评估
print('验证集精准率：',precision_score(test_y, test_pre2, average='macro')) 
print('验证集召回率：',recall_score(test_y, test_pre2, average='macro'))  


# 总结：
# 1. lgb.train中正则化参数为"lambda_l1", "lambda_l1"，sklearn中则为'reg_alpha', 'reg_lambda'。
# 2. 多分类时lgb.train除了'objective':'multiclass',还要指定"num_class":5，而sklearn接口只需要指定'objective':'multiclass'。
# 3. 迭代次数在sklearn中是'n_estimators':20，在初始化模型时指定
# 

# In[ ]:





# In[ ]:





# In[ ]:




