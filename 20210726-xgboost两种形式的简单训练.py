#!/usr/bin/env python
# coding: utf-8

# 原生态xgboost的使用形式

# In[1]:


# 从sklearn 调入所需要的包
from sklearn import datasets
from sklearn.model_selection import train_test_split #数据分隔出训练集和验证集
import xgboost as xgb
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


# In[ ]:





# In[2]:


data1.shape


# In[ ]:





# In[3]:


label1.label.value_counts()


# In[4]:


# 划分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(data1.values, label1.values, test_size=0.3, random_state=42)
print("训练集长度:", len(train_x))
print("测试集长度:", len(test_x))


# In[5]:


# 转换为DMatrix数据格式
# dtrain = xgb.DMatrix(train_x, label=train_y)
test_data = xgb.DMatrix(test_x, label=test_y)
# 设置参数
###multi：softmax是使用softmax后产生的分类结果，而multi:softprob是输出的概率矩阵。

xgb_params = {
    'eta': 0.3, #学习率
    'silent': True,  # 输出运行讯息
    'objective': 'multi:softprob',  # 使用多分类生成概率矩阵格式'multi:softmax',multi:softprob
    'num_class': 3,  # 共有几个类别
    'max_depth': 3  # 深度
}
num_round = 20  # 跑的步数

# 模型训练
model = xgb.train(xgb_params,xgb.DMatrix(train_x, label=train_y), num_round)
# 模型预测
test_pre = model.predict(test_data)

print(test_pre[:5])

# 选择表示最高概率的列
test_pre_1 = np.asarray([np.argmax(row) for row in test_pre])
print("test的预测结果:",test_pre_1)

# 模型评估
print('验证集精准率：',precision_score(test_y, test_pre_1, average='macro')) 
print('验证集召回率：',recall_score(test_y, test_pre_1, average='macro'))  


# Sklearn接口形式使用Xgboost

# In[6]:


from xgboost import XGBClassifier
model = XGBClassifier(
        learning_rate=0.01,#学习率
        n_estimators=3000,#步长
        max_depth=4,#深度
        objective='binary:logistic',
        seed=27
    )
model.fit(train_x,train_y)

# 预测
#输出预测结果
test_pre2 = model.predict(test_x)
print(test_pre2)
# 模型评估
print('验证集精准率：',precision_score(test_y, test_pre2, average='macro')) 
print('验证集召回率：',recall_score(test_y, test_pre2, average='macro'))  

