#!/usr/bin/env python
# coding: utf-8

# # 一、集成学习
# 
# ## 1. 原理
# <img src="./img/集成学习思想.png" width = "400" height = "300" alt="图片名称" align=left />
# 

# ## 2. 常用的集成方法
#     装袋法（Bagging）、提升法（ Boosting ）和堆叠法（Stacking）。

# 
# ## 3. 相关集成模型介绍
# 
# ### 3.1 Bagging
# 并行的建立一些学习器，尽可能使其相互独立。从方差-偏差的角度看，可以有效减小方差。
# #### 3.1.1 原理
#     核心方法：自助抽样(Bootstrap)和聚合(Aggregating)
#     分类问题：采用多数投票法进行预测；回归问题：采用平均法进行预测
#     袋外样本：每个基模型在训练时理论上只使用63.2%的训练样本，没使用的样本称为袋外样本。
# #### 3.1.2 RandomForest
#     由多个CART组成。
#     “随机”是核心。两个“随机”：
#         对样本进行有放回抽样（boostrap）
#         对特征进行随机抽样
#     “森林”指建立多棵决策树进行组合。
#     
# ### 3.2 Boosting
# #### 3.2.1 原理
# 
# 串行的建立一些学习器，通过一定策略提升弱学习器效果，组合得到强学习器，可以有效减小偏差
# 
# #### 3.2.2 GBDT
# 梯度提升树，Gradient Boosting Decision Tree  
# 以回归树为基学习器的boosting方法 
# 多个弱学习器合成强学习器的过程(加权求和)，每次迭代产生一个弱学习器，当前弱学习器是在之前分类器残差基础上训练。
# #### 3.2.3 Xgboost
# ##### XGB与GBDT区别
# XGBoost在代价函数中加入正则化项，控制模型复杂度，降低模型方差，模型更加简单，防止过拟合【目标函数的定义不同】；
# 
# GBDT用到一阶导数信息，XGBoost对代价函数进行了二阶泰勒展开，同时用到一阶与二阶导数，支持自定义代价函数(二阶可导)；
# 
# 其他特性：
# 
# a.行采样；
# 
# b.列采样；
# 
# c.Shrinkage：每次迭代中对树的每个叶子结点的分数乘上一个缩减权重$\eta$，降低了每棵独立树的影响，便于留更大的空间给后面生成的树去优化模型，类似于学习速率；
# 
# d.支持自定义损失函数(需二阶可导)。
# 
#     
# #### 3.2.4 LightGBM
# ##### LightGBM与XGB区别
# 
# 1. 切分算法（切分点的选取）
# 2. 占用的内存更低，只保存特征离散化后的值，而这个值一般用8位整型存储就足够了，内存消耗可以降低为原来的1/8
# 3. LightGBM直接支持类别特征
# 4. 决策树生长策略不同【XGBoost采用带深度限制的level-wise生长策略，LightGBM采用leaf-wise生长策略】
# 
# ### 3.3 Stacking
# 
# 建立多个不同基模型，将每个模型的预测结果当做输入，建立一个高层的综合模型，可以有效改进预测。
# 额外的知识：交叉验证(Cross-validation)
# <img src="./img/stacking.png" width = "600" height = "320" alt="图片名称" align=left />
# 

# ### 3.4 RF与GBDT区别
#     1、组成RF的树可以是分类树，也可以是回归树；而GBDT只由回归树组成，因为GBDT对所有树的结果累加，累加无法通过分类完成
#     2、组成RF的树并行生成；GBDT串行生成 ，GBDT更容易过拟合
#     3、输出结果，RF采用多数投票等；GBDT将所有结果累加，或加权累加
#     4、RF对异常值不敏感，GBDT对异常值敏感
#     5、RF对训练集一视同仁，每棵树分裂特征随机；GBDT基于权值的弱分类器的集成 ，前面的树优先分裂对大部分样本区分的特征，后分裂对小部分样本区分的特征
#     6、RF通过减少模型方差提高性能，GBDT通过减少模型偏差提高性能
#     7、RF参数主要是树的棵树，GBDT主要是树的深度，一般为1

# ## 二、 分类实战
# ### 学生成绩等级预测

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split


# In[3]:


df = pd.read_csv('data/StudentPerformance.csv') # 读取 csv 数据
df.head(10) # 查看前十行数据


# ### 2.1 特征工程

# In[4]:



X = df.drop('Class', axis=1)
y = df['Class']
X = pd.get_dummies(X) # 将所有的分类型特征转换为数字, 虚拟变量: dummy variables
# sel = SelectKBest(chi2,k=40)
# X = sel.fit_transform(X, y)


# ### 2.2 数据拆分

# In[5]:


X_train,X_test, y_train, y_test = train_test_split(
                    X, y, test_size=.2, random_state=10, stratify=y)


# ### 2.3 RandomForest

# In[6]:



# 训练并且测试模型
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

Logit = RandomForestClassifier(
    n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
Logit.fit(X_train, y_train)

Predict = Logit.predict(X_test)
print('Predict', Predict)

Score = accuracy_score(y_test, Predict)
print('accuracy: ', Score)


# In[ ]:





# ## 三、回归实战

# In[7]:


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


# In[8]:


cars = pd.read_csv('./data/auto-mpg.data',names=["燃油效率","气缸","排量","马力","重量","加速度","型号年份","编号","原产地"],delim_whitespace = True)
cars.head()


# In[9]:



error = cars[cars.马力 == '?']

#删除horsepower值为'?'的行
cars = cars[cars.马力 != '?']
cars['马力'] = cars['马力'].astype(float)
cars.dtypes # 检查数据类型


# ### 3.1 特征工程

# In[10]:



X = cars.drop('燃油效率', axis=1)
y_new = cars[['燃油效率']]
X_new = pd.get_dummies(X) # 将所有的分类型特征转换为数字, 虚拟变量: dummy variables


# ### 3.2 数据拆分

# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, 
                                                   test_size=.2, 
                                                   random_state=10)


# ### 3.3 RandomForest

# In[12]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

clf = RandomForestRegressor(random_state=0)
clf = clf.fit(X_train, y_train)

y_test['燃油效率预测值-rf'] = clf.predict(X_test)

mean_squared_error(y_test['燃油效率'], y_test['燃油效率预测值-rf'])


# In[13]:



error['马力'] = -1 #np.mean(cars['马力'])

errorDf = pd.get_dummies(error) # 将所有的分类型特征转换为数字, 虚拟变量: dummy variables
errorDf

for col in X_train.columns:
    if col not in errorDf.columns:
        errorDf[col] = 0
        
        
## 加入缺失特征的数据
X = pd.concat([X_train, errorDf[X_train.columns]])
Y = pd.concat([y_train, errorDf[y_train.columns]])

clf = RandomForestRegressor( random_state=0 )
clf = clf.fit(X, Y)
y_test['燃油效率预测值-rf-缺失值填充'] = clf.predict(X_test)

mean_squared_error(y_test['燃油效率'], y_test['燃油效率预测值-rf-缺失值填充'])


# ### 3.4 xgboost

# In[17]:


import xgboost as xgb


# #### 3.4.1 baseline

# In[18]:



params = {
    'objective': 'reg:linear',
    'colsample_bytree': 0.72,
    'max_depth': 8,
    'seed': 202003
}

## build xgb
xgtrain = xgb.DMatrix( X_train, y_train )
gbdt = xgb.train( params, xgtrain, 50)

importance = gbdt.get_score()
importance = sorted( importance.items(), key=lambda x:x[1], reverse=True )
importance = pd.DataFrame(importance, columns=['feature', 'score'])

y_test['燃油效率预测值-xgb'] = gbdt.predict( xgb.DMatrix( X_test ) )
mean_squared_error(y_test['燃油效率'], y_test['燃油效率预测值-xgb'])


# #### 3.4.2 填补缺失值
# 先对确实数据的样子做缺失值填补，再构造xgboost模型

# In[19]:



params = {
    'objective': 'reg:linear',
    'colsample_bytree': 0.72,
    'max_depth': 8,
    'seed': 202003,
    'missing':-1
}

X = pd.concat([X_train, errorDf[X_train.columns]])
Y = pd.concat([y_train, errorDf[y_train.columns]])

## build xgb
xgtrain = xgb.DMatrix( X, Y )
gbdt = xgb.train( params, xgtrain, 20)

importance = gbdt.get_score()
importance = sorted( importance.items(), key=lambda x:x[1], reverse=True )
importance = pd.DataFrame(importance, columns=['feature', 'score'])

y_test['燃油效率预测值-xgb-缺失值填充'] = gbdt.predict( xgb.DMatrix( X_test ) )
mean_squared_error(y_test['燃油效率'], y_test['燃油效率预测值-xgb-缺失值填充'])


# In[20]:


mape(y_test['燃油效率'], y_test['燃油效率预测值-xgb-缺失值填充'])


# #### 3.4.3 针对评测指标的优化
# 针对mape评测指标，做进一步模型优化

# In[21]:



def evalmape(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = np.abs((y_pred - y_true) / y_true)
    return 'mape',np.mean(err) * 100

params = {
    'objective': 'reg:linear',
    'colsample_bytree': 0.72,
    'max_depth': 8,
    'nthread': 8,
    'seed': 202003,
    'missing':-1
}

X = pd.concat([X_train, errorDf[X_train.columns]])
Y = pd.concat([y_train, errorDf[y_train.columns]])

## build xgb
xgtrain = xgb.DMatrix( X, Y )
xgval = xgb.DMatrix( X, Y )
watchlist = [(xgtrain,'train'), (xgval, 'val')]

gbdt = xgb.train( params, xgtrain, 20, evals = watchlist, feval=evalmape)

importance = gbdt.get_score()
importance = sorted( importance.items(), key=lambda x:x[1], reverse=True )
importance = pd.DataFrame(importance, columns=['feature', 'score'])

y_test['燃油效率预测值-xgb-mape'] = gbdt.predict( xgb.DMatrix( X_test ) )
mean_squared_error(y_test['燃油效率'], y_test['燃油效率预测值-xgb-mape'])


# In[22]:


mape(y_test['燃油效率'], y_test['燃油效率预测值-xgb-mape'])

