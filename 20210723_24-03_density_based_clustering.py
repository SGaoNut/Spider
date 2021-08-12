#!/usr/bin/env python
# coding: utf-8

# In[25]:


from time import sleep
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from numpy import atleast_2d
from random import shuffle
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.neighbors import KDTree


# In[26]:


get_ipython().run_line_magic('matplotlib', 'notebook')
cmap = cm.get_cmap('viridis')
pd.options.display.float_format = '{:,.2f}'.format


# ### Load Iris Data

# In[27]:


# 数据准备，使用Iris数据集
iris = load_iris()
iris.keys()


# ### Create DataFrame

# In[28]:


# 处理数据集，共需5列，前四列为特征，最后一列为label，为显示方便将特征标准化后降维至2维
features = iris.feature_names
data = pd.DataFrame(data=np.column_stack([iris.data, iris.target]), 
                    columns=features + ['label'])
data.label = data.label.astype(int)
data.info()


# In[29]:


len(set(data['label']))


# In[30]:


data[features]


# ### Standardize Data

# In[31]:


# StandardScaler类是一个用来讲数据进行归一化和标准化的类。-计算训练集的平均值和标准差，以便测试数据集使用相同的变换。
scaler = StandardScaler()
features_standardized = scaler.fit_transform(data[features])
n = len(data)


# ### Reduce Dimensionality to visualize clusters

# In[32]:


pca = PCA(n_components=2)
features_2D = pca.fit_transform(features_standardized)


# In[33]:


ev1, ev2 = pca.explained_variance_ratio_
ax = plt.figure().gca(title='2D Projection', 
                      xlabel='Explained Variance: {:.2%}'.format(ev1), 
                      ylabel='Explained Variance: {:.2%}'.format(ev2))
ax.scatter(*features_2D.T, c=data.label, s=10)
plt.tight_layout();


# ### Perform DBSCAN clustering

# In[34]:


# 初始化一个DBSCAN，未调整参数，MI值表现较差
clusterer = DBSCAN()
data['clusters'] = clusterer.fit_predict(features_standardized)
fig, axes = plt.subplots(ncols=2)
labels, clusters = data.label, data.clusters
mi = adjusted_mutual_info_score(labels, clusters)
axes[0].scatter(*features_2D.T, c=data.label, s=10)
axes[0].set_title('Original Data')
axes[1].scatter(*features_2D.T, c=data.clusters, s=10)
axes[1].set_title('Clusters | MI={:.2f}'.format(mi))
plt.tight_layout()


# ### Compare parameter settings

# In[35]:


# 根据热力图选择合适参数
eps_range = np.arange(.2, .91, .05)
min_samples_range = list(range(3, 10))
labels = data.label
mi = {}
for eps in eps_range:
    for min_samples in min_samples_range:
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = clusterer.fit_predict(features_standardized)  
        mi[(eps, min_samples)] = adjusted_mutual_info_score(clusters, labels)


# In[36]:


results = pd.Series(mi)
results.index = pd.MultiIndex.from_tuples(results.index)
fig, axes = plt.subplots()
ax = sns.heatmap(results.unstack(), annot=True, fmt='.2f')
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
plt.tight_layout()


# In[37]:


results.unstack()


# ### Run again

# In[38]:


clusterer = DBSCAN(eps=.8, min_samples=5)
data['clusters'] = clusterer.fit_predict(features_standardized)
fig, axes = plt.subplots(ncols=2)
labels, clusters = data.label, data.clusters
mi = adjusted_mutual_info_score(labels, clusters)
axes[0].scatter(*features_2D.T, c=data.label, s=10)
axes[0].set_title('Original Data')
axes[1].scatter(*features_2D.T, c=data.clusters, s=10)
axes[1].set_title('Clusters | MI={:.2f}'.format(mi))
plt.tight_layout()


# In[ ]:




