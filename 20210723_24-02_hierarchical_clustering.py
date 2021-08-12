#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import pdist
from IPython.display import HTML


# In[2]:


# if you have difficulties with ffmpeg to run the simulation, see https://stackoverflow.com/questions/13316397/matplotlib-animation-no-moviewriters-available
# plt.rcParams['animation.ffmpeg_path'] = your_windows_path
plt.rcParams['animation.ffmpeg_args'] = '-report'
plt.rcParams['animation.bitrate'] = 2000

get_ipython().run_line_magic('matplotlib', 'notebook')
pd.options.display.float_format = '{:,.2f}'.format


# ### Load Iris Data

# In[3]:


iris = load_iris()
iris.keys()


# In[4]:


print(iris.DESCR)


# ### Create DataFrame

# In[5]:


# 构造Iris数据集
features = iris.feature_names
data = pd.DataFrame(data=np.column_stack([iris.data, iris.target]), 
                    columns=features + ['label'])
data.label = data.label.astype(int)
data.info()


# ### Standardize Data

# In[6]:


scaler = StandardScaler()
features_standardized = scaler.fit_transform(data[features])
n = len(data)


# ### Reduce Dimensionality to visualize clusters

# In[7]:


pca = PCA(n_components=2)
features_2D = pca.fit_transform(features_standardized)


# In[8]:


ev1, ev2 = pca.explained_variance_ratio_
ax = plt.figure().gca(title='2D Projection', 
                      xlabel='Explained Variance: {:.2%}'.format(ev1), 
                      ylabel='Explained Variance: {:.2%}'.format(ev2))
ax.scatter(*features_2D.T, c=data.label, s=10)


# ### 不同距离度量方式的层次聚类

# In[9]:


# 距离度量方法实现：linkage
Z = linkage(features_standardized, 'ward')
Z[:5]


# In[10]:


linkage_matrix = pd.DataFrame(data=Z, 
                              columns=['cluster_1', 'cluster_2', 
                                       'distance', 'n_objects'],
                              index=range(1, n))
for col in ['cluster_1', 'cluster_2', 'n_objects']:
    linkage_matrix[col] = linkage_matrix[col].astype(int)
linkage_matrix.info()


# ### Compare linkage types

# In[11]:


# 对比四种距离度量方式的聚类效果
methods = ['single', 'complete', 'average', 'ward']
pairwise_distance = pdist(features_standardized)


# In[12]:


fig, axes = plt.subplots(figsize=(10, 15), nrows=4, sharex=True)
axes = axes.flatten()
for i, method in enumerate(methods):
    Z = linkage(features_standardized, method)
    c, coph_dists = cophenet(Z, pairwise_distance)
    dendrogram(Z, labels=data.label.values,
        orientation='top', leaf_rotation=0., 
        leaf_font_size=8., ax = axes[i])
    axes[i].set_title('Method: {} | Correlation: {:.2f}'.format(method.capitalize(), c))

