#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime
import numpy as np             #for numerical computations like log,exp,sqrt etc
import pandas as pd            #for reading & storing data, pre-processing
import matplotlib.pylab as plt #for visualization
#for making sure matplotlib plots are generated in Jupyter notebook itself
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf

##分解(decomposing) 可以用来把时序数据中的趋势和周期性数据都分离出来:
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6


# In[2]:


Data = pd.read_csv('./world.csv', index_col=0)#1月20日-3月18日 世界总确诊人数
#增加id递增列
Data = Data.reset_index()
##该步骤很重要！！建立时序数据
#Parse strings to datetime type  将字符串解析为datetime类型
#convert from string to datetime ###Name: date, dtype: datetime64[ns]
Data['日期']= pd.to_datetime(Data['日期'],infer_datetime_format=True) 
#set_index( ) 将 DataFrame 中的列转化为行索引
indexedDataset = Data.set_index(['日期'])
#数据反转
indexedDataset = indexedDataset.iloc[::-1]
indexedDataset


# In[3]:


# plot graph
#设置画布大小
plt.figure(figsize=(15,5),dpi = 80)
#横轴坐标旋转45°
plt.xticks(rotation=45)

plt.xlabel('Date')
plt.ylabel('confirmedCount')
plt.plot(indexedDataset)


# ### 三、时序数据变换，获得稳定性数据  

# #### 3.1构建稳定性检验函数：  
#     #滑动平均值  
#     #滑动标准差  
#     #Augmented Dickey–Fuller ADF稳定性检验

# In[4]:


#构建稳定性检验函数，一般pvalue值小于0.05，越小越好
def test_stationarity(timeseries):  
    
    #Determine rolling statistics
    movingAverage = timeseries.rolling(window=12).mean()  #滑动平均值
    movingSTD = timeseries.rolling(window=12).std()     #滑动标准差
    
    #Plot rolling statistics
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(movingAverage, color='red', label='Rolling Mean')
    std = plt.plot(movingSTD, color='black', label='Rolling Std')
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey–Fuller test:   Augmented Dickey–Fuller ADF稳定性检验
    print('Results of Dickey Fuller Test:')
    dftest = adfuller(timeseries['confirmedCount'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


# #### 3.2  Differencing差分方法-Timeshift transformation

# In[5]:


#Estimating trend 估计趋势
indexedDataset_logScale = np.log(indexedDataset)  #taking log 
plt.plot(indexedDataset_logScale)
plt.xticks(rotation=45)


# In[6]:


##Differencing--差分
##这里采用的是一阶差分：一阶差分就是离散函数中连续相邻两项之差。
datasetLogDiffShifting_1 = indexedDataset_logScale - indexedDataset_logScale.shift()
plt.plot(datasetLogDiffShifting_1)

#二阶差分：二阶差分就是一阶差分再进行一次差分。
#注：以下内容可以看出，二阶差分后数据已经稳定，所以ARIMA模型参数d=2。
datasetLogDiffShifting_2 = datasetLogDiffShifting_1 - datasetLogDiffShifting_1.shift()
plt.plot(datasetLogDiffShifting_2)
plt.xticks(rotation=45)


# In[7]:


example1 = indexedDataset_logScale.diff(1)
plt.plot(example1)
example2 = example1.diff(1)
plt.plot(example2)
example3 = example2.diff(1)
plt.plot(example3)
plt.xticks(rotation=45)


# In[8]:


datasetLogDiffShifting_1.dropna(inplace=True)#滤除缺失数据。
test_stationarity(datasetLogDiffShifting_1)


# In[9]:


datasetLogDiffShifting_2.dropna(inplace=True)#滤除缺失数据。
test_stationarity(datasetLogDiffShifting_2)


# 差分后平稳性都要比原数据好很多。

# ### 四、构建模型
# #### 4.1 自相关图和偏自相关图的分析

# In[10]:


import statsmodels.api as sm
fig = plt.figure(figsize=(12,8))

#acf   from statsmodels.tsa.stattools import acf, pacf
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(datasetLogDiffShifting_1, lags=20,ax=ax1)
ax1.xaxis.set_ticks_position('bottom')
fig.tight_layout();

#pacf
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(datasetLogDiffShifting_1, lags=20, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
fig.tight_layout();
#下图中的阴影表示置信区间，可以看出不同阶数自相关性的变化情况，从而选出p值和q值


# p=3/4  q=3

# #### 建立模型

# In[11]:


# AR+I+MA = ARIMA model
"""
ARIMA模型有三个参数:p,d,q。
p--代表预测模型中采用的时序数据本身的滞后数(lags) ,也叫做AR/Auto-Regressive项
d--代表时序数据需要进行几阶差分化，才是稳定的，也叫Integrated项。
q--代表预测模型中采用的预测误差的滞后数(lags)，也叫做MA/Moving Average项

"""
import warnings
warnings.filterwarnings('ignore')
model_3 = ARIMA(indexedDataset_logScale, order=(3,1,3))
results_ARIMA = model_3.fit(disp=-1)
plt.plot(datasetLogDiffShifting_1)
plt.plot(results_ARIMA.fittedvalues, color='red')#模型数据的差分值
plt.title('RSS: %.4f'%sum((results_ARIMA.fittedvalues - datasetLogDiffShifting_1['confirmedCount'])**2))
plt.xticks(rotation=45)
print('Plotting ARIMA model')


# #### 五、时序数据预测

# In[12]:


#ARIMA拟合的其实是一阶差分ts_log_diff，predictions_ARIMA_diff[i]是第i个天与i-1个天的ts_log的差值。
#由于差分化有一阶滞后，所以第一个天的数据是空的
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
predictions_ARIMA_diff#ARIMA拟合的一阶差分值


# In[13]:


indexedDataset_logScale


# In[14]:


#Convert to cumulative sum 累计和
#累加现有的diff，得到每个值与第一个天的差分（同log底的情况下）。
#即predictions_ARIMA_diff_cumsum[i] 是第i个月与第1个月的ts_log的差值。
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

print(predictions_ARIMA_diff_cumsum)


# In[15]:


#predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff_cumsum.cumsum()


# In[17]:


#先ts_log_diff => ts_log=>ts_log => ts 
#先以ts_log的第一个值作为基数，复制给所有值，然后每个时刻的值累加与第一个月对应的差值(这样就解决了，第1、2个天diff数据为空的问题了)
#然后得到了predictions_ARIMA_log => predictions_ARIMA
predictions_ARIMA_log = pd.Series(indexedDataset_logScale['confirmedCount'].iloc[0], index=indexedDataset_logScale.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA_log.head()


# In[20]:


# Inverse of log is exp.
predictions_ARIMA = np.exp(predictions_ARIMA_log)
#predictions_ARIMA

plt.xticks(rotation=45)
plt.plot(indexedDataset)
plt.plot(predictions_ARIMA)


# In[21]:


#We have 144(existing data of 12 yrs in months) data points. 
#And we want to forecast for additional 120 data points or 10 yrs.
results_ARIMA.plot_predict(1,79) 


# In[22]:


predictions_ARIMA_=results_ARIMA.predict(1,68)#取68天的预测结果


predictions_ARIMA_diff = pd.Series(predictions_ARIMA_, copy=True)
predictions_ARIMA_diff#ARIMA拟合的一阶差分值
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()


#先ts_log_diff => ts_log=>ts_log => ts 
#先以ts_log的第一个值作为基数，复制给所有值，然后每个时刻的值累加与第一个月对应的差值(这样就解决了，第1、2个天diff数据为空的问题了)
#然后得到了predictions_ARIMA_log => predictions_ARIMA
predictions_ARIMA_log = pd.Series(indexedDataset_logScale['confirmedCount'].iloc[0], index=predictions_ARIMA_.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)

# Inverse of log is exp.
predictions_ARIMA = np.exp(predictions_ARIMA_log)
print(predictions_ARIMA)


# In[21]:


Data1 = pd.read_csv('./world0120-0328.csv', index_col=0)#1月20日-3月28日 世界总确诊人数
#增加id递增列
#增加id递增列

Data1 = Data1.reset_index()
Data1['日期']= pd.to_datetime(Data1['日期'],infer_datetime_format=True) 
#set_index( ) 将 DataFrame 中的列转化为行索引
indexedDataset_original = Data1.set_index(['日期'])

#数据反转
indexedDataset_original = indexedDataset_original.iloc[::-1]



from matplotlib import font_manager

# my_font = font_manager.FontProperties(fname="C:/Windows/Fonts/simhei.ttf") #windows黑体
my_font = font_manager.FontProperties(fname="/System/Library/Fonts/PingFang.ttc")
plt.figure(figsize=(10,8),dpi = 80)
plt.xticks(rotation=45,fontsize=10)
plt.yticks(fontsize=10)

plt.plot(predictions_ARIMA,markersize=30,label='预测数据',color = 'blue',linewidth=2.0)#1月20日-3月28日 世界总确诊人数预测曲线
plt.plot(indexedDataset_original,markersize=30,label='实际数据',color = 'orange',linewidth=2.0)#1月20日-3月28日 世界总确诊人数实际曲线
plt.plot(indexedDataset,markersize=30,label='训练数据',color = 'green',linestyle='--',linewidth=2.0)#1月20日-3月18日 世界总确诊人数训练集曲线

##添加描述信息
plt.xlabel("时间日期",fontproperties = my_font,fontsize=15)
plt.ylabel("肺炎确诊总数",fontproperties = my_font,fontsize=15)
plt.title("1月20日至3月28日世界肺炎确诊人数趋势图",fontproperties = my_font,fontsize=20)
plt.grid(alpha=1) #网格线的透明程度  数值越大，网格颜色越深
## 添加图例
#############################待解决问题：为何我的图例 不会变大！！！！############################
plt.legend(prop = my_font,loc="upper left")
#plt.savefig('figure.eps')
plt.show()


# In[23]:


results_ARIMA.fittedvalues


# In[24]:


results_ARIMA.predict(1,69)

