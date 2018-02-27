#
# -*- coding: utf-8 -*-
# 第一行必须有，否则报中文字符非ascii码错误
'''
在Pytho2.x中使用import urllib2——-对应的，在Python3.x中会使用import urllib.request，urllib.error。
在Pytho2.x中使用import urllib——-对应的，在Python3.x中会使用import urllib.request，urllib.error，urllib.parse。
在Pytho2.x中使用import urlparse——-对应的，在Python3.x中会使用import urllib.parse。
在Pytho2.x中使用import urlopen——-对应的，在Python3.x中会使用import urllib.request.urlopen。
在Pytho2.x中使用import urlencode——-对应的，在Python3.x中会使用import urllib.parse.urlencode。
在Pytho2.x中使用import urllib.quote——-对应的，在Python3.x中会使用import urllib.request.quote。
在Pytho2.x中使用cookielib.CookieJar——-对应的，在Python3.x中会使用http.CookieJar。
在Pytho2.x中使用urllib2.Request——-对应的，在Python3.x中会使用urllib.request.Request。
http://blog.csdn.net/fengxinlinux/article/details/77281253
'''


import requests
import numpy as np
import json
import pandas as pd
from pandas import Series, DataFrame


def getlnglat(address):
    url = 'http://restapi.amap.com/v3/config/district?'

    # 高德上申请的key
    key = '67f6505babcddb0f4de8d358a2a98a4a'
    uri = url + 'keywords=' + address + '&key=' + key + '&subdistrict=1' + '&extensions=all'

    # 访问链接后，api会回传给一个json格式的数据
    temp = urllib.request.urlopen(uri)
    temp = json.loads(temp.read())

    # polyline是坐标，name是区域的名字
    Data = temp["districts"][0]['polyline']
    name = temp["districts"][0]['name']
    # polyline数据是一整个纯文本数据，不同的地理块按照|分，块里面的地理信息按照；分，横纵坐标按照，分，因此要对文本进行三次处理
    Data_Div1 = Data.split('|')  # 对结果进行第一次切割，按照|符号
    len_Div1 = len(Data_Div1)  # 求得第一次切割长度

    num = 0
    len_Div2 = 0  # 求得第二次切割长度，也即整个数据的总长度
    while num < len_Div1:
        len_Div2 += len(Data_Div1[num].split(';'))
        num += 1

    num = 0
    num_base = 0
    output = np.zeros((len_Div2, 5)).astype(np.float)  # 循环2次，分割；与，
    while num < len_Div1:
        temp = Data_Div1[num].split(';')
        len_temp = len(temp)
        num_temp = 0
        while num_temp < len_temp:
            output[num_temp + num_base, :2] = np.array(temp[num_temp].split(','))  # 得到横纵坐标
            output[num_temp + num_base, 2] = num_temp + 1  # 得到横纵坐标的连接顺序
            output[num_temp + num_base, 3] = num + 1  # 得到块的序号
            num_temp += 1
        num_base += len_temp
        num += 1

    output = DataFrame(output, columns=['经度', '纬度', '连接顺序', '块', '名称'])
    output['名称'] = name

    return output


def getSubName(keywords):  # 获取搜索区域的名称，部分区域例如鼓楼重名太多，因此返回城市代码，将城市代码作为参数给上述函数
    parameters = {'keywords':keywords,'key':'67f6505babcddb0f4de8d358a2a98a4a'}
    base = 'http://restapi.amap.com/v3/config/district?'
    response = requests.get(base,parameters)
    answer = response.json()

'''
    url = 'http://restapi.amap.com/v3/config/district?'

    key = '67f6505babcddb0f4de8d358a2a98a4a'
    uri = url + 'keywords=' + address + '&key=' + key + '&subdistrict=1' + '&extensions=all'

    temp = urllib.request.urlopen(uri)
    temp = json.loads(temp.read())

    list0 = temp['districts'][0]['districts']
    num_Qu = 0
    output = []
    while num_Qu < len(list0):
        output.append(list0[num_Qu]['adcode'])
        num_Qu += 1

    return output

'''
num = 0
ad = getSubName('福建')  # 得到福州下属区域的城市代码
add = getlnglat('福建')  # 得到福州整个的边界数据
while num < len(ad):
    add = pd.concat([add, getlnglat(ad[num].encode("utf-8"))])  # 得到福州下属的全部区域的边界数据
    num += 1
add.to_csv('add.csv', encoding='gbk')  # 输出到文件
