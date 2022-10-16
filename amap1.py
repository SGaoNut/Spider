#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 29/01/2018 11:47 PM
# @Author  : Shan Gao
# @Site    : 
# @File    : amap1.py
# @Software: PyCharm

import requests


def getSubName(keywords):  # 获取搜索区域的名称，部分区域例如鼓楼重名太多，因此返回城市代码，将城市代码作为参数给上述函数
    parameters = {'keywords': keywords, 'key': '67f6505babcddb0f4de8d358a2a98a4a'}
    base = 'http://restapi.amap.com/v3/config/district?'
    response = requests.get(base, parameters)
    answer = response.json()
    print(keywords + "的经纬度：", answer)


if __name__ == '__main__':
 keywords = '北京市海淀区'
getSubName(keywords)
