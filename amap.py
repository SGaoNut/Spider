#!/usr/bin/env
# -*- coding:utf-8 -*-
'''''
利用高德地图api实现经纬度与地址的批量转换
'''
'''
import requests


def geocode(address):
    parameters = {'address': address, 'key': '67f6505babcddb0f4de8d358a2a98a4a'}
    base = 'http://restapi.amap.com/v3/geocode/geo'
    response = requests.get(base, parameters)
    answer = response.json()
    print(address + "的经纬度：", answer['geocodes'][0]['location'])


if __name__ == '__main__':
    # address = input("请输入地址:")
    address = '北京市海淀区'
    geocode(address)
'''

import http.client
import json
from urllib.parse import quote_plus

base = '/v3/geocode/geo'
key = '67f6505babcddb0f4de8d358a2a98a4a'


def geocode(address):
    path = '{}?address={}&key={}'.format(base, quote_plus(address), key)
    # print(path)
    connection = http.client.HTTPConnection('restapi.amap.com', 80)
    connection.request('GET', path)
    rawreply = connection.getresponse().read()
    # print(rawreply)
    reply = json.loads(rawreply.decode('utf-8'))
    print(address + '的经纬度：', reply['geocodes'][0]['location'])

if __name__ == '__main__':
    # address = input("请输入你的地址：")
    address = '杭州市上城区'
    geocode(address)
