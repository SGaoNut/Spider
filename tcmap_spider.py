from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import pandas as pd


# if has Chinese, apply decode()

def get_html(url):
    html_text = urlopen(url).read().decode('gbk')
    return html_text


def get_city(html_text):
    soup = BeautifulSoup(html_text, features='html.parser')
    city_res = soup.select("table strong a")
    for c in city_res:
        print(c.get_text())




# def city(url):
#     html = urlopen(url).read().decode('gbk')
#     soup = BeautifulSoup(html, features='html.parser')
#     city_res = soup.select("table strong a")
#     for city_list in city_res:
#         print(city_list.get_text())
#         return city_res


if __name__ == '__main__':
    # address = input("请输入你的地址：")
    base_url = 'http://www.tcmap.com.cn'
    zhejiang_sub_url = '/zhejiang/'
    url = base_url + zhejiang_sub_url
    html_text = get_html(url)
    city_name = get_city(html_text)

