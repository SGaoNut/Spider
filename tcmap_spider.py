from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import pandas as pd


# if has Chinese, apply decode()

def get_html(url):
    html = urlopen(url).read().decode('gbk')
    return html


def get_city(html_text):
    soup = BeautifulSoup(html_text, features='html.parser')
    city_res = soup.select("table strong a")
    for c in city_res:
        print(c.get_text())

def get_sub_city(html_text):
    soup = BeautifulSoup(html_text, features='html.parser')
    soup = soup.table
    sub_city_res = soup.select("td > a")
    for sub_c in sub_city_res:
        print(sub_c.get_text())


if __name__ == '__main__':
    base_url = 'http://www.tcmap.com.cn'
    zhejiang_sub_url = '/zhejiangsheng/'
    url = base_url + zhejiang_sub_url
    html_text = get_html(url)
    city_name = get_city(html_text)
    sub_city_name = get_sub_city(html_text)
    print(city_name)

