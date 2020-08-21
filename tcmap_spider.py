from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import pandas as pd


def get_html(url):
    html = urlopen(url).read().decode('gbk')
    return html


def get_city(html_text):
    soup = BeautifulSoup(html_text, features='html.parser')
    city_res = soup.select('tr[bgcolor = "#f8f8f8"]') + soup.select('tr[bgcolor = "#ffffff"]')
    city_table = pd.DataFrame(columns=('city', 'sub_city'))
    for city_n in city_res:
        city_n_1 = city_n.select('strong')
        city_n_2 = city_n.select('td > a')
        for city_sub_n in city_n_2:
            print(city_n_1[0].string)
            print(city_sub_n.string)
            city_table = city_table.append({'city': city_n_1[0].string, 'sub_city': city_sub_n.string},
                                           ignore_index=True)
    return city_table


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
    print(city_name)
    # sub_city_name = get_sub_city(html_text)
    # print(city_name)
