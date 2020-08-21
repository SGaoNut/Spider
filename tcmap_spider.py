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
    # city_res = soup.select("table tr")
    city_res_1 = soup.select('tr[bgcolor = "#f8f8f8"]')
    city_res_2 = soup.select('tr[bgcolor = "#ffffff"]')
    city_res = city_res_1 + city_res_2
    # city_table = []
    city_table = pd.DataFrame(columns=('city', 'sub_city'))
    for city_n in city_res:
        city_n_1 = city_n.select('strong')
        city_n_2 = city_n.select('td > a')
        # city_table.append(city_n_1[0].string)
        # city_table.loc[0] = city_n_1[0].string
        for city_sub_n in city_n_2:
            # city_sub_n_1 = city_sub_n.select('td > a')
            print(city_n_1[0].string)
            print(city_sub_n.string)
            city_table = city_table.append({'city': city_n_1[0].string, 'sub_city': city_sub_n.string}, ignore_index=True)
            # city_table['city'].append(city_n_1[0].string)
            # city_table['sub_city'].append(city_sub_n_1[0].string)
            # city_table = city_table.append(pd.DataFrame(city_n_1[0].string)
        # city_table = city_table.append(pd.DataFrame(city_n_1[0].string)
        # print(city_n_1[0].string)
    return city_table





    # city_res_0 = city_res[0]
    # for c in city_res:
    #     print(c.get_text())

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

