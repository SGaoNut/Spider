from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
import pandas as pd


def get_html(url):
    html = urlopen(url).read().decode('gbk')
    return html


def get_city(html_text):
    soup = BeautifulSoup(html_text, features='lxml')
    city_res = soup.select('tr[bgcolor = "#f8f8f8"]') + soup.select('tr[bgcolor = "#ffffff"]')
    base_url = 'http://www.tcmap.com.cn'
    city_table = pd.DataFrame(columns=('city', 'sub_city', 'sub_city_url'))
    for city_n in city_res:
        city_n_1 = city_n.select('strong')
        city_n_2 = city_n.select('td > a')
        for city_sub_n in city_n_2:
            sub_city_url = city_sub_n.get('href')
            sub_city_url = base_url + sub_city_url
            sub_city_url_html = urlopen(sub_city_url).read().decode('gbk')
            soup_street = BeautifulSoup(sub_city_url_html, features='lxml')
            street_res = soup_street.select('strong > a')
            for street_sub_n in street_res:
                street_sub_name = street_sub_n.get_text()
                city_table = city_table.append(
                    {'city': city_n_1[0].string, 'sub_city': city_sub_n.string, 'sub_city_url': sub_city_url,
                     'street_sub': street_sub_name}
                    , ignore_index=True)
    return city_table


# def get_street(html_text):
#     soup = BeautifulSoup(html_text, features='html.parser')
#     street_res = soup.select('strong')


if __name__ == '__main__':
    base_url = 'http://www.tcmap.com.cn'
    zhejiang_sub_url = '/zhejiangsheng/'
    url = base_url + zhejiang_sub_url
    html_text = get_html(url)
    city_name = get_city(html_text)
    # for sub_city_url in city_name[['sub_city_url']]:
    #     html_text = sub_city_url
    #     street_info = get_street(html_text)
    print(city_name)
