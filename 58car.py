import urllib.request
from bs4 import BeautifulSoup
import pandas as pd


def get_html(url):
    html = urllib.request.urlopen(url)
    htmltext = html.read().decode('utf-8')
    return htmltext


def get_data(htmltext, list_title, list_info, list_price):
    soup = BeautifulSoup(htmltext)
    main_part = soup.find('ul', 'car_list ac_container')
    for x in main_part.children:
        try:
            list_title.append(x.find('h1', 'info_tit').text.strip())
            list_info.append(x.find('div', 'info_param').text.strip())
            list_price.append(x.find('div', 'col col3').text.strip())
        except:
            continue


list_title = []
list_info = []
list_price = []

for i in range(1, 71):
    if i == 1:
        url = 'http://cd.58.com/ershouche/'
    else:
        url = 'http://cd.58.com/ershouche/pn{}/'.format(i)
    htmltext = get_html(url)
    get_data(htmltext, list_title, list_info, list_price)
    print('正在爬取第%d页' % i)

df = pd.DataFrame()
df['Title'] = list_title
df['Info'] = list_info
df['Price'] = list_price
