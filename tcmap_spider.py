from bs4 import BeautifulSoup
from urllib.request import urlopen
import re

# if has Chinese, apply decode()
html = urlopen("http://www.tcmap.com.cn/zhejiangsheng/").read().decode('gbk')

soup = BeautifulSoup(html, features='html.parser')

city_table_tag = soup.select("table strong a")
# city_table_strong = city_table_tag.find_all('strong')


city_result = city_table_tag.select('a[class="blue"]')

# city = soup.find('table').find_all('strong')
# <strong><a href=/zhejiangsheng/hangzhou.html  class=blue>杭州市</a></strong></td>
# city_table = soup.find_all('a',href = re.compile(".+qu\.html"))
# city_table_f8 = city_table.find_all('<strong><a href(.*?)</strong></td>')
# city = soup.find_all('a', {"class": "blue"})
# city = soup.select('page_left table tbody tr td a')
# use class to narrow search
city = soup.find('a', {"class": "blue"})
# city = soup.find_all("a", {"src": re.compile('.*?\.jpg')})
# document.querySelector("#page_left > table:nth-child(3) > tbody > tr:nth-child(2) > td:nth-child(6) > a:nth-child(1)")