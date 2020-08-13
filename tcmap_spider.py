from bs4 import BeautifulSoup
from urllib.request import urlopen

# if has Chinese, apply decode()
html = urlopen("http://www.tcmap.com.cn/zhejiangsheng/").read().decode('gbk')

soup = BeautifulSoup(html, features='lxml')

city = soup.find_all('a', {"class": "blue"})
# city = soup.select('page_left table tbody tr td a')
# use class to narrow search
city = soup.find('a', {"class": "blue"})
# city = soup.find_all("a", {"src": re.compile('.*?\.jpg')})
# document.querySelector("#page_left > table:nth-child(3) > tbody > tr:nth-child(2) > td:nth-child(6) > a:nth-child(1)")