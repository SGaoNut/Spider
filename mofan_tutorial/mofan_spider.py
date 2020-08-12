import re
from bs4 import BeautifulSoup
from urllib.request import urlopen

html = urlopen("https://morvanzhou.github.io/static/scraping/basic-structure.html").read().decode('utf-8')
print(html)

res = re.findall(r"<title>(.+?)</title>", html)
print(res)

res = re.findall(r"<p>(.*?)</p>", html, flags=re.DOTALL)
print(res)

soup = BeautifulSoup(html, features='html.parser')
print(soup.h1)
print('\n', soup.p)

all_href = soup.find_all('a')
for l in all_href:
    print(l['href'])
