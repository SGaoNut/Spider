from bs4 import BeautifulSoup
from urllib.request import urlopen

html = urlopen("https://morvanzhou.github.io/static/scraping/list.html").read().decode('utf-8')

soup = BeautifulSoup(html, features='html.parser')

month = soup.find_all('li', {'class': 'month'})
for m in month:
    print(m.get_text())

day_jan = soup.find_all('ul', {'class': 'jan'})
for d in day_jan:
    print(d.get_text())


