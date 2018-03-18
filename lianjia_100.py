import urllib.request
from bs4 import BeautifulSoup
import pandas as pd


def get_html(url):
    html = urllib.request.urlopen(url)  # 请求网页，并打开
    htmltext = html.read().decode('utf-8')  # 读取内容，并解码
    return htmltext


def get_data(htmltext, list_http, list_title, list_size, list_position, list_followinfo, list_price):
    soup = BeautifulSoup(htmltext)
    main_part = soup.find('ul', 'sellListContent')  # 将含有车辆信息的主要HTML代码块找到
    items = main_part.find_all('li', 'clear')  # 将这个主代码块按照每个房子一条tag，分成若干条
    for item in items:  # 对每个房子进行迭代处理
        try:  # try 避免空行导致出错
            item.text  # 判断是否为空行
            try:  # try 避免有漏掉某一方面信息导致出错
                list_http.append(item.find('a', href=True)['href'])  # 将网址提取，并放入list_http列表中
                list_title.append(item.find('div', 'title').text)
                list_size.append(item.find('div', 'houseInfo').text)
                list_position.append(item.find('div', 'positionInfo').text)
                list_followinfo.append(item.find('div', 'followInfo').text)
                list_price.append(item.find('div', 'priceInfo').text)
            except:
                continue
        except:
            continue


# 创建6个空列表，以便存放6个方面的信息
list_http = []
list_title = []
list_size = []
list_position = []
list_followinfo = []
list_price = []

# 开始迭代操作
for i in range(1, 101):
    url = 'https://cd.lianjia.com/ershoufang/pg%d/' % i
    htmltext = get_html(url)
    get_data(htmltext, list_http, list_title, list_size, list_position, list_followinfo, list_price)
    print('爬取完成第%d页' % i)

# 将爬取的信息放入空的DataFrame里面，并存入excel
df = pd.DataFrame()
df['标题'] = list_title
df['信息'] = list_size
df['位置'] = list_position
df['其他'] = list_followinfo
df['价格'] = list_price
df['网址'] = list_http
df.to_excel(r'/Users/shan/PycharmProjects/ershoufang.xlsx')
