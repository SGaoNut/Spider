# ----------------------------------------- #
#  从百度地图获取区域边界点的经纬度数据        #
# ----------------------------------------- #
# 导入需要使用的Python库
import requests
import json
import re
import pandas as pd


# --------------------------------------------- #
#   定义从百度地图获取区域边界点经纬度的函数       #
# --------------------------------------------- #
def getRegion_baidu(keyword):
    # 获取uid的网址格式
    uidUrl = "http://map.baidu.com/su?wd={}&cid=289&type=0&pc_ver=2"
    # 通过格式化函数得到网址，并进行抓取
    r_uid = requests.get(uidUrl.format(keyword), headers={'user-agent': 'Mozilla/5.0'})
    # 编码转换
    r_uid.encoding = 'utf-8'
    # 使用正则表达式提取内容
    uids = re.findall('[a-zA-Z0-9]{24}', r_uid.text)
    # 用来保存区域边缘的点的经纬度
    lat_lng = []
    # 循环每一个子区域
    for uid in uids:
        # 把网页上的数据抓取到本地
        poinstUrl = 'http://map.baidu.com/?pcevaname=pc4.1&qt=ext&uid={}&ext_ver=new&l=12'
        r_point = requests.get(poinstUrl.format(uid), headers={'user-agent': 'Mozilla/5.0'})
        r_point.encoding = 'ascii'
        # 转换为python字典类型
        jd = json.loads(r_point.text)
        # 使用正则表达式进行提取
        points = re.findall('[0-9]{8}.[0-9]+,[0-9]{7}.[0-9]+', jd['content']['geo'])
        sub_lat_lng = []
        # 将中间都逗号去掉
        for str in points:
            # 将经纬度分开，并得到实际的经纬度
            temp = str.split(',')
            temp[0] = round(float(temp[0]) / 100000, 6)
            temp[1] = round(float(temp[1]) / 100000, 6)
            sub_lat_lng.append(temp)
            # 转换成dataframe
        lat_lng.append(sub_lat_lng)
        df = pd.DataFrame(lat_lng[0], columns=['longitude', 'latitude'])
        df['name'] = keyword
        # 返回结果
    return df

if __name__ == '__main__':
    print("\n###### Analysis Start ######\n")
    # 获取调用参数：
    # print "name: ", sys.argv[0] # 脚本名
print(getRegion_baidu('华南理工大学').head())
