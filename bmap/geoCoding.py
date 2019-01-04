import codecs
import re
import requests
import json

def geoCoding(filename, ak):
    content = ""
# @see 写入文件内容（数组会使用writelines进行写入）codec.open实现
    try:
        fo = codecs.open(filename, 'r', "utf-8")
        print(u"读取文件名：", filename)
        for line in fo.readlines():
            spline = line.split(',')
            grid_id = spline[0]
            city = spline[1]
            address = spline[2]
            address = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",address)

# 调用百度地图正 / 逆地理编码服务API
            url = "http://api.map.baidu.com/geocoder/v2/?address=" + \
                  address + "&city=" + city + "&output=json&ak=" + ak + "&callback=showLocation"
            geo_content = requests.get(url).content.decode('utf-8')
            showloc = re.search(r'showLocation&&showLocation\((.*)\)', geo_content, re.M | re.I)
            hjson = json.loads(showloc.group(1))
            if hjson['status'] == 0:
                lat = hjson['result']['location']['lat']
                lng = hjson['result']['location']['lng']
            else:
                lat = 0
                lng = 0

            result = [grid_id, city, address, lat, lng]
            content += ','.join([str(x) for x in result]) + '\r\n'
    except IOError as e:
        # print "文件不存在或者文件读取失败"
        print(e)
        return ""
    else:
        fo.close()
        return content