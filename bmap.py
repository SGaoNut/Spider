# -*- coding: utf-8 -*-
"""
脚本运行命令：python bmap.py source.csv target.csv 百度地图API应用AK
sys.argv获取命令传入的参数
python bmap.py test.csv target.csv SCTnixYv4j2bQNnwZTD07as3VsiG2er8
python bmap.py bmap_test_nb_1.csv bmap_target_nb_1.csv SCTnixYv4j2bQNnwZTD07as3VsiG2er8
python bmap.py bmap_test_huzhou.csv bmap_target_huzhou.csv SCTnixYv4j2bQNnwZTD07as3VsiG2er8
"""

import sys
import requests
import codecs
import re
import json

# reload(sys)
# sys.setdefaultencoding('utf-8')


# @see 读取文件内容
# @see 调用百度地图正/逆地理编码服务API
# @see 写入文件内容（数组会使用writelines进行写入）codec.open实现
# @param filename,toFile 文件名
#        ak 内容
def geoCoding(filename, ak):
    content = ""
    try:
        fo = codecs.open(filename, 'r', "utf-8")
        print(u"读取文件名：", filename)
        for line in fo.readlines():
            spline = line.split(',')
            grid_id = spline[0]
            city = spline[1]
            address = spline[2].replace('\r\n', '').replace('\n', '')

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


def writeFile(toFile, content):
    try:
        fo = codecs.open(toFile, 'wb', "utf-8")
        print(u"文件名：", toFile)
        if type(content) == type([]):
            fo.writelines(content)
        else:
            fo.write(content)
    except IOError:
        print(u"没有找到文件或文件读取失败")
    else:
        print(u"文件写入成功")
        fo.close()


if __name__ == '__main__':
    print("\n###### Analysis Start ######\n")
    # 获取调用参数：
    # print "name: ", sys.argv[0] # 脚本名
    sourceFile = sys.argv[1]
    targetFile = sys.argv[2]
    bmapAK = sys.argv[3]

    geoCoder = geoCoding(sourceFile, bmapAK)
    writeFile(targetFile, geoCoder)