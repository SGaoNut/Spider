# -*- coding: utf-8 -*-
import codecs
import re
import requests
import json
import codecs

class geo_file():

    # 写入文件内容（数组会使用writelines进行写入）codec.open实现
    def geo_coding(filename, ak):
        content = ""
        try:
            fo = codecs.open(filename, 'r', "utf-8")
            print(u"读取文件名：", filename)
            for line in fo.readlines():
                spline = line.split(',')
                grid_id = spline[0]
                city = spline[1]
                address = spline[2]
                address = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", address)

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

    def write_file(toFile, content):
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
