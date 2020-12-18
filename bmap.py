# -*- coding: utf-8 -*-
"""
脚本运行命令：python bmap.py source.csv target.csv 百度地图API应用AK
sys.argv获取命令传入的参数
python bmap.py test.csv target.csv SCTnixYv4j2bQNnwZTD07as3VsiG2er8
bmap_test_tz_ls_zs.csv
"""

import sys
from geo_file import geo_file
# from geo_file import geo_coding as geo_coding
# from geo_file import write_file as write_file

if __name__ == '__main__':
    print("\n###### Analysis Start ######\n")
    # 获取调用参数：
    # print "name: ", sys.argv[0] # 脚本名
    sourceFile = sys.argv[1]
    targetFile = sys.argv[2]
    bmapAK = sys.argv[3]

    geoCoder = geo_file.geo_coding(sourceFile, bmapAK)
    geo_file.write_file(targetFile, geoCoder)
