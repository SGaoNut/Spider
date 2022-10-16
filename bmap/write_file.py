# -*- coding: utf-8 -*-
import codecs


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
