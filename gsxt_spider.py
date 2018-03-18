#coding:UTF-8
import json
import re
import threading
import time

import jpype
import redis
import requests
from bs4 import BeautifulSoup
from jpype import *


jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=/code/java/forpython/target/classes/")

class SearchItem(threading.Thread):
    session=requests.session()
    keyword=""
    proxy=""
    semaphore=None
    def getGTChallenge(self):
        print("getGTChallenge start")
        loginurl="http://www.gsxt.gov.cn/SearchItemCaptcha"
        result=self.session.get(loginurl)
        if "y.replace(" not in result.text:
            raise Exception("被屏蔽了")
        mycookies= result.cookies
        jpype.attachThreadToJVM()
        jpype.isThreadAttachedToJVM()
        A = jpype.JClass("com.GovTest")
        self.Aobj=A()
        fu=self.Aobj.challenge(result.text)
        print("fu="+fu)
        jslarr= fu.split("=")
        jsl_clearance=jslarr[1]
        self.session.cookies['__jsl_clearance']=jsl_clearance
        result=self.session.get(loginurl)
        challengeJson=json.loads(result.text)
        return  challengeJson

    def getImageGif(self):
        print("getImageGif start")
        url="http://www.gsxt.gov.cn/corp-query-custom-geetest-image.gif?v="
        localTime=time.localtime(time.time())
        url=url+str(localTime.tm_min+localTime.tm_sec)
        resp=self.session.get(url)
        aaa=self.Aobj.getImageGif(resp.text)
        matchObj = re.search( 'location_info = (\d+);', aaa)
        if matchObj:
            return matchObj.group(1)
        else:
            Exception("没有找到location_info")

    def getValidateInput(self,location_info):
        print("getValidateInput start")
        url="http://www.gsxt.gov.cn/corp-query-geetest-validate-input.html?token="+location_info
        resp=self.session.get(url)
        aaa=self.Aobj.getImageGif(resp.text)
        matchObj = re.search( 'value: (\d+)}', aaa)
        if matchObj:
            location_info= matchObj.group(1)
            token=int(location_info) ^ 536870911;
            print("token=",token)
            return str(token)
        else:
            Exception("没有找到location_info")

    def searchTest(self,keyword):
        print("searchTest start")
        url="http://www.gsxt.gov.cn/corp-query-search-test.html?searchword="+keyword
        resp=self.session.get(url);
        print("searchTest ",resp.text)

    def jianYan(self,challengeJson):
        print("jianYan start")
        url="http://jiyanapi.c2567.com/shibie?user=帐号&pass=密码&gt="+challengeJson["gt"]+"&challenge="+challengeJson["challenge"]+"&referer=http://www.gsxt.gov.cn&return=json&format=utf8"
        sess=requests.session()
        resp=sess.get(url);
        jiyanJson=  json.loads(resp.text)
        print resp.text
        return jiyanJson

    def querySearch(self,jiYanJson,token,keyword):
        print("querySearch start")
        url="http://www.gsxt.gov.cn/corp-query-search-1.html"
        postData={
            'tab':'ent_tab',
            'province':'',
            'geetest_challenge':jiYanJson['challenge'],
            'geetest_validate':jiYanJson['validate'],
            'geetest_seccode':jiYanJson['validate']+'|jordan',
            'token':token,
            'searchword':keyword
        }
        resp=self.session.post(url,postData)
        return resp.text ,postData

    def dealPageUrl(self,html):
        print("dealPageUrl start")
        soup = BeautifulSoup(html,"html.parser")
        urlsItem=soup.find_all("a",class_="search_list_item db")
        pageNums=0
        for urlItem in urlsItem:
            print("urlItem['href']=",urlItem['href'])
        if len(urlsItem)>1:
            pageForm=soup.find_all(id="pageForm")
            tabAs=pageForm[0].find_all("a",text=re.compile("\d+"))
            pageNums=len(tabAs)
        return pageNums

    def dealPageUrlNum(self,pageNums,postData):
        print("dealPageUrlNum start")
        url="http://www.gsxt.gov.cn/corp-query-search-advancetest.html"
        for i in range(pageNums):
            postData['page']=i+1
            resp=self.session.get(url,params=postData)
            soup = BeautifulSoup(resp.text)
            urlsItem=soup.find_all("a",class_="search_list_item db")
            for urlItem in urlsItem:
                print("urlItem['href']=",urlItem['href'])


    def getCorpUrl(self):
        self.session.timeout=1
        self.session.max_redirects=1
        if self.proxy:
            self.session.proxies={ "http": "http://"+self.proxy, "https": "http://"+self.proxy, }
        headers={'Host': 'www.gsxt.gov.cn',
                 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:57.0) Gecko/20100101 Firefox/57.0',
                 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                 'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
                 'Accept-Encoding': 'gzip, deflate',
                 'Referer': 'http://www.gsxt.gov.cn/SearchItemCaptcha',
                 'Connection': 'keep-alive',
                 'Upgrade-Insecure-Requests': '1',
                 'Cache-Control': 'max-age=0, no-cache'}
        self.session.headers=headers
        challengeJson=self.getGTChallenge()
        localtion_info= self.getImageGif()
        token=self.getValidateInput(localtion_info)
        self.searchTest(self.keyword)
        jiyanJson=self.jianYan(challengeJson)
        html,postData=self.querySearch(jiyanJson,token,self.keyword)
        pageNums=self.dealPageUrl(html)
        print 'pageNums=',pageNums
        self.dealPageUrlNum(pageNums,postData)
        return 1

    def run(self):
        try:
            self.getCorpUrl()
        except Exception,e:
            print("run exception ",e.message)
        self.session.close()
        self.semaphore.release()
        print "search Item run finish"

    def __init__(self, keyword,proxy,semaphore):
        threading.Thread.__init__(self)
        self.keyword = keyword
        self.proxy = proxy
        self.semaphore = semaphore


semaphore=threading.Semaphore(1)
while 1:
        try:
            semaphore.acquire()
            t1=SearchItem("百度",None,semaphore)
            t1.start()
        except Exception, e:
            print('main e.message:\t', e.message)
        time.sleep(1)