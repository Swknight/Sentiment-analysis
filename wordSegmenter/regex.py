#-*-coding:utf-8-*-
import re
temp = ",,,一分钱一分货。。。"
temp = temp.decode("utf8")
string = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+".decode("utf8"), ",".decode("utf8"),temp)
print string.strip()
