#-*-coding:utf-8-*-
import xlrd

data = xlrd.open_workbook('pingce.xlsx')
sh = data.sheet_by_name("pingce")
fout = open("pingce1", 'w')
for n in range(sh.nrows):
    text = sh.cell_value(n,1).encode('utf-8')
    fout.write(text+'\n')
fout.close()