import pymysql

c = pymysql.connect(host='172.16.38.150',
                port=3306,
                user='richer',
                password='daming',
                database='rich',
                charset='utf8')

print(c)