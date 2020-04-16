import scrawler
import pymysql
import datetime
from time import sleep

def get_year_season(d):
    '''
    :param d: like '2020-02-20'
    :return: str(year), str(season)
    '''
    s = d.split("-")
    _year = s[0]
    _month = s[1]
    if int(_month) <= 3:
        _season = 1
    elif int(_month) <= 6:
        _season = 2
    elif int(_month) <= 9:
        _season = 3
    else:
        _season = 4
    return _year, str(_season)

class OnedayCrawler:
    def __init__(self, d, max_try=3):
        _year, _season = get_year_season(d)
        self.year = _year
        self.season = _season
        self.max_try = max_try
        self.date = d
        self.dbc = pymysql.connect(host='172.16.38.150',
                                   port=3306,
                                   user='richer',
                                   password='daming',
                                   db='rich',
                                   charset='utf8')

    def batch(self):
        # get stock list from db
        cursor = self.dbc.cursor()
        cursor.execute('select distinct code from `stock_list`')
        codes = cursor.fetchall()

        for i in range(len(codes)):
            code = codes[i][0]
            print("Start process {}".format(code))

            ## crawler data
            c = scrawler.StockCrawler(code, self.year, self.season, self.max_try)
            rs = c.crawl()

            ## save to db
            for r in rs:
                sale_date = str(r["sale_date"])
                if sale_date != self.date:
                    continue

                price_open = float(r["price_open"])
                price_max = float(r["price_max"])
                price_min = float(r["price_min"])
                price_close = float(r["price_close"])
                up_value = float(r["up_value"])
                up_ratio = float(r["up_ratio"])
                deal_volume = int(r["deal_volume"])
                deal_money = int(r["deal_money"])
                amplitude = float(r["amplitude"])
                turnover_rate = float(r["turnover_rate"])
                #print(type(sale_date), type(amplitude), type(deal_money))

                # not exist to add
                cursor.execute('select * from `stock_sales` where `code` = %s and `sale_date` = %s', (code, sale_date))
                sales = cursor.fetchone()
                if sales == None or len(sales) == 0:
                    cursor.execute(
                        'insert into `stock_sales` (`code`, `sale_date`, `price_open`, `price_max`, `price_min`, '
                        '`price_close`, `up_value`, `up_ratio`, `deal_volume`, `deal_money`, `amplitude`, `turnover_rate`) '
                        'values(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)',
                        (code, sale_date, price_open, price_max, price_min, price_close, up_value, up_ratio, deal_volume, deal_money, amplitude, turnover_rate))
                    self.dbc.commit()

            print("Finish process {}".format(code))
            sleep(0.5)

if __name__ == '__main__':
    b = OnedayCrawler(datetime.date.today())
    b.batch()