import scrawler
import pymysql
from time import sleep

class BatchCrawler:
    def __init__(self, year, season, max_try=3):
        self.year = str(year)
        self.season = str(season)
        self.max_try = max_try
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
    b = BatchCrawler(2020, 1, 3)
    b.batch()