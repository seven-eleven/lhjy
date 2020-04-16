"""
__title__ = '数据爬虫'
__author__ = 'Richer'
__mtime__ = '2020/2/15'
"""
from time import sleep
import gc
import bs4
from datetime import datetime

import requests
from log_manager import add_error_logs, add_info_logs

# def get_cur_season():
#     _year = datetime.now().date().year
#     _month = datetime.now().date().month
#     if int(_month) <= 3:
#         _season = 1
#     elif int(_month) <= 6:
#         _season = 2
#     elif int(_month) <= 9:
#         _season = 3
#     else:
#         _season = 4
#     return str(_year), str(_season)

class StockCrawler:
    def __init__(self, code, year, season, max_try):
        self.headers = {
            "User-Agent": ":Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36"
        }
        self.year = str(year)
        self.season = str(season)
        self.code = code
        self.max_try = max_try

    def crawl(self):
        url = "http://quotes.money.163.com/trade/lsjysj_" + self.code + ".html?year=" + \
              self.year + "&season=" + self.season
        print(url)

        result = None
        for tries in range(self.max_try):
            try:
                content = requests.get(url)
                result = self.parse(content.content, self.code)
                break
            except Exception:
                if tries < (self.max_try - 1):
                    sleep(2)
                    continue
                else:
                    add_error_logs("crawl_error", "501", key)

        return result

    def parse(self, content, code):
        soup = bs4.BeautifulSoup(content, "html")
        parse_list = soup.select("div.inner_box tr")

        sales = []
        for item in parse_list[1:]:
            data = [x.string for x in item.select("td")]
            sale = {
                "sale_date": data[0], # 日期
                "price_open": data[1].replace(",", ""), # 开盘价
                "price_max": data[2].replace(",", ""), # 最高价
                "price_min": data[3].replace(",", ""), # 最低价
                "price_close": data[4].replace(",", ""), # 收盘价
                "up_value": data[5], # 涨跌额
                "up_ratio": data[6], # 涨跌幅（%）
                "deal_volume": data[7].replace(",", ""), # 成交量（手）
                "deal_money": data[8].replace(",", ""), # 成交量（金额万元）
                "amplitude": data[9], # 振幅（%）
                "turnover_rate": data[10] # 换手率（%）
            }
            sales.append(sale)

        # 在配置较低的机器上运行应该加上这一句
        gc.collect()
        return sales

# if __name__ == '__main__':
#     dc = ENDataCrawl()
#     dc.start_crawl()
#     dc.end_crawl()