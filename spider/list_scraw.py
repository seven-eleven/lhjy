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

def get_cur_season():
    _year = datetime.now().date().year
    _month = datetime.now().date().month
    if int(_month) <= 3:
        _season = 1
    elif int(_month) <= 6:
        _season = 2
    elif int(_month) <= 9:
        _season = 3
    else:
        _season = 4
    return str(_year), str(_season)

class ENDataCrawl:
    def __init__(self):
        #self.dm = DBManager("tk_details")
        self.headers = {
            "User-Agent": ":Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36"
        }

    def start_crawl(self):
        add_info_logs("start_crawl", "-启动爬虫业务-")
        _year, _season = get_cur_season()
        self.get_url()

    def end_crawl(self):
        #self.dm.clsoe_db()
        add_info_logs("end_crawl", "-结束爬虫业务-")

    def get_url(self):
        url = "http://quotes.money.163.com/old/#query=EQA_EXCHANGE_CNSESH&DataType=HS_RANK&sort=PERCENT&order=desc&count=3000&page=0"
        print(url)

        # 请求失败后重新请求(最多8次)
        max_try = 3
        for tries in range(max_try):
            try:
                content = requests.get(url)
                self.parse_pager(content.content)
                break
            except Exception:
                if tries < (max_try - 1):
                    sleep(2)
                    continue
                else:
                    add_error_logs("crawl_error", "501")

    def parse_pager(self, content):
        print(content)
        soup = bs4.BeautifulSoup(content, "html")
        parse_list = soup.select("div.panelContentWrap tr")
        for item in parse_list[1:]:
            data = [x.string for x in item.select("td")]
            print(data[1], data[2])
            # price = {
            #     "date": data[0], # 日期
            #     "price_open": data[1], # 开盘价
            #     "price_max": data[2], # 最高价
            #     "price_min": data[3], # 最低价
            #     "price_close": data[4], # 收盘价
            #     "change": data[5], # 涨跌额
            #     "change_ratio": data[6], # 涨跌幅（%）
            #     "transaction_volume": data[7].replace(",", ""), # 成交量（手）
            #     "transaction_money": data[8].replace(",", ""), # 成交量（金额万元）
            #     "amplitude": data[9], # 振幅（%）
            #     "turnover_rate": data[10] # 换手率（%）
            # }
            #
            # # todo: save to json
            # print(key, price)

        # 在配置较低的机器上运行应该加上这一句
        gc.collect()


if __name__ == '__main__':
    dc = ENDataCrawl()
    dc.start_crawl()
    dc.end_crawl()