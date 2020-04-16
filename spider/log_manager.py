"""
__title__ = '日志管理类'
__author__ = 'Richer'
__mtime__ = '2020/2/15'
"""
import os
from datetime import datetime


def add_error_logs(tag, code, content):
    path = os.path.dirname(os.path.abspath(__file__)) + "/logs_error.txt"
    fo = open(path, "a")
    log_dict = {
        "tag": tag,
        "code": code,
        "content": content,
        "created_time": str(datetime.now().date())+" "+str(datetime.now().time())
    }
    fo.write(str(log_dict)+"\n")
    fo.close()


def add_info_logs(tag, content):
    path = os.path.dirname(os.path.abspath(__file__)) + "/logs_info.txt"
    fo = open(path, "a")
    log_dict = {
        "tag": tag,
        "content": content,
        "created_time": str(datetime.now().date())+" "+str(datetime.now().time())
    }
    fo.write(str(log_dict) + "\n")
    fo.close()