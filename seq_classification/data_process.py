#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/26 14:23
# @Author  : 云帆
# @Site    : 
# @File    : data_process.py
# @Software: PyCharm
import pandas as pd
import re
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

def news_data_process(file_path):
    news_type = []
    news_content = []
    with open(file_path, mode='r', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split('\t')
            news_type.append(line[0])
            news_content.append(ILLEGAL_CHARACTERS_RE.sub(r'',line[1]))
    return news_type, news_content


if __name__ == '__main__':
    news_type, news_content = news_data_process('./data/cnews/cnews.test.txt')
    df = pd.DataFrame({'news_type': news_type, 'news_content': news_content})
    df.to_excel('./data/cnews/test.xlsx', index=False)
    # news_type_set = set(news_type)
    # with open('./data/cnews/news_type.txt', mode='w', encoding='utf8') as f:
    #     for index, item in enumerate(news_type_set):
    #         f.writelines('{}\t{}\n'.format(item, index))