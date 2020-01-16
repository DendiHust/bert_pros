#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/3 9:54
# @Author  : 云帆
# @Site    : 
# @File    : data_process.py
# @Software: PyCharm
import json
from tqdm import tqdm
import codecs
import re

all_50_schemas = set()

with open('./source_data/all_50_schemas', mode='rb') as f:
    for l in tqdm(f):
        a = json.loads(l)
        all_50_schemas.add(a['predicate'])

id2predict = {i: j for i, j in enumerate(all_50_schemas)}
predict2id = {j: i for i, j in id2predict.items()}

with codecs.open('./pro_data/all_50_schemas.json', mode='w', encoding='utf-8') as f:
    json.dump([id2predict, predict2id], f, indent=4, ensure_ascii=False)

# 训练数据集
train_data_list = []

# 字符集
chars = {}
#
min_count = 2

def repair(d):
    d['text'] = d['text'].lower()
    something = re.findall(u'《([^《》]*?)》', d['text'])
    something = [s.strip() for s in something]
    zhuanji = []
    gequ = []
    for sp in d['spo_list']:
        sp[0] = sp[0].strip(u'《》').strip().lower()
        sp[2] = sp[2].strip(u'《》').strip().lower()
        for some in something:
            if sp[0] in some and d['text'].count(sp[0]) == 1:
                sp[0] = some
        if sp[1] == u'所属专辑':
            zhuanji.append(sp[2])
            gequ.append(sp[0])
    spo_list = []
    for sp in d['spo_list']:
        if sp[1] in [u'歌手', u'作词', u'作曲']:
            if sp[0] in zhuanji and sp[0] not in gequ:
                continue
        spo_list.append(tuple(sp))
    d['spo_list'] = spo_list
max_length = 0
with open('./source_data/train_data.json', mode='rb') as f:
    for l in tqdm(f, desc='process train data'):
        a = json.loads(l)
        if len(a['text']) > max_length:
            max_length = len(a['text'])
        if not a['spo_list']:
            continue
        a = {
            'text': a['text'],
            'spo_list': [[item['subject'], item['predicate'], item['object']] for item in a['spo_list']]
        }
        repair(a)
        train_data_list.append(a)

        for c in a['text']:
            chars[c] = chars.get(c, 0) + 1
print('max_length:{}'.format(max_length))
with codecs.open('./pro_data/train_data.json', mode='w', encoding='utf8') as f:
    json.dump(train_data_list, f, indent=4, ensure_ascii=False)

with codecs.open('./pro_data/train_data-sim.json', mode='w', encoding='utf8') as f:
    json.dump(train_data_list[:100000], f, indent=4, ensure_ascii=False)

# with codecs.open('./pro_data/train_data-sim.json', mode='w', encoding='utf8') as f:
#     json.dump(train_data_list[:1000], f, indent=4, ensure_ascii=False)


dev_data_list = []

with open('./source_data/dev_data.json', mode='rb') as f:
    for l in tqdm(f, desc='process dev data'):
        a = json.loads(l)
        if not a['spo_list']:
            continue

        a = {
            'text': a['text'],
            'spo_list': [[item['subject'], item['predicate'], item['object']] for item in a['spo_list']]
        }
        repair(a)
        dev_data_list.append(a)

        for c in a['text']:
            chars[c] = chars.get(c, 0) + 1

with codecs.open('./pro_data/dev_data.json', mode='w', encoding='utf8') as f:
    json.dump(dev_data_list, f, indent=4, ensure_ascii=False)

with codecs.open('./pro_data/all_chars.json', mode='w', encoding='utf8') as f:
    chars = {i: j for i, j in chars.items() if j >= min_count}
    id2char = {i + 2: j for i, j in enumerate(chars)}
    char2id = {j: i for i, j in id2char.items()}
    json.dump([id2char, char2id], f, indent=4, ensure_ascii=False)
