#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/15 16:45
# @Author  : 云帆
# @Site    : 
# @File    : data_pro.py
# @Software: PyCharm
import json
from tqdm import tqdm

label_set = set()

train_dataset = []
with open('./example.train', mode='r', encoding='utf8')  as f:
    text_tmp = []
    label_tmp = []
    for line in tqdm(f.readlines()):
        line = line.strip()
        item_list = line.split(' ')
        if len(item_list) < 2:
            train_dataset.append({'text': ''.join(text_tmp), 'label': label_tmp})
            text_tmp = []
            label_tmp = []
        elif len(item_list) == 2:
            text_tmp.append(item_list[0])
            label_tmp.append(item_list[1])
            label_set.add(item_list[1])

with open('./train.json', mode='w', encoding='utf8') as f:
    json.dump(train_dataset, f, indent=4, ensure_ascii=False)

with open('./label.json', mode='w', encoding='utf8') as f:
    label2id = {i: j for i, j in enumerate(label_set)}
    id2label = {j: i for i, j in label2id.items()}
    json.dump([label2id, id2label], f, indent=4, ensure_ascii=False)

test_dataset = []

with open('./example.test', mode='r', encoding='utf8')  as f:
    text_tmp = []
    label_tmp = []
    for line in tqdm(f.readlines()):
        line = line.strip()
        item_list = line.split(' ')
        if len(item_list) < 2:
            test_dataset.append({'text': ''.join(text_tmp), 'label': label_tmp})
            text_tmp = []
            label_tmp = []
        elif len(item_list) == 2:
            text_tmp.append(item_list[0])
            label_tmp.append(item_list[1])

with open('./test.json', mode='w', encoding='utf8') as f:
    json.dump(test_dataset, f, indent=4, ensure_ascii=False)


dev_dataset = []
with open('./example.dev', mode='r', encoding='utf8')  as f:
    text_tmp = []
    label_tmp = []
    for line in tqdm(f.readlines()):
        line = line.strip()
        item_list = line.split(' ')
        if len(item_list) < 2:
            dev_dataset.append({'text': ''.join(text_tmp), 'label': label_tmp})
            text_tmp = []
            label_tmp = []
        elif len(item_list) == 2:
            text_tmp.append(item_list[0])
            label_tmp.append(item_list[1])

with open('./dev.json', mode='w', encoding='utf8') as f:
    json.dump(dev_dataset, f, indent=4, ensure_ascii=False)
