#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/3 14:19
# @Author  : 云帆
# @Site    : 
# @File    : test.py
# @Software: PyCharm
import json
import re
import numpy as np
from tqdm import tqdm
import ahocorasick
from random import choice
import pyhanlp

mode = 0
char_size = 128
maxlen = 512

train_data = json.load(open('./pro_data/train_data.json', encoding='utf8'))
dev_data = json.load(open('./pro_data/dev_data.json', encoding='utf8'))
id2predicate, predicate2id = json.load(open('./pro_data/all_50_schemas.json', encoding='utf8'))
id2predicate = {int(i): j for i, j in id2predicate.items()}
id2char, char2id = json.load(open('./pro_data/all_chars.json', encoding='utf8'))
num_classes = len(id2predicate)

predicates = {}  # 格式：{predicate: [(subject, predicate, object)]}



def tokenize(s):
    return [i.word for i in pyhanlp.HanLP.segment(s)]


def sent2vec(S):
    """S格式：[[w1, w2]]
    """
    V = []
    # for s in S:
    #     V.append([])
    #     for w in s:
    #         for _ in w:
    #             V[-1].append(word2id.get(w, 0))
    # V = seq_padding(V)
    # V = word2vec[V]
    return V

def repair(d):
    d['text'] = d['text'].lower()
    something = re.findall('《([^《》]*?)》', d['text'])
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


for d in train_data:
    repair(d)
    for sp in d['spo_list']:
        if sp[1] not in predicates:
            predicates[sp[1]] = []
        predicates[sp[1]].append(sp)

for d in dev_data:
    repair(d)


# 随机采样
def random_generate(d, spo_list_key):
    r = np.random.random()
    if r > 0.5:
        return d
    else:
        # 随机在远spo_list中找到一个spo
        k = np.random.randint(len(d[spo_list_key]))
        spi = d[spo_list_key][k]
        # 在统计的所有predicates中随机选择一个同类的 spo
        k = np.random.randint(len(predicates[spi[1]]))
        spo = predicates[spi[1]][k]
        # 将 d 中的 s 换成同类型p 随机的 s，将 d 中的 o 换成同类型p 随机的 o
        F = lambda s: s.replace(spi[0], spo[0]).replace(spi[2], spo[2])
        text = F(d['text'])
        spo_list = [(F(sp[0]), sp[1], F(sp[2])) for sp in d[spo_list_key]]
        return {'text': text, spo_list_key: spo_list}


class AC_Unicode:
    """稍微封装一下，弄个支持unicode的AC自动机
    """

    def __init__(self):
        self.ac = ahocorasick.Automaton()

    def add_word(self, k, v):
        # k = k.encode('utf-8')
        return self.ac.add_word(k, v)

    def make_automaton(self):
        return self.ac.make_automaton()

    def iter(self, s):
        # s = s.encode('utf-8')
        return self.ac.iter(s)


class spo_searcher:
    def __init__(self, train_data):
        self.s_ac = AC_Unicode()
        self.o_ac = AC_Unicode()
        self.sp2o = {}
        self.spo_total = {}
        for i, d in tqdm(enumerate(train_data), desc=u'构建三元组搜索器'):
            for s, p, o in d['spo_list']:
                self.s_ac.add_word(s, s)
                self.o_ac.add_word(o, o)
                if (s, o) not in self.sp2o:
                    self.sp2o[(s, o)] = set()
                if (s, p, o) not in self.spo_total:
                    self.spo_total[(s, p, o)] = set()
                self.sp2o[(s, o)].add(p)
                self.spo_total[(s, p, o)].add(i)
        self.s_ac.make_automaton()
        self.o_ac.make_automaton()

    def extract_items(self, text_in, text_idx=None):
        R = set()
        for s in self.s_ac.iter(text_in):
            for o in self.o_ac.iter(text_in):
                if (s[1], o[1]) in self.sp2o:
                    for p in self.sp2o[(s[1], o[1])]:
                        if text_idx is None:
                            R.add((s[1], p, o[1]))
                        elif (self.spo_total[(s[1], p, o[1])] - set([text_idx])):
                            R.add((s[1], p, o[1]))
        return list(R)


spoer = spo_searcher(train_data)
# print(spoer)

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

class data_generator:
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            # T1    字索引
            # T2    词索引
            # S1    随机采样后的所有 subject 起始位置，1表示该位置开始，0表示其他
            # S2    随机采样后的所有 subject 结束位置，1表示该位置结束，0表示其他
            # K1    随机选择一个 subject 的起始位置
            # K2    随机选择一个 subject 的结束位置
            # O1    随机采样后的 object 起始位置和p索引，横轴1表示object起始位置，纵轴1表示p索引，0表示其他
            # O2    随机采样后的 object 结束位置和p索引，横轴1表示object起始位置，纵轴1表示p索引，0表示其他
            # PRES  源数据中的 subject 起始和结束位置，shape：len_text, 2
            # PREO  源数据中的 object 起始和结束位置，shape： len_text, num_class * 2

            T1, T2, S1, S2, K1, K2, O1, O2, PRES, PREO = [], [], [], [], [], [], [], [], [], []
            for i in idxs:
                spo_list_key = 'spo_list'  # if np.random.random() > 0.5 else 'spo_list_with_pred'
                d = random_generate(self.data[i], spo_list_key)
                text = d['text'][:maxlen]
                text_words = tokenize(text)
                text = ''.join(text_words)
                # (s_start_index, s_end_index):[(o_start_index, o_end_index, p_id)]
                items = {}
                for sp in d[spo_list_key]:
                    subjectid = text.find(sp[0])
                    objectid = text.find(sp[2])
                    if subjectid != -1 and objectid != -1:
                        key = (subjectid, subjectid + len(sp[0]))
                        if key not in items:
                            items[key] = []
                        items[key].append((objectid,
                                           objectid + len(sp[2]),
                                           predicate2id[sp[1]]))
                pre_items = {}
                for sp in spoer.extract_items(text, i):
                    subjectid = text.find(sp[0])
                    objectid = text.find(sp[2])
                    if subjectid != -1 and objectid != -1:
                        key = (subjectid, subjectid + len(sp[0]))
                        if key not in pre_items:
                            pre_items[key] = []
                        pre_items[key].append((objectid,
                                               objectid + len(sp[2]),
                                               predicate2id[sp[1]]))
                if items:
                    T1.append([char2id.get(c, 1) for c in text])  # 1是unk，0是padding
                    T2.append(text_words)
                    s1, s2 = np.zeros(len(text)), np.zeros(len(text))
                    for j in items:
                        s1[j[0]] = 1
                        s2[j[1] - 1] = 1
                    pres = np.zeros((len(text), 2))
                    for j in pre_items:
                        pres[j[0], 0] = 1
                        pres[j[1] - 1, 1] = 1
                    # k1, k2 = np.array(items.keys()).T
                    k1, k2 = np.array(list(items.keys())).T
                    k1 = choice(k1)
                    k2 = choice(k2[k2 >= k1])
                    o1, o2 = np.zeros((len(text), num_classes)), np.zeros((len(text), num_classes))
                    for j in items.get((k1, k2), []):
                        o1[j[0], j[2]] = 1
                        o2[j[1] - 1, j[2]] = 1
                    preo = np.zeros((len(text), num_classes, 2))
                    for j in pre_items.get((k1, k2), []):
                        preo[j[0], j[2], 0] = 1
                        preo[j[1] - 1, j[2], 1] = 1
                    preo = preo.reshape((len(text), -1))
                    S1.append(s1)
                    S2.append(s2)
                    K1.append([k1])
                    K2.append([k2 - 1])
                    O1.append(o1)
                    O2.append(o2)
                    PRES.append(pres)
                    PREO.append(preo)
                    if len(T1) == self.batch_size or i == idxs[-1]:
                        T1 = seq_padding(T1)
                        T2 = sent2vec(T2)
                        S1 = seq_padding(S1)
                        S2 = seq_padding(S2)
                        O1 = seq_padding(O1, np.zeros(num_classes))
                        O2 = seq_padding(O2, np.zeros(num_classes))
                        K1, K2 = np.array(K1), np.array(K2)
                        PRES = seq_padding(PRES, np.zeros(2))
                        PREO = seq_padding(PREO, np.zeros(num_classes * 2))
                        yield [T1, T2, S1, S2, K1, K2, O1, O2, PRES, PREO], None
                        T1, T2, S1, S2, K1, K2, O1, O2, PRES, PREO = [], [], [], [], [], [], [], [], [], []


train_D = data_generator(train_data)


for item in iter(train_D):
    print(item)

