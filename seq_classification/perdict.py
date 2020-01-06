#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/30 17:47
# @Author  : 云帆
# @Site    : 
# @File    : perdict.py
# @Software: PyCharm
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
import torch
import pandas as pd

news_type_id_dict = {'娱乐': 0, '财经': 1, '体育': 2, '家居': 3, '教育': 4, '房产': 5, '时尚': 6, '游戏': 7, '科技': 8, '时政': 9}

news_id_type_dict = {v: k for k, v in news_type_id_dict.items()}


def get_atten_mask(tokens_ids, pad_index=0):
    return list(map(lambda x: 1 if x != pad_index else 0, tokens_ids))


config = BertConfig.from_pretrained('./model/bert_config.json', num_labels=len(news_type_id_dict))
print(config)
classifier = BertForSequenceClassification.from_pretrained('./fine_tune_model/model_7', config=config)
classifier.eval()
print(classifier)

tokenizer = BertTokenizer('./model/vocab.txt')

index = 0


def predict(text):
    global index
    text = str(text).strip()
    token_ids = tokenizer.encode(ILLEGAL_CHARACTERS_RE.sub(r'', text), max_length=256,
                                 pad_to_max_length=True)
    token_mask = get_atten_mask(token_ids)

    token_segment_type = tokenizer.create_token_type_ids_from_sequences(token_ids_0=token_ids[1:-1])

    token_ids = torch.LongTensor(token_ids).unsqueeze(0)
    token_mask = torch.LongTensor(token_mask).unsqueeze(0)
    token_segment_type = torch.LongTensor(token_segment_type).unsqueeze(0)

    inputs = {
        'input_ids': token_ids,
        'token_type_ids': token_segment_type,
        'attention_mask': token_mask,
        # 'labels': batch[0]
    }
    logits = classifier(**inputs)
    _, predict = logits[0].max(1)
    # print(str(index) + news_id_type_dict[predict.item()])
    index += 1
    return news_id_type_dict[predict.item()]


if __name__ == '__main__':

    news = '''
    对于我国的科技巨头华为而言，2019年注定是不平凡的一年，由于在5G领域遥遥领先于其他国家，华为遭到了不少方面的觊觎，并因此承受了太多不公平地对待，在零部件供应、核心技术研发、以及市场等多个领域受到了有意打压。
    但是华为并没有因此而一蹶不振，而是亮出了自己的一张又一张“底牌”，随着麒麟处理器、海思半导体以及鸿蒙操作系统的闪亮登场，华为也向世界证明了自己的实力，上演了一场几乎完美的绝地反击。
    '''
    print(predict(news))

    # df = pd.read_excel('./data/cnews/test.xlsx')
    # df['predict'] = df['news_content'].apply(predict)
    # df['tag'] = df.apply(lambda x: x['news_type'] == x['predict'], axis=1)
    #
    # df.to_excel('./data/cnews/result_text.xlsx', index=False)
