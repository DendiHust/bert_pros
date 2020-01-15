#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/6 16:15
# @Author  : 云帆
# @Site    : 
# @File    : re_dataset.py
# @Software: PyCharm
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import random
import torch
from tqdm import tqdm
from src.utils import file_util



id2predict, predict2id = json.load(open(file_util.get_project_path() + './data/pro_data/all_50_schemas.json', encoding='utf8', mode='r'))
num_classes = len(id2predict)
tokenizer = BertTokenizer(file_util.get_project_path() + './bert_model/vocab.txt')


# 得到attention mask
def get_atten_mask(tokens_ids, pad_index=0):
    return list(map(lambda x: 1 if x != pad_index else 0, tokens_ids))


class RE_Dataset(Dataset):

    @classmethod
    def find_sub_list_index(cls, list1, list2):
        len_2 = len(list2)
        for i in range(len(list1)):
            if list1[i: i + len_2] == list2:
                return i
        return -1

    def __init__(self, file_path, max_length=256, device = None):
        source_data = json.load(open(file_path, mode='r', encoding='utf8'))
        # 字符索引
        token_ids_list = []
        # atten_mask
        token_atten_mask_list = []
        # seq type
        seq_type_list = []
        # subject_start_token
        s_start_token = []
        # subject_end_token
        s_end_token = []
        # subject 起始位置
        s_pos_start_index = []
        # subject 结束位置
        s_pos_end_index = []
        # object 起始位置 and p 类型索引
        o_start_token = []
        # object 结束位置 and p 类型索引
        o_end_token = []

        for data in tqdm(source_data,desc='pro data'):
            text = data['text']
            #
            token_ids = tokenizer.encode(text, max_length=max_length, pad_to_max_length=True)

            for spo in data['spo_list']:
                spo = (tokenizer.encode(spo[0])[1:-1], predict2id[spo[1]], tokenizer.encode(spo[2])[1:-1])
                # subject 起始位置
                # subject_index = self.find_sub_list_index(token_ids[1:-1], spo[0])
                subject_index = self.find_sub_list_index(token_ids, spo[0])
                # object 起始位置
                # object_index = self.find_sub_list_index(token_ids[1:-1], spo[2])
                object_index = self.find_sub_list_index(token_ids, spo[2])
                # items
                items = {}
                if subject_index != -1 and object_index != -1:
                    key = (subject_index, subject_index + len(spo[0]) - 1)
                    if key not in items:
                        items[key] = []
                    items[key].append(
                        # object 起始位置， object 结束位置， p index
                        (object_index, object_index + len(spo[2]) - 1, spo[1])
                    )

                if items:
                    subject_start_pos, subject_end_pos = np.zeros(len(token_ids)), np.zeros(len(token_ids))
                    # 将所有的subject 起始 和 结束位置 标记出来
                    for i in items:
                        subject_start_pos[i[0]] = 1
                        subject_end_pos[i[1]] = 1

                    object_start_token, object_end_token = np.zeros((len(token_ids), num_classes)), np.zeros(
                        (len(token_ids), num_classes))

                    # 随机采样一个subject
                    random_subject_key = list(items.keys())[random.randint(0, len(items) - 1)]

                    for i in items.get(random_subject_key, []):
                        object_start_token[i[0]][i[2]] = 1
                        object_end_token[i[1]][i[2]] = 1
                    # 添加到list
                    # print(data)
                    # print(token_ids)
                    # print(get_atten_mask(token_ids))
                    # print(tokenizer.create_token_type_ids_from_sequences(token_ids[1:-1]))
                    # print(subject_start_pos)
                    # print(subject_end_pos)
                    # print(random_subject_key)
                    # print(object_start_token)
                    # print(object_end_token)
                    token_ids_list.append(token_ids)
                    token_atten_mask_list.append(get_atten_mask(token_ids))
                    seq_type_list.append(tokenizer.create_token_type_ids_from_sequences(token_ids[1:-1]))

                    s_start_token.append(subject_start_pos)
                    s_end_token.append(subject_end_pos)
                    s_pos_start_index.append([random_subject_key[0]])
                    s_pos_end_index.append([random_subject_key[1]])
                    o_start_token.append(object_start_token)
                    o_end_token.append(object_end_token)

        self.TOKEN_LIST = torch.from_numpy(np.array(token_ids_list)).long()
        self.TOKEN_ATTEN_MASK_LIST = torch.from_numpy(np.array(token_atten_mask_list)).long()
        self.SEQ_TYPE_LIST = torch.from_numpy(np.array(seq_type_list)).long()

        self.S_POS_START_INDEX = torch.from_numpy(np.array(s_pos_start_index)).long()
        self.S_POS_END_INDEX = torch.from_numpy(np.array(s_pos_end_index)).long()

        self.S_START_TOKEN = torch.from_numpy(np.array(s_start_token)).float()
        self.S_END_TOKEN = torch.from_numpy(np.array(s_end_token)).float()
        self.O_START_TOKEN = torch.from_numpy(np.array(o_start_token)).float()
        self.O_END_TOKEN = torch.from_numpy(np.array(o_end_token)).float()

    def __len__(self):
        return self.TOKEN_LIST.shape[0]

    def __getitem__(self, index):
        return self.TOKEN_LIST[index], self.TOKEN_ATTEN_MASK_LIST[index], self.SEQ_TYPE_LIST[index], \
               self.S_POS_START_INDEX[index], self.S_POS_END_INDEX[index], self.S_START_TOKEN[index], self.S_END_TOKEN[index], \
               self.O_START_TOKEN[index], self.O_END_TOKEN[index]


if __name__ == '__main__':
    re_data = RE_Dataset('../../data/pro_data/train_data.json', max_length=50)
