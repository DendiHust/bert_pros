#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/13 15:17
# @Author  : 云帆
# @Site    : 
# @File    : train_model.py
# @Software: PyCharm
import torch
import torch.nn as nn
from src.model.bert2re import REModel
from src.dataset.re_dataset import RE_Dataset
from transformers import BertConfig
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
import time

lr = 1e-5
batch_size = 8
max_length = 256
cuda = True



def train_func(train_dataset:RE_Dataset, model:REModel, optimizer, criterion:nn.BCELoss, batch_size):
    model.train()
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    losses = 0.0
    steps = 0
    for step, batch in tqdm(enumerate(train_dataloader)):
        # 模型输入
        token_ids = batch[0]
        token_attn_mask = batch[1]
        token_seq_type = batch[2]
        subject_start_pos = batch[3]
        subject_end_pos = batch[4]
        # label
        subject_start_label = batch[5]
        subject_end_label = batch[6]
        object_start_label = batch[7]
        object_end_label = batch[8]

        inputs = {
            'input_ids': token_ids,
            'attention_mask': token_attn_mask,
            'token_type_ids': token_seq_type,
            'subject_start_pos_index': subject_start_pos,
            'subject_end_pos_index': subject_end_pos
        }


        s_start, s_end, o_start, o_end = model(**inputs)
        # subject 损失
        subject_start_label = subject_start_label.unsqueeze(-1)
        subject_end_label = subject_end_label.unsqueeze(-1)
        mask = token_attn_mask.unsqueeze(-1)
        s_start_loss = criterion(s_start, subject_start_label)
        s_start_loss = s_start_loss * mask
        s_start_loss = torch.sum(s_start_loss) / torch.sum(mask)

        s_end_loss = criterion(s_end, subject_end_label)
        s_end_loss = s_end_loss * mask
        s_end_loss = torch.sum(s_end_loss) / torch.sum(mask)

        # subject_loss = s_start_loss +

        # object 损失
        mask = token_attn_mask.unsqueeze(-1).repeat(1, 1, 49)
        o_start_loss = criterion(o_start, object_start_label)
        o_start_loss = o_start_loss * mask
        o_start_loss = torch.sum(o_start_loss) / torch.sum(mask)


        o_end_loss = criterion(o_end, object_end_label)
        o_end_loss = o_end_loss * mask
        o_end_loss = torch.sum(o_end_loss) / torch.sum(mask)

        all_loss = s_start_loss + s_end_loss + o_start_loss + o_end_loss

        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()
        s_start = (s_start > 0.5).float()
        s_end = (s_end > 0.5).float()
        s_size = s_start.shape[0] * s_start.shape[1] * s_start.shape[2]
        s_start_correct = (s_start == subject_start_label).sum().float() / s_size
        s_end_correct = (s_end == subject_end_label).sum().float() / s_size

        o_start = (o_start > 0.5)
        o_end = (o_end > 0.5)
        o_size = o_start.shape[0] * o_start.shape[1] * o_start.shape[2]


        o_start_correct = (o_start == object_start_label).sum().float() / o_size
        o_end_correct = (o_end == object_end_label).sum() / o_size

        steps += 1
        losses += all_loss.item()





















def evaluate_func():
    pass


def epoch_time(start_time, end_time):
    delta_time = end_time - start_time
    delta_mins = int(delta_time / 60)
    delta_secs = int(delta_time - delta_mins * 60)
    return delta_mins, delta_secs







if __name__ == '__main__':

    train_dataset = RE_Dataset('./data/pro_data/train_data-sim.json')

    bert_conf = BertConfig.from_pretrained('./bert_model/bert_config.json')
    re_model = REModel(bert_conf)

    #
    bce_loss = nn.BCELoss(reduce=False)

    optimizer = torch.optim.Adam(re_model.parameters(), lr=lr)

    train_func(train_dataset, re_model, optimizer=optimizer, criterion=bce_loss, batch_size=batch_size)

















