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
from src.utils import file_util, logger
from transformers import BertConfig, BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm
import time
import numpy as np
import json
import codecs

lr = 5e-5
batch_size = 8
max_length = 160
cuda = True
epoches = 50
train_data_path = file_util.get_project_path() + './data/pro_data/train_data-sim.json'
eval_data_path = file_util.get_project_path() + './data/pro_data/dev_data.json'

tokenizer = BertTokenizer(file_util.get_project_path() + './bert_model/vocab.txt')
id2predict, predict2id = json.load(
    open(file_util.get_project_path() + './data/pro_data/all_50_schemas.json', encoding='utf8', mode='r'))


# 得到attention mask
def get_atten_mask(tokens_ids, pad_index=0):
    return list(map(lambda x: 1 if x != pad_index else 0, tokens_ids))


def extract_items(text, model: REModel, device=None):
    _token_ids = tokenizer.encode(text, max_length=max_length, pad_to_max_length=True)
    _token_attn_mask = get_atten_mask(_token_ids)
    _token_seq_type = tokenizer.create_token_type_ids_from_sequences(_token_ids[1:-1])

    token_ids = torch.from_numpy(np.array([_token_ids])).long().to(device)
    token_attn_mask = torch.from_numpy(np.array([_token_attn_mask])).long().to(device)
    token_seq_type = torch.from_numpy(np.array([_token_seq_type])).long().to(device)

    model.eval()
    inputs = {
        'input_ids': token_ids,
        'attention_mask': token_attn_mask,
        'token_type_ids': token_seq_type
    }
    s1, s2 = model(**inputs)

    s1 = s1.squeeze()
    s2 = s2.squeeze()

    s1 = s1.data.cpu().numpy()
    s2 = s2.data.cpu().numpy()

    s_start = np.where(s1 > 0.5)[0]
    s_end = np.where(s2 > 0.5)[0]
    subject = []
    for i in s_start:
        j = s_end[s_end >= i]
        if len(j) > 0:
            subject.append((i, j[0]))
    spos = []
    # print(subject)
    # if len(subject) > 3:
    #     subject = subject[:3]
    # print('len subject:{} '.format(len(subject)))
    if subject:

        s_k1, s_k2 = np.array(subject).T
        token_ids_o = np.repeat(np.array([_token_ids]), len(subject), axis=0)
        token_attn_mask_o = np.repeat(np.array([_token_attn_mask]), len(subject), axis=0)
        token_seq_type_o = np.repeat(np.array([_token_seq_type]), len(subject), axis=0)
        subject_start_o = np.repeat(np.array([s_k1]), len(subject), axis=0)
        subject_end_o = np.repeat(np.array([s_k2]), len(subject), axis=0)

        token_ids_o = torch.from_numpy(token_ids_o).long()
        token_attn_mask_o = torch.from_numpy(token_attn_mask_o).long()
        token_seq_type_o = torch.from_numpy(token_seq_type_o).long()
        subject_start_o = torch.from_numpy(subject_start_o).long()
        subject_end_o = torch.from_numpy(subject_end_o).long()

        o_data_set = TensorDataset(token_ids_o, token_attn_mask_o, token_seq_type_o, subject_start_o, subject_end_o)
        o_data_loader = DataLoader(o_data_set, batch_size=int(batch_size / 2))

        is_first = True
        o_start = None
        o_end = None
        for batch in o_data_loader:
            if cuda:
                torch.cuda.empty_cache()
            inputs = {
                'input_ids': batch[0].to(device),
                'attention_mask': batch[1].to(device),
                'token_type_ids': batch[2].to(device),
                'subject_start_pos_index': batch[3].to(device),
                'subject_end_pos_index': batch[4].to(device)
            }
            if is_first:
                _, _, o_start, o_end = model(**inputs)
                o_start = o_start.to(torch.device('cpu'))
                is_first = False
            else:
                _, _, o_tmp_start, o_tmp_end = model(**inputs)
                o_tmp_start = o_tmp_start.to(torch.device('cpu'))
                o_tmp_end = o_tmp_end.to(torch.device('cpu'))
                o_start = torch.cat((o_start, o_tmp_start), dim=0)
                o_end = torch.cat((o_end, o_tmp_end), dim=0)

        o_start = o_start.data.numpy()
        o_end = o_end.data.numpy()
        # print('o_start shape:{}'.format(o_start.shape))
        # print('o_end shape:{}'.format(o_end.shape))
        for i, s in enumerate(subject):
            if s[0] == 0 or s[1] == 0 or len(text[s[0]: s[1] + 1]) == 0:
                continue
            o1 = np.where(o_start[i] > 0.5)
            o2 = np.where(o_end[i] > 0.5)
            # print('{}:o1 len:{}'.format(i, len(o1[0])))
            # print('{}:o2 len:{}'.format(i, len(o2[0])))
            for _o1, _c1 in zip(*o1):
                if _o1 == 0:
                    continue
                for _o2, _c2 in zip(*o2):
                    if _o2 >= _o1 and _c1 == _c2:
                        spos.append(((s[0] - 1, s[1] - 1), _c1, (_o1 - 1, _o2 - 1)))
                        break

    return [SPO((text[s[0]: s[1] + 1], id2predict[str(p)], (text[o[0]: o[1] + 1]))) for s, p, o in spos]


class SPO(tuple):

    def __init__(self, spo):
        self.spo = spo

    def __hash__(self):
        return self.spo.__hash__()

    def __eq__(self, other):
        return self.spo == other


def train_func(train_dataset: RE_Dataset, model: REModel, optimizer, criterion: nn.BCELoss, batch_size=batch_size,
               device=None):
    model.train()
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    print(len(train_dataloader))
    losses = 0.0
    steps = 0
    for step, batch in tqdm(enumerate(train_dataloader), desc='train process'):
        if cuda:
            torch.cuda.empty_cache()
        # 模型输入
        token_ids = batch[0]
        token_attn_mask = batch[1]
        token_seq_type = batch[2]
        subject_start_pos = batch[3]
        subject_end_pos = batch[4]
        # label
        subject_start_label = batch[5].to(device)
        subject_end_label = batch[6].to(device)
        object_start_label = batch[7].to(device)
        object_end_label = batch[8].to(device)

        inputs = {
            'input_ids': token_ids.to(device),
            'attention_mask': token_attn_mask.to(device),
            'token_type_ids': token_seq_type.to(device),
            'subject_start_pos_index': subject_start_pos.to(device),
            'subject_end_pos_index': subject_end_pos.to(device)
        }

        s_start, s_end, o_start, o_end = model(**inputs)
        # subject 损失
        subject_start_label = subject_start_label.unsqueeze(-1)
        subject_end_label = subject_end_label.unsqueeze(-1)
        mask = token_attn_mask.unsqueeze(-1).to(device)
        s_start_loss = criterion(s_start, subject_start_label)
        s_start_loss = s_start_loss * mask
        s_start_loss = torch.sum(s_start_loss) / torch.sum(mask)

        s_end_loss = criterion(s_end, subject_end_label)
        s_end_loss = s_end_loss * mask
        s_end_loss = torch.sum(s_end_loss) / torch.sum(mask)

        # subject_loss = s_start_loss +

        # object 损失
        mask = token_attn_mask.unsqueeze(-1).repeat(1, 1, 49).to(device)
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

        #
        # s_start = (s_start > 0.5).float()
        # s_end = (s_end > 0.5).float()
        # s_size = s_start.shape[0] * s_start.shape[1] * s_start.shape[2]
        # s_start_correct = (s_start == subject_start_label).sum().float() / s_size
        # s_end_correct = (s_end == subject_end_label).sum().float() / s_size
        #
        # o_start = (o_start > 0.5)
        # o_end = (o_end > 0.5)
        # o_size = o_start.shape[0] * o_start.shape[1] * o_start.shape[2]
        #
        #
        # o_start_correct = (o_start == object_start_label).sum().float() / o_size
        # o_end_correct = (o_end == object_end_label).sum() / o_size

        steps += 1
        losses += all_loss.item()

        # extract_items(token_ids, token_attn_mask, token_seq_type, model)

    return losses / steps


def evaluate_func(eval_data_file_path, model: REModel, device=None):
    eval_data = json.load(open(eval_data_file_path, mode='r', encoding='utf8'))
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = codecs.open(file_util.get_project_path() + './dev_pred.json', 'w', encoding='utf-8')
    f1, precision, recall = 0.0, 0.0, 0.0
    pbar = tqdm()
    result_list = []
    for data in tqdm(eval_data, desc='eval data'):
        spo_pre_list = extract_items(data['text'], model, device)
        # print('pre:{}'.format(spo_pre_list))
        spo_list = [SPO((item[0], item[1], item[2])) for item in data['spo_list']]
        # print('or:{}'.format(spo_list))
        R = set(spo_pre_list)
        T = set(spo_list)

        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description('f1: %.5f, precision: %.5f, recall: %.5f' %
                             (f1, precision, recall))
        s = {
            'text': data['text'],
            'spo_list': list(T),
            'spo_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        }
        result_list.append(s)
        # f.write(s + '\n')
    pbar.close()
    json.dump(result_list, f, indent=4, ensure_ascii=False)
    f.close()
    # print(X)
    return f1, precision, recall


def epoch_time(start_time, end_time):
    delta_time = end_time - start_time
    delta_mins = int(delta_time / 60)
    delta_secs = int(delta_time - delta_mins * 60)
    return delta_mins, delta_secs


if __name__ == '__main__':

    # t1 = set([SPO(('喜剧之王', '主演', '周星驰'))])
    # t2 = set([SPO(('好', '董事长', '我修养》《喜剧')),SPO(('喜剧之王', '主演', '周星驰'))])
    # print(t1 & t2)

    device = torch.device('cuda' if cuda else 'cpu')
    train_dataset = RE_Dataset(train_data_path, max_length=max_length, device=device)
    bert_conf = BertConfig.from_pretrained('./bert_model/bert_config.json')
    re_model = REModel(bert_conf, device=device).to(device)

    bce_loss = nn.BCELoss(reduce=False)

    optimizer = torch.optim.Adam(re_model.parameters(), lr=lr)

    e = 0
    max_f1 = 1e-5
    while e < epoches + 1:
        e += 1
        start_time = time.time()
        train_losses = train_func(train_dataset, re_model, optimizer, bce_loss, batch_size, device=device)
        f1, precision, recall = evaluate_func(eval_data_path, re_model, device=device)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if f1 > max_f1:
            max_f1 = f1
            torch.save(re_model.state_dict(), file_util.get_project_path() + './models/re_model_{}'.format(e))

        logger.info('Epoch: {:02} | Time: {}m {}s'.format(e, epoch_mins, epoch_secs))
        logger.info(
            '\tTrain Loss: {:.6f} | Eval F1: {:.6f} | Eval Pre: {:.6f} | Eval Cal: {:.6f}'.format(train_losses, f1,
                                                                                                  precision, recall))

    # train_func(train_dataset, re_model, optimizer=optimizer, criterion=bce_loss, batch_size=batch_size)

    # evaluate_func('./data/pro_data/train_data-sim.json', re_model)
