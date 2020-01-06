#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/26 10:11
# @Author  : 云帆
# @Site    : 
# @File    : train.py
# @Software: PyCharm
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler
import logger
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
import numpy as np
from transformers import AdamW
import time
from tqdm import tqdm

# batch_size
batch_size = 8
# 学习率
lr = 1e-5
# 是否使用gpu
cuda = False
# 训练批次
epoches = 20
# sequence 最大长度
max_length = 256


# 得到attention mask
def get_atten_mask(tokens_ids, pad_index=0):
    return list(map(lambda x: 1 if x != pad_index else 0, tokens_ids))


# 类别: id
news_type2id_dict = {'娱乐': 0, '财经': 1, '体育': 2, '家居': 3, '教育': 4, '房产': 5, '时尚': 6, '游戏': 7, '科技': 8, '时政': 9}


class NewsDataset(Dataset):

    def __init__(self, file_path, tokenizer: BertTokenizer, max_length=512, device=None):
        news_type = []
        news_content = []
        news_atten_mask = []
        seq_typ_ids = []
        with open(file_path, mode='r', encoding='utf8') as f:
            for line in tqdm(f.readlines()):
                line = line.strip()
                line = line.split('\t')

                news_type.append(news_type2id_dict[line[0]])
                token_ids = tokenizer.encode(ILLEGAL_CHARACTERS_RE.sub(r'', line[1]), max_length=max_length,
                                             pad_to_max_length=True)
                news_content.append(token_ids)
                news_atten_mask.append(get_atten_mask(token_ids))
                seq_typ_ids.append(tokenizer.create_token_type_ids_from_sequences(token_ids_0=token_ids[1:-1]))

        self.label = torch.from_numpy(np.array(news_type)).unsqueeze(1).long()
        self.token_ids = torch.from_numpy(np.array(news_content)).long()
        self.seq_type_ids = torch.from_numpy(np.array(seq_typ_ids)).long()
        self.atten_masks = torch.from_numpy(np.array(news_atten_mask)).long()
        if device is not None:
            self.label = self.label.to(device)
            self.token_ids = self.token_ids.to(device)
            self.seq_type_ids = self.seq_type_ids.to(device)
            self.atten_masks = self.atten_masks.to(device)

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, item):
        return self.label[item], self.token_ids[item], self.seq_type_ids[item], self.atten_masks[item]


def train(train_dataset, model: BertForSequenceClassification, optimizer: AdamW, batch_size=batch_size):
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    model.train()
    tr_loss = 0.0
    tr_acc = 0
    global_step = 0
    if cuda:
        torch.cuda.empty_cache()
    for step, batch in tqdm(enumerate(train_loader)):
        # print(step)
        inputs = {
            'input_ids': batch[1],
            'token_type_ids': batch[2],
            'attention_mask': batch[3],
            'labels': batch[0]
        }
        outputs = model(**inputs)
        loss = outputs[0]
        # print(loss)
        logits = outputs[1]

        tr_loss += loss.item()

        model.zero_grad()
        loss.backward()
        optimizer.step()
        # 计算准确率
        _, pred = logits.max(1)
        number_corr = (pred == batch[0].view(-1)).long().sum().item()
        tr_acc += number_corr
        global_step += 1

    return tr_loss / global_step, tr_acc / len(train_dataset)


def evalate(eval_dataset, model: BertForSequenceClassification, batch_size=batch_size):
    model.eval()
    eval_sampler = RandomSampler(eval_dataset)
    eval_loader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)
    tr_acc = 0
    if cuda:
        torch.cuda.empty_cache()
    for step, batch in tqdm(enumerate(eval_loader)):
        inputs = {
            'input_ids': batch[1],
            'token_type_ids': batch[2],
            'attention_mask': batch[3],
            'labels': batch[0]
        }
        outputs = model(**inputs)
        # loss = outputs[0]
        logits = outputs[1]

        # tr_loss += loss.item()

        # 计算准确率
        _, pred = logits.max(1)
        number_corr = (pred == batch[0].view(-1)).long().sum().item()
        tr_acc += number_corr

    return tr_acc / len(eval_dataset)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    device = torch.device('cuda' if cuda else 'cpu')
    # 创建config
    config = BertConfig.from_pretrained('./model/bert_config.json', num_labels=len(news_type2id_dict))
    # 创建分类器
    classifier = BertForSequenceClassification.from_pretrained('./model/pytorch_model.bin', config=config).to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in classifier.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in classifier.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # 创建tokenizer
    tokenizer = BertTokenizer('./model/vocab.txt')

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)

    logger.info('create train dataset')

    train_dataset = NewsDataset('./data/cnews/cnews.train.txt', tokenizer, max_length=max_length,
                                device=device)
    logger.info('create eval dataset')
    eval_dataset = NewsDataset('./data/cnews/cnews.val.txt', tokenizer, max_length=max_length,
                               device=device)
    best_val_acc = 0.0
    for e in range(1, epoches):
        start_time = time.time()
        train_loss, train_acc = train(train_dataset, classifier, optimizer, batch_size)
        eval_acc = evalate(eval_dataset, classifier, batch_size)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        logger.info('Epoch: {:02} | Time: {}m {}s'.format(e, epoch_mins, epoch_secs))
        logger.info(
            'Train Loss: {:.6f} | Train Acc: {:.6f} | Eval Acc: {:.6f}'.format(train_loss, train_acc, eval_acc))
        if eval_acc > best_val_acc:
            best_val_acc = eval_acc
            torch.save(classifier.state_dict(), './fine_tune_model/model_{}'.format(e))
