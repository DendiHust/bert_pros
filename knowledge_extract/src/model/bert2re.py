#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/6 16:08
# @Author  : 云帆
# @Site    : 
# @File    : bert2re.py
# @Software: PyCharm
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
import numpy as np
from src.utils import file_util


class REModel(nn.Module):

    def __batch_gather(self, data:torch.Tensor, index:torch.Tensor):
        length = index.shape[0]
        t_index = index.data.numpy()
        t_data = data.data.numpy()
        result = []
        for i in range(length):
            result.append(t_data[i, t_index[i], :])

        return torch.from_numpy(np.array(result))


    def __init__(self, bert_conf: BertConfig):
        super(REModel, self).__init__()
        self.bert = BertModel.from_pretrained(file_util.get_project_path() + './bert_model/pytorch_model.bin', config=bert_conf)
        # subject 开始位置
        self.subject_start_cls = nn.Sequential(
            nn.Dropout(bert_conf.hidden_dropout_prob),
            nn.Linear(bert_conf.hidden_size, 1),
            nn.Sigmoid()
        )

        # subject 结束位置
        self.subject_end_cls = nn.Sequential(
            nn.Dropout(bert_conf.hidden_dropout_prob),
            nn.Linear(bert_conf.hidden_size, 1),
            nn.Sigmoid()
        )

        # object-predicate 开始位置
        self.object_start_cls = nn.Sequential(
            nn.Dropout(bert_conf.hidden_dropout_prob),
            nn.Linear(bert_conf.hidden_size, bert_conf.num_labels),
            nn.Sigmoid()
        )

        # object-predicate 结束位置
        self.object_end_cls = nn.Sequential(
            nn.Dropout(bert_conf.hidden_dropout_prob),
            nn.Linear(bert_conf.hidden_size, bert_conf.num_labels),
            nn.Sigmoid()
        )

    def forward(self, subject_start_pos_index=None, subject_end_pos_index=None, input_ids=None, attention_mask=None,
                token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        bert_output = outputs[0]
        #
        s1 = self.subject_start_cls(bert_output)
        s2 = self.subject_end_cls(bert_output)
        # 获得k1，k2
        s_k1 = self.__batch_gather(bert_output, subject_start_pos_index)
        s_k2 = self.__batch_gather(bert_output, subject_end_pos_index)

        k_v = torch.cat([s_k1, s_k2], dim=1)
        k_v = k_v.mean(dim=1).unsqueeze(1).repeat(1, bert_output.shape[1], 1)

        bert_output = bert_output + k_v
        # 获得 o1, o2
        o1 = self.object_start_cls(bert_output)
        o2 = self.object_end_cls(bert_output)

        # s_k1 = torch.gather(outputs[0], 1, subject_start_pos_index)
        # s_k2 = torch.gather(outputs[0], 1, subject_end_pos_index)

        #

        return s1, s2, o1, o2




if __name__ == '__main__':
    from torch.utils.data import DataLoader, RandomSampler
    from src.dataset.re_dataset import RE_Dataset
    from tqdm import tqdm

    re_data = RE_Dataset('../../data/pro_data/train_data-sim.json', max_length=50)
    train_sampler = RandomSampler(re_data)
    train_loader = DataLoader(re_data, sampler=train_sampler, batch_size=5)
    bert_conf = BertConfig.from_pretrained('../../bert_model/bert_config.json')
    re_model = REModel(bert_conf)
    for step, batch in tqdm(enumerate(train_loader)):
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
            'subject_start_pos_index': batch[3],
            'subject_end_pos_index': batch[4]
        }
        outputs = re_model(**inputs)









