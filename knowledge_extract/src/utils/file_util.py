#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/9 17:52
# @Author  : 云帆
# @Site    : 
# @File    : file_util.py
# @Software: PyCharm
# 文件工具
import json
import os


def get_project_path():
    '''
    获取工程的路径
    :return:
    '''
    return os.path.join(os.path.dirname(__file__), '../../')