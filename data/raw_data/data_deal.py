#!-*- coding: utf-8 -*-
# 这里的内容暂时不用，直接从test1.py开始

import json

# # 获取所有词牌名
# cipai = set()    # 创建一个set（集合），set中不允许插入相同元素
# for i in range(0,23000,1000):
#     raw_data_path = f'ci.song.{i}.json'
#     with open(raw_data_path, 'r',encoding='utf8') as f:
#          raw_data = json.load(f)
#     for everydict in raw_data:
#         cipai.add(everydict['rhythmic'])
# print('共有词牌名：',len(cipai))    # 共有词牌名： 1421
#
# # 写入 JSON 数据
# with open('cipai.json', 'w',encoding='utf-8') as f:    # 在代码当前目录生成一个cipai.json的文件
#      json.dump(list(cipai), f,ensure_ascii=False)     # 写入json文件，要先转为list

# 获取指定词牌名的诗词。并生成raw_train_data.json文件
train_data = []
for i in range(0,23000,1000):
    raw_data_path = f'ci.song.{i}.json'
    with open(raw_data_path, 'r',encoding='UTF-8') as f:
         raw_data = json.load(f)
    for everydict in raw_data:
        if everydict['rhythmic'] == '浣溪沙':
            train_data.append(everydict["paragraphs"])
print(len(train_data))        # 一共776首浣溪沙

with open('raw_train_data.json', 'w', encoding='UTF-8') as f:
    json.dump(train_data, f, ensure_ascii=False)
















