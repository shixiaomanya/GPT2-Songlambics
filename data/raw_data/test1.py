# test1.py，作为数据处理第一步，用于将raw_data中ci.song.0.json到ci.song.22000.json中所有词牌名'rhythmic'为“浣溪沙”的'paragraphs'提取出来
# 并保存在当前目录下的content.json文件中
# 注意！！！！！！！！不要轻易运行此程序！！！因为content.json中的文件是二次处理过的，再次运行会被覆盖！！！！
# 二次处理包括：删除含识别不了的字（表现为正方形）的词、删除缺句的词、增加漏字，删除12首“浣溪沙”的变体，content.json现剩742首
# 将content.json复制到data/processed/中，运行test2.py做下一步处理

import json

train_data = []
for i in range(0,23000,1000):
    raw_data_path = f'ci.song.{i}.json'
    with open(raw_data_path, 'r',encoding='UTF-8') as f:
         raw_data = json.load(f)
    for everydict in raw_data:
        if everydict['rhythmic'] == '浣溪沙':
            train_data.append(everydict["paragraphs"])
print(len(train_data))        # 一共776首浣溪沙，删除错误字后，剩754首，其中12首变体

with open('content.json', 'w', encoding='UTF-8') as f:
    for everylist in train_data:
        line = ''.join(everylist)
        f.write(f'{line}\n')
# 如果删去line = ''.join(everylist)，直接for line in train_data:，则得到new_train_data.json