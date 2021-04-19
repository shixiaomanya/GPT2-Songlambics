# GPT2-Songlambics
## 文件结构
* config文件夹内是预训练模型，训练中使用的是config.json
* data文件夹内是训练数据，raw_data文件夹中包含[chinese-poetry](https://github.com/chinese-poetry/chinese-poetry/tree/master/ci)中的ci.song.[0-21000].json，该文件下的test1.py用于从这些ci.song.[0-21000].json文件中提取出所有词牌名为“浣溪沙”的词，并保存在content.json中，content.json中的数据经过二次处理（去除格式错误、缺字的词，增加漏字等）后复制在processed文件夹中。processed文件夹中是处理后的数据，其中test2.py用于将content.json中的742首词处理为finals.json、sentences.json、pos.json、pingze.json四个json文件。
* test3.py用于对data/processed中的五个json文件做tokenize并划分num_piece（默认=20），并将结果保存在data/processed/tokenized中
* test4.py，本来应该命名为train.py，用于训练
* test5.py，本来应该命名为generate.py，用于生成
* model/lyrics/final_model文件夹内保存test4.py中经过训练后保存的模型，因为它们实在是太大了就都删掉了，第一次运行最终loss在4左右......
* tokenizations文件夹下保存可以选用的三种tokenizer，包括默认的Bert Tokenizer，分词版Bert Tokenizer以及BPE Tokenizer。同时也保存了vocab_guwen.txt，finals.txt，sentences.txt，pingze.txt四个语料库
