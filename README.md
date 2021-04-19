# GPT2-Songlambics
## 文件结构
* config文件夹内是预训练模型，训练中使用的是config.json
* data文件夹内是训练数据，raw_data文件夹中包含[chinese-poetry](https://github.com/chinese-poetry/chinese-poetry/tree/master/ci)中的ci.song.[0-21000].json，该文件下的test1.py用于从这些ci.song.[0-21000].json文件中提取出所有词牌名为“浣溪沙”的词，并保存在content.json中，content.json中的数据经过二次处理（去除格式错误、缺字的词，增加漏字等）后复制在processed文件夹中。processed文件夹中是处理后的数据，其中test2.py用于将content.json中的742首词处理为finals.json、sentences.json、pos.json、pingze.json四个json文件。
* test3.py用于对data/processed中的五个json文件做tokenize并划分num_piece（默认=20），并将结果保存在data/processed/tokenized中
* test4.py，本来应该命名为train.py，用于训练
* test5.py，本来应该命名为generate.py，用于生成
* model/lyrics/final_model文件夹内保存test4.py中经过训练后保存的模型，因为它们实在是太大了就都删掉了，第一次运行最终loss在4左右......
* tokenizations文件夹下保存可以选用的三种tokenizer，包括默认的Bert Tokenizer，分词版Bert Tokenizer以及BPE Tokenizer。同时也保存了vocab_guwen.txt，finals.txt，sentences.txt，pingze.txt四个语料库

## Possible Solution & Some Tips
1. Take each peom/songci as an input. 每一条输入就只是一首诗或词。也就是说，不需要把许多诗词什么的拼接成一整个训练语料，然后分成pieces输入。
2. Padding each input to same length. 把每个输入，也就是每首诗词padding到同一个长度。举个例子，假设每个输入的长度都是32:
``` bash
原文：锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。
输入：锄禾日当午[SEP]汗滴禾下土[SEP]谁知盘中餐[SEP]粒粒皆辛苦[SEP][PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD]
```
3. Take all punctuations as [SEP]. 如2所示，标点就用[SEP]表示了，表示句子间的分割。
4. Input format in details:
``` bash
原文：
临江仙  
夜饮东坡醒复醉，归来仿佛三更。
家童鼻息已雷鸣。敲门都不应，倚杖听江声。
长恨此身非我有，何时忘却营营？
夜阑风静縠纹平。小舟从此逝，江海寄余生。

输入：[PAI]临江仙[CI]夜饮东坡醒复醉[SEP]归来仿佛三更[SEP]家童鼻息已雷鸣[SEP]敲门都不应[SEP]倚杖听江声[SEP]长恨此身非我有[SEP]何时忘却营营[SEP]夜阑风静縠纹平[SEP]小舟从此逝[SEP]江海寄余生[SEP]
```
然后这里，position embedding可以从[CI]后面才算。这里加的特殊token[PAI]表示这里接下来是词牌名，[CI]表示这里接下来是词的内容。你也可以用别的方式表示。这里就是举个例子。

5. 统计一下一共有多少首词，如果只是词的生成的话。我感受一下这个语料库的量级。
6. 如果语料不够，后期可以用别的诗词来pretrain一下，然后再用词的来fine-tune。

## Github 维护:
<kbd>master<\kdb>分支是我建的，不用管它。
```bash
# 如果要从github把新内容同步到自己本地
git pull origin main
    
# 如果要王github上更新本地内容上去 (更新前记得先pull)
git push -u origin main
```
