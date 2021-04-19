# test2.py，用于将二次处理后的content.json文件，进一步处理成pos.json（0 1 2 3 4 ...），sentences.json（0 0 0 0 ...），finals.json
# 以及pingze.json
# ！！！！！！！！！！！！注意！！！！！！！！！！！！！！如果要重新生成这些json文件，必须把原json文件删除！因为是追加，非覆盖写！
# 得到五个json文件后，转到根目录ci/test3.py做五个json文件的tokenize

from pypinyin import lazy_pinyin, Style
import re
punctuation = ['，', '。', '？', '！', '《', '》']


# pos.json（0 1 2 3 4 ...）
def get_pos(line):
    cn = [one for one in line]    # 浣溪沙，每首固定格式：  7字。7字。7字。7字，7字。7字。    共48位（包括符号）
    cn = cn[:-1]      # 把cn列表最后一个元素'\n'删去
    pos = ['0'] * len(cn)     # len(cn) = 48
    cnt = 0
    for i,item in enumerate(cn):
        pos[i] = str(cnt)
        cnt += 1
    pos = ' '.join(pos) + ' '
    with open('pos.json', 'a', encoding='UTF-8') as fp:     # a 表示追加
        fp.write(f'{pos}\n')


# sentences.json（0 0 0 0 ...）师姐的是每一个[SEP]算一句话，一个标记，那我就以每一个句号。算一句话？？
def get_sentences(line):
    cn = [one for one in line]
    cn = cn[:-1]
    sentences = ['0'] * len(cn)
    cnt = 0
    for i,item in enumerate(cn):
        if item == '。':
            sentences[i] = str(cnt)
            cnt += 1
        else:
            sentences[i] = str(cnt)
    sentences = ' '.join(sentences) + ' '
    with open('sentences.json', 'a', encoding='UTF-8') as fp:
        fp.write(f'{sentences}\n')


# finals.json，得到每个字的韵母，标点符号保留
def get_finals(line):
    cn = [one for one in line]
    cn = cn[:-1]
    finals = lazy_pinyin(cn,style=Style.FINALS)
    finals = ' '.join(finals) + ' '
    with open('finals.json', 'a', encoding='UTF-8') as fp:
        fp.write(f'{finals}\n')


# pingze.json，得到每个字的音调，标点符号保留
def get_pingze(line):
    cn = [one for one in line]
    cn = cn[:-1]
    pingze = lazy_pinyin(cn,style=Style.TONE3)
    for i,item in enumerate(pingze):
        if item in punctuation:
            pingze[i] = item
        else:
            temp = re.findall("\d+",item)
            if temp == []:        # 如果返回的是空列表，说明没有音调，是个轻音字
                pingze[i] = '[QING]'
            elif temp == ['1'] or temp == ['2']:
                pingze[i] = '[PING]'
            elif temp == ['3'] or temp == ['4']:
                pingze[i] = '[ZE]'
            else:
                pingze[i] = item
    pingze = ' '.join(pingze) + ' '
    with open('pingze.json', 'a', encoding='UTF-8') as fp:
        fp.write(f'{pingze}\n')


if __name__ == '__main__':
    with open('content.json',encoding='utf-8') as f:
        for line in f:
            get_pos(line)
            get_sentences(line)
            get_finals(line)
            # get_pingze(line)

