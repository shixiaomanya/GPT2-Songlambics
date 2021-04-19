# test3.py，用于对data/processed/里的五个json做tokenize，并将得到的五个json文件保存在data/processed/tokenized/里
# 参数：num_pieces = 20
import os
from tqdm.auto import tqdm

from tokenizations import tokenization_bert  # 一个字一个字的分，用的是这种


def read_ci():
    contents = []    # 对应content.json
    finals = []    # 对应finals.json
    sentences = []    # 对应sentences.json
    pos = []    # 对应pos.json
    pingze = []    # 对应pingze.json

    if os.path.exists('data/processed/content.json') and os.path.exists('data/processed/finals.json') and \
            os.path.exists('data/processed/pos.json') and os.path.exists('data/processed/pos.json') and \
            os.path.exists('data/processed/pingze.json'):
        with open('data/processed/content.json', encoding='utf8') as fp:
            for l in fp:
                contents.append(l)
        with open('data/processed/finals.json', encoding='utf8') as fp:
            for l in fp:
                finals.append(l)
        with open('data/processed/sentences.json', encoding='utf8') as fp:
            for l in fp:
                sentences.append(l)
        with open('data/processed/pos.json', encoding='utf8') as fp:
            for l in fp:
                pos.append(l)
        with open('data/processed/pingze.json', encoding='utf8') as fp:
            for l in fp:
                pingze.append(l)
        return contents, finals, sentences, pos, pingze
    else:
        print('Please try test2.py first')


def build_files(num_pieces,
                min_length,
                lines=None,
                finals=None,
                sentences=None,
                pos=None,
                pingze=None,
                tokenized_data_path=None,
                finalized_data_path=None,
                sentenced_data_path=None,
                posed_data_path=None,
                pingzed_data_path=None,
                full_tokenizer=None,
                full_finalizer=None,
                full_sentencer=None,
                full_poser=None,
                full_pingzer=None,
                enable_final=False,
                enable_sentence=False,
                enable_pos=False,
                enable_pingze=False):
    print('Start tokenizing..')
    path = tokenized_data_path.rsplit('/', 1)[0]
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    if enable_final:
        if not os.path.exists(finalized_data_path):
            os.mkdir(finalized_data_path)
    if enable_sentence:
        if not os.path.exists(sentenced_data_path):
            os.mkdir(sentenced_data_path)
    if enable_pos:
        if not os.path.exists(posed_data_path):
            os.mkdir(posed_data_path)
    if enable_pingze:
        if not os.path.exists(pingzed_data_path):
            os.mkdir(pingzed_data_path)

    all_len = len(lines)
    for k in tqdm(range(num_pieces)):
        sublines = lines[all_len // num_pieces * k: all_len // num_pieces * (k + 1)]
        if k == num_pieces - 1:
            sublines.extend(lines[all_len // num_pieces * (k + 1):])  # 把尾部例子添加到最后一个piece
        if enable_final:
            subfinals = finals[all_len // num_pieces * k: all_len // num_pieces * (k + 1)]
            if k == num_pieces - 1:
                subfinals.extend(finals[all_len // num_pieces * (k + 1):])  # 把尾部例子添加到最后一个piece
        if enable_sentence:
            subsentences = sentences[all_len // num_pieces * k: all_len // num_pieces * (k + 1)]
            if k == num_pieces - 1:
                subsentences.extend(sentences[all_len // num_pieces * (k + 1):])  # 把尾部例子添加到最后一个piece
        if enable_pos:
            subpos = pos[all_len // num_pieces * k: all_len // num_pieces * (k + 1)]
            if k == num_pieces - 1:
                subpos.extend(pos[all_len // num_pieces * (k + 1):])  # 把尾部例子添加到最后一个piece
        if enable_pingze:
            subpingze = pingze[all_len // num_pieces * k: all_len // num_pieces * (k + 1)]
            if k == num_pieces - 1:
                subpingze.extend(pingze[all_len // num_pieces * (k + 1):])  # 把尾部例子添加到最后一个piece

        for i in range(len(sublines)):
            line = sublines[i]
            if len(line) > min_length:
                line = full_tokenizer.tokenize(line)
                line = full_tokenizer.convert_tokens_to_ids(line)
                if enable_final:
                    final = subfinals[i]
                    final = full_finalizer.tokenize(final)
                    final = full_finalizer.convert_tokens_to_ids(final)
                    assert len(final) == len(line)
                if enable_sentence:
                    sentence = subsentences[i]
                    sentence = full_sentencer.tokenize(sentence)
                    sentence = full_sentencer.convert_tokens_to_ids(sentence)
                    assert len(sentence) == len(line)
                if enable_pos:
                    p = subpos[i]
                    p = full_poser.tokenize(p)
                    p = full_poser.convert_tokens_to_ids(p)
                    assert len(p) == len(line)
                if enable_pingze:
                    beat = subpingze[i]
                    beat = full_pingzer.tokenize(beat)
                    beat = full_pingzer.convert_tokens_to_ids(beat)
                    assert len(beat) == len(line)

                sublines[i] = line
                if enable_final:
                    subfinals[i] = final
                if enable_sentence:
                    subsentences[i] = sentence
                if enable_pos:
                    subpos[i] = p
                if enable_pingze:
                    subpingze[i] = beat

        full_line, full_final, full_sentence, full_pos, full_pingze = [], [], [], [], []
        for i in range(len(sublines)):
            mask = full_tokenizer.convert_tokens_to_ids('[MASK]')
            clss = full_tokenizer.convert_tokens_to_ids('[CLS]')
            full_line.append(mask)  # 文章开头添加MASK表示文章开始
            full_line.extend(sublines[i])
            full_line.append(clss)  # 文章之间添加CLS表示文章结束
            if enable_final:
                mask = full_finalizer.convert_tokens_to_ids('[MASK]')
                clss = full_finalizer.convert_tokens_to_ids('[CLS]')
                full_final.append(mask)  # 文章开头添加MASK表示文章开始
                full_final.extend(subfinals[i])
                full_final.append(clss)  # 文章之间添加CLS表示文章结束
            if enable_sentence:
                mask = full_sentencer.convert_tokens_to_ids('[MASK]')
                clss = full_sentencer.convert_tokens_to_ids('[CLS]')
                full_sentence.append(mask)  # 文章开头添加MASK表示文章开始
                full_sentence.extend(subsentences[i])
                full_sentence.append(clss)  # 文章之间添加CLS表示文章结束
            if enable_pos:
                mask = full_poser.convert_tokens_to_ids('[MASK]')
                clss = full_poser.convert_tokens_to_ids('[CLS]')
                full_pos.append(mask)  # 文章开头添加MASK表示文章开始
                full_pos.extend(subpos[i])
                full_pos.append(clss)  # 文章之间添加CLS表示文章结束
            if enable_pingze:
                mask = full_pingzer.convert_tokens_to_ids('[MASK]')
                clss = full_pingzer.convert_tokens_to_ids('[CLS]')
                full_pingze.append(mask)  # 文章开头添加MASK表示文章开始
                full_pingze.extend(subpingze[i])
                full_pingze.append(clss)  # 文章之间添加CLS表示文章结束

        if enable_final:
            assert len(full_line) == len(full_final), f'line: {len(full_line)}, final: {len(full_final)}'
        if enable_sentence:
            assert len(full_line) == len(full_sentence), f'line: {len(full_line)}, sentence: {len(full_sentence)}'
        if enable_pos:
            assert len(full_line) == len(full_pos), f'line: {len(full_line)}, pos: {len(full_pos)}'
        if enable_pingze:
            assert len(full_line) == len(full_pingze), f'line: {len(full_line)}, pingze: {len(full_pingze)}'

        with open(os.path.join(tokenized_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
            for idx in full_line:
                f.write(str(idx) + ' ')
        if enable_final:
            with open(os.path.join(finalized_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
                for idx in full_final:
                    f.write(str(idx) + ' ')
        if enable_sentence:
            with open(os.path.join(sentenced_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
                for idx in full_sentence:
                    f.write(str(idx) + ' ')
        if enable_pos:
            with open(os.path.join(posed_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
                for idx in full_pos:
                    f.write(str(idx) + ' ')
        if enable_pingze:
            with open(os.path.join(pingzed_data_path, 'tokenized_train_{}.txt'.format(k)), 'w') as f:
                for idx in full_pingze:
                    f.write(str(idx) + ' ')
    print('finish')


if __name__ == '__main__':
    num_pieces = 20
    # set tokenizer
    full_tokenizer = tokenization_bert.BertTokenizer(
        vocab_file='tokenizations/vocab_guwen.txt',  # 共有32044行，包含古诗词中许多生僻字
        do_lower_case=False  # 不要小写化处理
    )
    full_finalizer = tokenization_bert.BertTokenizer(
        vocab_file='tokenizations/finals.txt',  # 韵母库，在后面增加了，。等标点符号
        tokenize_chinese_chars=False,  # 不在中文字之间添加空格？这个文件里又没有中文字
        do_lower_case=False
    )
    full_sentencer = tokenization_bert.BertTokenizer(
        vocab_file='tokenizations/sentences.txt',  # 句子句库（只有数字）
        tokenize_chinese_chars=False,
        do_lower_case=False
    )
    full_poser = tokenization_bert.BertTokenizer(
        vocab_file='tokenizations/sentences.txt',  # 选择相对位置词库，和句子句库一样
        tokenize_chinese_chars=False,
        do_lower_case=False
    )
    full_pingzer = tokenization_bert.BertTokenizer(
        vocab_file='tokenizations/pingze.txt',  # 选择平仄词库
        tokenize_chinese_chars=False,
        do_lower_case=False
    )

    print('Loading processed data for training...')
    train_contents, train_finals, train_sentences, train_pos, train_pingze = read_ci()

    print('Tokenizing processed data for training...')
    build_files(num_pieces=num_pieces,    # 将训练语料分成多少份
                min_length=0,
                lines=train_contents,
                finals=train_finals,
                sentences=train_sentences,
                pos=train_pos,
                pingze=train_pingze,
                tokenized_data_path='data/processed/tokenized/contents',
                finalized_data_path='data/processed/tokenized/finals',
                sentenced_data_path='data/processed/tokenized/sentences',
                posed_data_path='data/processed/tokenized/pos',
                pingzed_data_path='data/processed/tokenized/pingze',
                full_tokenizer=full_tokenizer,
                full_finalizer=full_finalizer,
                full_sentencer=full_sentencer,
                full_poser=full_poser,
                full_pingzer=full_pingzer,
                enable_final=True,
                enable_sentence=True,
                enable_pos=True,
                enable_pingze=True)
    print('End')
