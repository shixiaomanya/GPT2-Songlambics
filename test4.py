# train.py!!!!!!!!!
from tqdm import tqdm
import os
import torch
import transformers
from _datetime import datetime
import numpy as np
import random

enable_final = True
enable_sentence = True
enable_relative_pos = True
enable_pingze = True

tokenized_data_path = 'data/processed/tokenized/contents/'
finalized_data_path = 'data/processed/tokenized/finals'
sentenced_data_path = 'data/processed/tokenized/sentences'
posed_data_path = 'data/processed/tokenized/pos'
pingzed_data_path = 'data/processed/tokenized/pingze'

num_pieces = 20    # test3.py中设定的
stride = 48
epochs = 5
batch_size = 1    # 考虑到数据量小
gradient_accumulation = 1
model_config = 'config/config.json'
init_device = 0
warmup_steps = 2000
output_dir = os.path.join('model','lyrics')    # 模型保存路径
start_epoch = 0


def get_shuffled_samples(a, b, c, d, e):
    length = len(a)
    flag = [1, 1, 1, 1]
    if b == []:
        b = np.zeros(length)
        flag[0] = 0
    if c == []:
        c = np.zeros(length)
        flag[1] = 0
    if d == []:
        d = np.zeros(length)
        flag[2] = 0
    if e == []:
        e = np.zeros(length)
        flag[3] = 0
    samples = list(zip(a, b, c, d, e))
    random.shuffle(samples)
    a, b, c, d, e = zip(*samples)
    if flag[0] == 0:
        b = None
    if flag[1] == 0:
        c = None
    if flag[2] == 0:
        d = None
    if flag[3] == 0:
        e = None
    return a, b, c, d, e


######################################
# Training settings
################################
# calculate total training steps
full_len = 0
print('calculating total steps')
for i in tqdm(range(num_pieces)):
    with open(os.path.join(tokenized_data_path, 'tokenized_train_{}.txt'.format(i)), 'r') as f:
        full_len += len([int(item) for item in f.read().strip().split()])
total_steps = int(full_len / stride * epochs / batch_size / gradient_accumulation)
print('total steps = {}'.format(total_steps))

# build model
from module import GPT2Config, GPT2Model, GPT2LMHeadModel
model_config = GPT2Config.from_json_file(model_config)
print('config:\n' + model_config.to_json_string())
model = GPT2LMHeadModel(config=model_config)

model.train()
# set whether to use cuda
gpu_count = torch.cuda.device_count()
if gpu_count > 0:
    from utils import swap_value
    device_ids = [int(i) for i in range(gpu_count)]
    swap_value(device_ids, 0, init_device)
    device = f'cuda:{device_ids[0]}'
else:
    device = 'cpu'
print('using device:', device)
model.to(device)

# check parameters number of the built model
num_parameters = 0
parameters = model.parameters()
for parameter in parameters:
    num_parameters += parameter.numel()
print('number of parameters: {}'.format(num_parameters))

# set optimizer
optimizer = transformers.AdamW(model.parameters(), lr=1.5e-4, correct_bias=True)
# change WarmupLinearSchedule to get_linear_schedule_with_warmup for current version of Transformers
scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                         num_training_steps=total_steps)
# 以上是transformer = 3.4.0版本的写法，如果transformer = 2.1.1，以上写法会报错，如下：
# AttributeError: module 'transformers' has no attribute 'get_linear_schedule_with_warmup'
# 原get_linear_schedule_with_warmup改为WarmupLinearSchedule
# 原num_warmup_steps改为warmup_steps，num_training_steps改为t_total

# set whether to use multi GPUs
multi_gpu = False
if gpu_count > 1:
    from torch.nn import DataParallel
    print("Let's use", gpu_count, "GPUs!", device_ids)
    model = DataParallel(model, device_ids=device_ids)
    multi_gpu = True

# set log info
# log_dir = os.path.join('tensorboard_summary/', 'lyrics', 'lyric_with_final_small', '1a')
# tb_writer = SummaryWriter(log_dir=log_dir)
# assert args.log_step % args.gradient_accumulation == 0

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print('starting training')
overall_step = 0
running_loss = 0
for epoch in range(0, epochs):
    print('epoch {}'.format(epoch + 1))
    now = datetime.now()
    print('time: {}'.format(now))

    # shuffle pieces of data
    x = np.linspace(0, num_pieces-1, num_pieces, dtype=np.int32)
    random.shuffle(x)

    piece_num = 0
    # enumerate data pieces
    for i in x:
        # prepare training sentences
        with open(os.path.join(tokenized_data_path, 'tokenized_train_{}.txt'.format(i)), 'r') as f:
            line = f.read().strip()
        tokens = line.split()
        tokens = [int(token) for token in tokens]
        # tokens = torch.Tensor(tokens)

        if enable_final:  # 加入韵母embedding
            with open(os.path.join(finalized_data_path, 'tokenized_train_{}.txt'.format(i)), 'r') as f:
                final = f.read().strip()
            finals = final.split()
            # print(len(finals))
            finals = [int(final) for final in finals]
            # finals = torch.Tensor(finals)

        if enable_sentence:
            with open(os.path.join(sentenced_data_path, 'tokenized_train_{}.txt'.format(i)), 'r') as f:
                sentence = f.read().strip()
            sentences = sentence.split()
            # print(len(sentences))
            sentences = [int(sentence) for sentence in sentences]
            # sentences = torch.Tensor(sentences)

        if enable_relative_pos:
            with open(os.path.join(posed_data_path, 'tokenized_train_{}.txt'.format(i)), 'r') as f:
                p = f.read().strip()
            pos = p.split()
            # print(len(sentences))
            pos = [int(p) for p in pos]
            # sentences = torch.Tensor(sentences)

        if enable_pingze:
            with open(os.path.join(pingzed_data_path, 'tokenized_train_{}.txt'.format(i)), 'r') as f:
                pingze = f.read().strip()
            pingzes = pingze.split()
            # print(len(sentences))
            pingzes = [int(pingze) for pingze in pingzes]
            # sentences = torch.Tensor(sentences)
        # print('training: ', len(tokens), len(finals), len(sentences))

        start_point = 0
        samples_token, samples_final, samples_sentence, samples_pos, samples_pingze = [], [], [], [], []
        n_ctx = model_config.n_ctx  # the length of a sentence for training
        while start_point < len(tokens) - stride:
            samples_token.append(tokens[
                                 start_point: start_point + stride])
            if enable_final:  # args.enable_final
                samples_final.append(finals[start_point: start_point + stride])
            if enable_sentence:  # args.enable_sentence
                samples_sentence.append(sentences[start_point: start_point + stride])
            if enable_relative_pos:  # args.enable_relative_pos
                samples_pos.append(pos[start_point: start_point + stride])
            if enable_pingze:  # args.enable_beat
                samples_pingze.append(pingzes[start_point: start_point + stride])
            start_point += stride
        if start_point < len(tokens):  # 最后不够stride要求的话，后面的都会被舍掉，所以要再加回来，注意每次append都是连了一个列表过去
            samples_token.append(tokens[len(tokens) - stride:])
            if enable_final:  # args.enable_final
                samples_final.append(finals[len(tokens) - stride:])
            if enable_sentence:  # args.enable_sentence
                samples_sentence.append(sentences[len(tokens) - stride:])
            if enable_relative_pos:  # args.enable_relative_pos
                samples_pos.append(pos[len(tokens) - stride:])
            if enable_pingze:  # args.enable_beat
                samples_pingze.append(pingzes[len(tokens) - stride:])

        samples_token, samples_final, samples_sentence, samples_pos, samples_pingze = get_shuffled_samples(
            samples_token, samples_final,
            samples_sentence, samples_pos, samples_pingze
        )

        # enumerate batch data
        for step in range(len(samples_token)):  # drop last
            print('step =', step)
            #  prepare batch data
            batch_token = samples_token[step: (step + 1)]
            batch_inputs_token = torch.Tensor(batch_token).long().to(device)

            if samples_final is not None:
                batch_final = samples_final[step: (step + 1)]
                batch_inputs_final = torch.Tensor(batch_final).long().to(device)
            else:
                batch_inputs_final = None

            if samples_sentence is not None:
                batch_sentence = samples_sentence[step: (step + 1)]
                batch_inputs_sentence = torch.Tensor(batch_sentence).long().to(device)
            else:
                batch_inputs_sentence = None

            if samples_pos is not None:
                batch_pos = samples_pos[step: (step + 1)]
                batch_inputs_pos = torch.Tensor(batch_pos).long().to(device)
            else:
                batch_inputs_pos = None

            if samples_pingze is not None:
                batch_pingze = samples_pingze[step: (step + 1)]
                batch_inputs_pingze = torch.Tensor(batch_pingze).long().to(device)
            else:
                batch_inputs_pingze = None

            #  forward pass
            # Notes: Using Transformers, the labels are shifted inside the model,
            #           i.e. you can set labels = input_ids
            outputs = model.forward(input_ids=batch_inputs_token,  # 第一步：前向传播
                                    sentence_ids=batch_inputs_sentence,
                                    final_ids=batch_inputs_final,
                                    pos_ids=batch_inputs_pos,
                                    pingze_ids=batch_inputs_pingze,
                                    labels=batch_inputs_token)
            loss, logits = outputs[:2]
            # print(loss,logits)

            #  get loss
            if multi_gpu:
                loss = loss.mean()
                '''
                running_loss += loss
                overall_step += 1
                '''
            if gradient_accumulation > 1:
                loss = loss / gradient_accumulation

            #  loss backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            #  optimizer step
            if (overall_step + 1) % 1 == 0:
                running_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # log info of training process
            if (overall_step + 1) % 1 == 0:
                loss_log = running_loss
                # tb_writer.add_scalar('loss', loss_log, overall_step)
                print('now time: {}:{}. Step {} of piece {} of epoch {}, loss {}'.format(datetime.now().hour,
                                                                                         datetime.now().minute,
                                                                                         step + 1, piece_num,
                                                                                         epoch + 1, loss_log))
                running_loss = 0

            overall_step += 1

        piece_num += 1

    # save model per epoch
    print('saving model for epoch {}'.format(epoch + 1))
    if not os.path.exists(os.path.join(output_dir, 'model_epoch{}'.format(epoch + 1))):
        os.mkdir(os.path.join(output_dir, 'model_epoch{}'.format(epoch + 1)))
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(os.path.join(output_dir, 'model_epoch{}'.format(epoch + 1)))
    # torch.save(scheduler.state_dict(), output_dir + 'model_epoch{}/scheduler.pt'.format(epoch + 1))
    # torch.save(optimizer.state_dict(), output_dir + 'model_epoch{}/optimizer.pt'.format(epoch + 1))
    print('epoch {} finished'.format(epoch + 1))

    then = datetime.now()
    print('time: {}'.format(then))
    print('time for one epoch: {}'.format(then - now))

# save final model
print('training finished')
if not os.path.exists(os.path.join(output_dir, 'final_model')):
    os.mkdir(os.path.join(output_dir, 'final_model'))
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(os.path.join(output_dir, 'final_model'))
# torch.save(scheduler.state_dict(), output_dir + 'final_model/scheduler.pt')
# torch.save(optimizer.state_dict(), output_dir + 'final_model/optimizer.pt')