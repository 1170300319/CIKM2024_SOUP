import sys

sys.path.append('../../')

import collections
import os
import random
from pathlib import Path
import logging
import shutil
import time
from packaging import version
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import gzip
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from transformers import T5Tokenizer, T5TokenizerFast

from util.param import parse_args
from util.utils import LossMeter
from util.dist_utils import reduce_dict
from models.tokenization import P5Tokenizer, P5TokenizerFast

from pretrain_model import P5Pretraining
from pretrain_model_icbu import Q5Pretraining

_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

from ..trainer_base import TrainerBase

import pickle


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


import json


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def ReadLineFromFile(path):
    lines = []
    with open(path, 'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines


def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


args = DotDict()

args.distributed = False
args.multiGPU = True
args.fp16 = True
args.train = "icbu_text"
args.valid = "icbu_text"
args.test = "icbu_text"
args.batch_size = 10
args.optim = 'adamw'
args.warmup_ratio = 0.05
args.lr = 1e-3
args.num_workers = 4
args.clip_grad_norm = 1.0
args.losses = 'text'
args.path = './pretrained_model/t5'
args.backbone = 't5-base'  # small or base
args.output = 'snap/icbu-base-text'
args.epoch = 10
args.local_rank = 0

args.comment = ''
args.train_topk = -1
args.valid_topk = -1
args.dropout = 0.1
args.database = 'text'

args.tokenizer = 'p5'
args.max_text_length = 512
args.do_lower_case = False
args.word_mask_rate = 0.15
args.gen_max_length = 64

args.weight_decay = 0.01
args.adam_eps = 1e-6
args.gradient_accumulation_steps = 1

'''
Set seeds
'''
args.seed = 2022
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

'''
Whole word embedding
'''
args.whole_word_embed = True

cudnn.benchmark = True
ngpus_per_node = torch.cuda.device_count()
args.world_size = ngpus_per_node

LOSSES_NAME = [f'{name}_loss' for name in args.losses.split(',')]
if args.local_rank in [0, -1]:
    print(LOSSES_NAME)
LOSSES_NAME.append('total_loss')  # total loss

args.LOSSES_NAME = LOSSES_NAME

gpu = 0  # Change GPU ID
args.gpu = gpu
args.rank = gpu
print(f'Process Launching at GPU {gpu}')

torch.cuda.set_device('cuda:{}'.format(gpu))

comments = []
dsets = []
if 'icbu-text' in args.train:
    dsets.append('icbu-text')
comments.append(''.join(dsets))
if args.backbone:
    comments.append(args.backbone)
comments.append(''.join(args.losses.split(',')))
if args.comment != '':
    comments.append(args.comment)
comment = '_'.join(comments)

if args.local_rank in [0, -1]:
    print(args)


def create_config(args):
    from transformers import T5Config, BartConfig

    if 't5' in args.backbone:
        config_class = T5Config
    else:
        return None

    config = config_class.from_pretrained(args.path)
    config.dropout_rate = args.dropout
    config.dropout = args.dropout
    config.attention_dropout = args.dropout
    config.activation_dropout = args.dropout
    config.losses = args.losses

    return config


def create_tokenizer(args):
    from transformers import T5Tokenizer, T5TokenizerFast
    from models import P5Tokenizer, P5TokenizerFast

    if 'p5' in args.tokenizer:
        tokenizer_class = P5Tokenizer

    tokenizer_name = args.path

    tokenizer = tokenizer_class.from_pretrained(
        tokenizer_name,
        max_length=args.max_text_length,
        do_lower_case=args.do_lower_case,
    )

    print(tokenizer_class, tokenizer_name)

    return tokenizer


def create_model(model_class, config=None):
    print(f'Building Model at GPU {args.gpu}')

    model_name = args.path

    model = model_class.from_pretrained(
        model_name,
        config=config
    )
    return model


config = create_config(args)

if args.tokenizer is None:
    args.tokenizer = args.backbone

tokenizer = create_tokenizer(args)

model_class = Q5Pretraining
model = create_model(model_class, config)

model = model.cuda()

if 'p5' in args.tokenizer:
    model.resize_token_embeddings(tokenizer.vocab_size)

model.tokenizer = tokenizer

args.load = "./snap/icbu-base-u2qqacq2qi2qq2cu2cq2trq2tctra/Epoch05.pth"

# Load Checkpoint
from ..util.utils import load_state_dict, LossMeter, set_global_logging_level
from pprint import pprint

def load_checkpoint(ckpt_path):
    state_dict = load_state_dict(ckpt_path, 'cpu')
    results = model.load_state_dict(state_dict, strict=False)
    print('Model loaded from ', ckpt_path)
    pprint(results)

ckpt_path = args.load
load_checkpoint(ckpt_path)

from src.icbu_templates.all_templates import all_tasks as task_templates

from torch.utils.data import DataLoader, Dataset, Sampler
from pretrain_data_icbu import get_loader
from evaluate.utils import rouge_score, bleu_score, unique_sentence_percent, root_mean_square_error, mean_absolute_error, feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity
from evaluate.metrics4rec import evaluate_all

test_task_list = {'text': ['3-10']  # or '2-13'
                  }
test_sample_numbers = {'text': (1, 1, 1)}

zeroshot_test_loader = get_loader(
    args,
    test_task_list,
    test_sample_numbers,
    split=args.test,
    mode='test',
    batch_size=args.batch_size,
    workers=args.num_workers,
    distributed=args.distributed,
    multi_dataset=False,
    online=True,
    # cold=True,
)
print(len(zeroshot_test_loader))


def test_all():
    all_info = []
    import common_io
    writer = common_io.table.TableWriter('odps://icbu_ensa_dev/tables/icbu_search_query_res_text_with_query2')

    for i, batch in tqdm(enumerate(zeroshot_test_loader)):
        with torch.no_grad():
            results = model.generate_step(batch)
            beam_outputs = model.generate(
                batch['input_ids'].to('cuda'),
                max_length=50,
                num_beams=10,
                no_repeat_ngram_size=0,
                num_return_sequences=10,
                early_stopping=True
            )
            generated_sents = model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
            # for j, item in enumerate(zip(results, batch['target_text'], batch['source_text'])):
            for j, item in enumerate(
                    zip(results, batch['user_id'], batch['source_text'], batch['target_text'])):
                new_info = {}
                new_info['target_item'] = item[1]
                new_info['gen_item_list'] = generated_sents[j * 10: (j + 1) * 10]
                all_info.append(new_info)
                writer.write(values=[item[1], item[2], item[3], item[0], ','.join(generated_sents[j * 10: (j + 1) * 10])],
                             col_indices=[0, 1, 2, 3, 4])

    writer.close()


def test_parts():
    from util.odps_download import get_odps_object
    o = get_odps_object()

    tokens_predict = []
    tokens_test = []

    # 10*100=1000
    for i, batch in tqdm(enumerate(zeroshot_test_loader)):
        # batch = tqdm(enumerate(zeroshot_test_loader))
        #if i > 10:
        #    break
        with torch.no_grad():
            outputs = model.generate(
                batch['input_ids'].to('cuda'),
                min_length=9,
                num_beams=12,
                num_return_sequences=1,
                num_beam_groups=3,
                repetition_penalty=0.7
            )
            results = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            tokens_predict.extend(results)
            tokens_test.extend(batch['target_text'])

    new_tokens_predict = [l.split() for l in tokens_predict]
    new_tokens_test = [ll.split() for ll in tokens_test]
    BLEU1 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=1, smooth=False)
    BLEU4 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=4, smooth=False)
    ROUGE = rouge_score(tokens_test, tokens_predict)

    print('BLEU-1 {:7.4f}'.format(BLEU1))
    print('BLEU-4 {:7.4f}'.format(BLEU4))
    for (k, v) in ROUGE.items():
        print('{} {:7.4f}'.format(k, v))

    #import common_io
    #writer = common_io.table.TableWriter('odps://icbu_ensa_dev/tables/icbu_search_query_res_text_with_query')

    #for each in all_info:
        #sql = 'insert into icbu_ensa_dev.icbu_search_query_res_text_with_query (utdid, source, source_text, target, ' \
        #      'gen_item_list) values ({}, \'{}\', {}, {}, {})'.format(each[1], each[2], each[3], each[4], each[0])
        #print(sql)
        #writer.write(values=[each[1], each[2], each[3], each[4], each[0]], col_indices=[0, 1, 2, 3, 4])

    #writer.close()

    # sql = 'select DISTINCT query_index, query from icbu_ensa_dev.icbu_search_query_rec_user_history_raw_data where query_index in (' + ', '.join(all_info[0]['gen_item_list'][:5]) + ');'
    #res = o.execute_sql(sql)
    #res = res.open_reader(tunnel=True).to_pandas()
    #res.fillna(" ")
    #res.to_csv('./res.csv')


def test_online():
    import common_io
    writer = common_io.table.TableWriter('odps://icbu_ensa_dev/tables/icbu_search_query_res_text_with_query_online')
    tokens_predict = []
    tokens_test = []
    token_online = []

    for i, batch in tqdm(enumerate(zeroshot_test_loader)):
        with torch.no_grad():
            results = model.generate_step(batch)
            beam_outputs = model.generate(
                batch['input_ids'].to('cuda'),
                max_length=50,
                num_beams=10,
                no_repeat_ngram_size=0,
                num_return_sequences=10,
                early_stopping=True
            )
            generated_sents = model.tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
            for j, item in enumerate(
                    zip(results, batch['user_id'], batch['source_text'], batch['target_text'],
                        batch['online_imps_querylist'], batch['online_imps_poslist'])):
                writer.write(values=[item[1], item[2], item[3], item[0], ', '.join(generated_sents[j * 10: (j + 1) * 10]),
                                     item[4], item[5]],
                             col_indices=[0, 1, 2, 3, 4, 5, 6])
                tokens_predict.append(generated_sents[j * 10])
                tokens_test.append(item[3])
                token_online.append(item[4].split(', ')[0])

    writer.close()

    new_tokens_predict = [l.split() for l in tokens_predict]
    new_tokens_test = [ll.split() for ll in tokens_test]
    new_tokens_online = [lll.split() for lll in token_online]

    print('Q5:')
    BLEU1 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=1, smooth=False)
    BLEU4 = bleu_score(new_tokens_test, new_tokens_predict, n_gram=4, smooth=False)
    ROUGE = rouge_score(tokens_test, tokens_predict)
    print('BLEU-1 {:7.4f}'.format(BLEU1))
    print('BLEU-4 {:7.4f}'.format(BLEU4))
    for (k, v) in ROUGE.items():
        print('{} {:7.4f}'.format(k, v))

    print('Online:')
    BLEU1 = bleu_score(new_tokens_test, new_tokens_online, n_gram=1, smooth=False)
    BLEU4 = bleu_score(new_tokens_test, new_tokens_online, n_gram=4, smooth=False)
    ROUGE = rouge_score(tokens_test, new_tokens_online)
    print('BLEU-1 {:7.4f}'.format(BLEU1))
    print('BLEU-4 {:7.4f}'.format(BLEU4))
    for (k, v) in ROUGE.items():
        print('{} {:7.4f}'.format(k, v))


# 现有一个list，其中每一条都是一个map，存放target和gen item
# test_parts()
# test_all()
test_online()


