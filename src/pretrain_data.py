from torch.utils.data import DataLoader, Dataset
import json
import gzip
import random
import pickle
import math
import torch
import numpy as np
from models.multiSetSampler import DisMultiSetSampler, MyDistributedSampler

from models.tokenization import P5Tokenizer


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


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


class P5_ICBU_Dataset(Dataset):
    def __init__(self, all_tasks, task_list, tokenizer, args, sample_numbers, mode='train', split='toys',
                 rating_augment=False, sample_type='random', cold=False, online=False):
        self.all_tasks = all_tasks
        self.task_list = task_list
        self.tokenizer = tokenizer
        self.args = args
        self.sample_numbers = sample_numbers
        self.split = split
        self.rating_augment = rating_augment
        self.sample_type = sample_type
        self.cold = cold
        self.online = online

        print('Data sources: ', split.split(','))
        self.mode = mode

        print('Load data from ODPS...')
        self.u2q_data = None
        self.qac_data = None
        self.q2q_data = None
        self.i2q_data = None
        self.q2c_data = None
        self.u2c_data = None
        self.u2qc_data = None
        self.q2t_data = None
        self.pro_data = None
        self.rm_data = None
        self.tradition_data = 0
        self.download_data()

        print('compute_datum_info')
        print('task list, ', self.task_list)
        # print('data example, ', self.data.iloc[0])
        self.total_length = 0
        self.datum_info = []
        self.compute_datum_info()

        if 'Q2C' in self.task_list.keys():
            print('Calculate category list')
            self.cateSet = None
            self.getQ2CateSet()

        print('Get query set')
        self.querySet = None
        self.getQuerySet()

        # 没有写task转换的逻辑，要从dataset传不同的task进来
        #if 'sequential' in self.task_list.keys():
        #    self.task_name = 'sequential'
        #if 'text' in self.task_list.keys():
        #    self.task_name = 'text'

    def getQ2CateSet(self):
        self.cateSet = set()
        if self.q2c_data is not None:
            for i in range(len(self.q2c_data)):
                seq_data = self.q2c_data.iloc[i]
                cate_seq = seq_data['predict_cate_list']
                cate_seq = cate_seq.split(', ')
                self.cateSet.update(set(cate_seq))
        # list才可以遍历
        self.cateSet = list(self.cateSet)

    def getQuerySet(self):
        self.querySet = set()
        if self.u2q_data is not None:
            for i in range(len(self.u2q_data)):
                seq_data = self.u2q_data.iloc[i]
                self.querySet.add(seq_data['target_seq'])
        # if self.qac_data is not None:
        #     for i in range(len(self.qac_data)):
        #         seq_data = self.qac_data.iloc[i]
        #         self.querySet.add(seq_data['target_seq'])
        if self.q2q_data is not None:
            for i in range(len(self.q2q_data)):
                seq_data = self.q2q_data.iloc[i]
                self.querySet.add(seq_data['target_seq'])

        self.querySet = list(self.querySet)

    def download_data(self):
        if self.mode == 'train':
            if 'sequential' in self.task_list.keys():
                raise RuntimeError('Dont use ID based data now.')
            if 'text' in self.task_list.keys():
                self.u2q_data = getU2QTrains()
            if 'QAC' in self.task_list.keys():
                self.qac_data = getQACTrains()
            if 'Q2Q' in self.task_list.keys():
                self.q2q_data = getQ2QTrains()
            if 'I2Q' in self.task_list.keys():
                self.i2q_data = getI2QTrains()
            if 'Q2C' in self.task_list.keys():
                self.q2c_data = getQ2CTrains()
            if 'U2C' in self.task_list.keys():
                self.u2c_data = getU2CTrains()
            if 'U2QC' in self.task_list.keys():
                self.u2qc_data = getU2QCTrains()
            if 'Q2T' in self.task_list.keys() or 'Q2TR' in self.task_list.keys() or 'Q2TC' in self.task_list.keys():
                self.q2t_data = getQ2TTrains()
            if 'PRO' in self.task_list.keys():
                self.pro_data = getPROTrains()
            if 'RM' in self.task_list.keys():
                self.rm_data = getRMTrains()

            # pd_reader = pd.read_csv('./data_new.csv')
            # self.id_data = pd_reader[: int(0.8 * len(pd_reader))]
            # pd_reader = pd.read_csv('./data_text.csv')
            # self.text_data = pd_reader[: int(0.8 * len(pd_reader))]
        elif self.mode == 'val':
            if 'sequential' in self.task_list.keys():
                raise RuntimeError('Dont use ID based data now.')
            if 'text' in self.task_list.keys():
                self.u2q_data = getU2QVals()
            if 'QAC' in self.task_list.keys():
                self.qac_data = getQACVals()
            if 'Q2Q' in self.task_list.keys():
                self.q2q_data = getQ2QVals()
            if 'I2Q' in self.task_list.keys():
                self.i2q_data = getI2QVals()
            if 'Q2C' in self.task_list.keys():
                self.q2c_data = getQ2CVals()
            if 'U2C' in self.task_list.keys():
                self.u2c_data = getU2CVals()
            if 'U2QC' in self.task_list.keys():
                self.u2qc_data = getU2QCVals()
            if 'Q2T' in self.task_list.keys() or 'Q2TR' in self.task_list.keys() or 'Q2TC' in self.task_list.keys():
                self.q2t_data = getQ2TVals()
            if 'PRO' in self.task_list.keys():
                self.pro_data = getPROVals()
            if 'RM' in self.task_list.keys():
                self.rm_data = getRMVals()

        elif self.mode == 'test':
            if 'sequential' in self.task_list.keys():
                raise RuntimeError('Dont use ID based data now.')
            if 'text' in self.task_list.keys():
                if self.cold:
                    self.u2q_data = getU2QColdTests()
                elif self.online:
                    self.u2q_data = getU2QOnline()
                else:
                    self.u2q_data = getU2QTests()
            if 'QAC' in self.task_list.keys():
                self.qac_data = getQACTests()
            if 'Q2Q' in self.task_list.keys():
                self.q2q_data = getQ2QTests()
            if 'I2Q' in self.task_list.keys():
                self.i2q_data = getI2QTests()
            if 'Q2C' in self.task_list.keys():
                if self.cold:
                    self.q2c_data = getQ2C_zeroshot()
                else:
                    self.q2c_data = getQ2CTests()
            if 'U2C' in self.task_list.keys():
                self.u2c_data = getU2CTests()
            if 'U2QC' in self.task_list.keys():
                self.u2qc_data = getU2QCTests()
            if 'Q2T' in self.task_list.keys() or 'Q2TR' in self.task_list.keys() or 'Q2TC' in self.task_list.keys():
                self.q2t_data = getQ2TTests()
            if 'PRO' in self.task_list.keys():
                self.pro_data = getPROTests()
            if 'RM' in self.task_list.keys():
                self.rm_data = getRMTests()

        else:
            raise NotImplementedError

    # compute_datum_info function intends to plan which data sample to be used for which task group according to the
    # sample numbers in train_sample_numbers of pretrain.py
    def compute_datum_info(self):
        # 先算information再加length，这样第一个数据集的bias就是0，第二个数据集的bias就是第一个数据集的长度，以此类推
        # P5的代码里还加入了多个prompt的逻辑
        # 如果要训练每个prompt，就要每个prompt在重新计算一遍数据，但是感觉没必要这么做
        # 我这里没有写，就是每个task都是平等的，然后prompt随机抽
        if 'sequential' in self.task_list.keys():
            self.datum_info += [('sequential', self.total_length) for _ in range(len(self.u2q_data))]
            self.total_length += len(self.u2q_data)
        if 'text' in self.task_list.keys():
            self.datum_info += [('text', self.total_length) for _ in range(len(self.u2q_data))]
            self.total_length += len(self.u2q_data)
        if 'QAC' in self.task_list.keys():
            self.datum_info += [('QAC', self.total_length) for _ in range(len(self.qac_data))]
            self.total_length += len(self.qac_data)
        if 'Q2Q' in self.task_list.keys():
            self.datum_info += [('Q2Q', self.total_length) for _ in range(len(self.q2q_data))]
            self.total_length += len(self.q2q_data)
        if 'I2Q' in self.task_list.keys():
            self.datum_info += [('I2Q', self.total_length) for _ in range(len(self.i2q_data))]
            self.total_length += len(self.i2q_data)
        if 'Q2C' in self.task_list.keys():
            self.datum_info += [('Q2C', self.total_length) for _ in range(len(self.q2c_data))]
            self.total_length += len(self.q2c_data)
        if 'U2C' in self.task_list.keys():
            self.datum_info += [('U2C', self.total_length) for _ in range(len(self.u2c_data))]
            self.total_length += len(self.u2c_data)
        if 'U2QC' in self.task_list.keys():
            self.datum_info += [('U2QC', self.total_length) for _ in range(len(self.u2qc_data))]
            self.total_length += len(self.u2qc_data)
        if 'Q2T' in self.task_list.keys():
            self.datum_info += [('Q2T', self.total_length) for _ in range(len(self.q2t_data))]
            self.total_length += len(self.q2t_data)
        # use q2t data
        if 'Q2TR' in self.task_list.keys():
            self.datum_info += [('Q2TR', self.total_length) for _ in range(len(self.q2t_data))]
            self.total_length += len(self.q2t_data)
        if 'Q2TC' in self.task_list.keys():
            self.datum_info += [('Q2TC', self.total_length) for _ in range(len(self.q2t_data))]
            self.total_length += len(self.q2t_data)
        if 'PRO' in self.task_list.keys():
            self.datum_info += [('PRO', self.total_length) for _ in range(len(self.pro_data))]
            self.total_length += len(self.pro_data)
        if 'RM' in self.task_list.keys():
            self.datum_info += [('RM', self.total_length) for _ in range(len(self.rm_data))]
            self.total_length += len(self.rm_data)

        # traditional这个任务的数据实际上就是U2Q，QAC，Q2Q这三个任务
        if 'traditional' in self.task_list.keys():
            self.tradition_data = self.total_length
            # if 'text' in self.task_list.keys():
            #     self.tradition_data += len(self.u2q_data)*2
            # if 'QAC' in self.task_list.keys():
            #     self.tradition_data += len(self.qac_data)
            # if 'Q2Q' in self.task_list.keys():
            #     self.tradition_data += len(self.q2q_data)*2

            self.datum_info += [('traditional', self.total_length) for _ in range(self.tradition_data)]
            self.total_length += self.tradition_data

        print('total length: ', self.total_length)

    # output the length for all dataset
    def getLengthAllDataset(self):
        datamap = {}
        if 'sequential' in self.task_list.keys():
            datamap['sequential'] = len(self.u2q_data)
        if 'text' in self.task_list.keys():
            datamap['text'] = len(self.u2q_data)
        if 'QAC' in self.task_list.keys():
            datamap['QAC'] = len(self.qac_data)
        if 'Q2Q' in self.task_list.keys():
            datamap['Q2Q'] = len(self.q2q_data)
        if 'I2Q' in self.task_list.keys():
            datamap['I2Q'] = len(self.i2q_data)
        if 'Q2C' in self.task_list.keys():
            datamap['Q2C'] = len(self.q2c_data)
        if 'U2C' in self.task_list.keys():
            datamap['U2C'] = len(self.u2c_data)
        if 'U2QC' in self.task_list.keys():
            datamap['U2QC'] = len(self.u2qc_data)
        if 'Q2T' in self.task_list.keys():
            datamap['Q2T'] = len(self.q2t_data)
        if 'Q2TR' in self.task_list.keys():
            datamap['Q2TR'] = len(self.q2t_data)
        if 'Q2TC' in self.task_list.keys():
            datamap['Q2TC'] = len(self.q2t_data)
        if 'PRO' in self.task_list.keys():
            datamap['PRO'] = len(self.pro_data)
        if 'RM' in self.task_list.keys():
            datamap['RM'] = len(self.rm_data)
        if 'traditional' in self.task_list.keys():
            datamap['traditional'] = self.tradition_data

        return datamap

    # return (utdid, county, category)
    def getTraditionalIdx(self, idx):
        # 把traditional的数据映射回三个任务上
        task_name, bias = self.datum_info[idx]
        if task_name == 'text':
            seq_data = self.u2q_data.iloc[idx - bias]
        # elif task_name == 'QAC':
        #     seq_data = self.qac_data.iloc[idx - bias]
        elif task_name == 'Q2Q':
            seq_data = self.q2q_data.iloc[idx - bias]
        else:
            raise NotImplementedError

        return seq_data

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):

        out_dict = {'args': self.args}

        loss_weight = 1.0

        # 存task name和bias
        # task name用于判断使用哪些prompt
        # bias用于多个数据集处理时从全局idx计算某个数据集内的idx
        task_name, bias = self.datum_info[idx]

        # 可能用到的中间属性
        user_id = None
        source = None
        prefix = None
        query_trigger = None
        query_trigger_neg = None
        cate_list_seq = None
        time_seq = None
        country = None
        prod_id = None
        prod_title = None
        prod_keywords = None
        online_imps_querylist = None
        online_imps_poslist = None
        source_text_neg = None
        reward_neg = None
        reward_pos = None

        if task_name == 'sequential':
            seq_data = self.u2q_data.iloc[idx-bias]
            # sequential_datum = self.sequential_data[datum_idx]
            # sequence = sequential_datum.split()  # seq里存的数据是id
            user_id = seq_data['utdid']
            source = seq_data['source']
            target = seq_data['target']
            # user_desc = self.user_id2name[user_id]

            task_candidates = self.task_list[task_name]
            # 随机抽一个prompt
            task_idx = random.randint(0, len(task_candidates) - 1)  # random choose the task index for task_candidates
            task_template = self.all_tasks['sequential'][task_candidates[task_idx]]
            assert task_template['task'] == 'sequential'

            if task_template['id'] in ['2-1', '2-2', '2-3']:
                source_text = task_template['source'].format(user_id, source)
                target_text = task_template['target'].format(target)
            else:
                raise NotImplementedError
        elif task_name == 'text':
            text_data = self.u2q_data.iloc[idx-bias]
            user_id = text_data['utdid']
            source = text_data['source_seq']
            target = text_data['target_seq']
            cate_list_seq = text_data['cate_list_seq']
            country = text_data['county']
            time_seq = text_data['time_seq']

            if self.online:
                online_imps_querylist = text_data['online_imps_querylist']
                online_imps_poslist = text_data['online_imps_poslist']

            task_candidates = self.task_list[task_name]
            task_idx = random.randint(0, len(task_candidates) - 1)  # random choose the task index for task_candidates
            task_template = self.all_tasks['text'][task_candidates[task_idx]]
            assert task_template['task'] == 'text'

            if task_template['id'] in ['3-1', '3-2', '3-3', '3-4', '3-5', '3-6']:
                source_text = task_template['source'].format(user_id, source)
                target_text = task_template['target'].format(target)
            elif task_template['id'] in ['3-7', '3-8', '3-9']:
                source_text = task_template['source'].format(user_id, source, country, cate_list_seq)
                target_text = task_template['target'].format(target)
            elif task_template['id'] in ['3-10', '3-11', '3-12']:
                source_text = task_template['source'].format(user_id, country, cate_list_seq, source)
                target_text = task_template['target'].format(target)
            elif task_template['id'] in ['3-13', '3-15']:
                source_text = task_template['source'].format(user_id, country, cate_list_seq)
                target_text = task_template['target'].format(target)
            elif task_template['id'] in ['3-14', '3-16']:
                source_text = task_template['source'].format(user_id, source)
                target_text = task_template['target'].format(target)
            else:
                raise NotImplementedError
        elif task_name == 'QAC':
            seq_data = self.qac_data.iloc[idx-bias]
            user_id = seq_data['utdid']
            prefix = seq_data['prefix']
            cate_list_seq = seq_data['cate_list_seq']
            country = seq_data['county']

            source = seq_data['source_seq']
            target = seq_data['target_seq']

            task_candidates = self.task_list[task_name]
            # 随机抽一个prompt
            task_idx = random.randint(0, len(task_candidates) - 1)  # random choose the task index for task_candidates
            task_template = self.all_tasks['QAC'][task_candidates[task_idx]]
            assert task_template['task'] == 'QAC'

            if task_template['id'] in ['4-1', '4-2', '4-3']:
                source_text = task_template['source'].format(user_id, source, prefix)
                target_text = task_template['target'].format(target)
            elif task_template['id'] in ['4-4', '4-5', '4-6']:
                source_text = task_template['source'].format(user_id, cate_list_seq, country, prefix)
                target_text = task_template['target'].format(target)
            elif task_template['id'] in ['4-7', '4-8', '4-9', '4-11']:
                source_text = task_template['source'].format(prefix, user_id, country, cate_list_seq)
                target_text = task_template['target'].format(target)
            elif task_template['id'] in ['4-10']:
                source_text = task_template['source'].format(prefix, user_id)
                target_text = task_template['target'].format(target)
            else:
                raise NotImplementedError
        elif task_name == 'Q2Q':
            seq_data = self.q2q_data.iloc[idx - bias]
            user_id = seq_data['utdid']
            query_trigger = seq_data['query_trigger']
            cate_list_seq = seq_data['cate_list_seq']
            country = seq_data['county']
            time_seq = seq_data['time_seq']

            source = seq_data['source_seq']
            target = seq_data['target_seq']

            task_candidates = self.task_list[task_name]
            # 随机抽一个prompt
            task_idx = random.randint(0,
                                      len(task_candidates) - 1)  # random choose the task index for task_candidates
            task_template = self.all_tasks['Q2Q'][task_candidates[task_idx]]
            assert task_template['task'] == 'Q2Q'

            if task_template['id'] in ['5-1', '5-2', '5-3', '5-4']:
                source_text = task_template['source'].format(user_id, source, query_trigger)
                target_text = task_template['target'].format(target)
            elif task_template['id'] in ['5-5', '5-6', '5-7', '5-8']:
                source_text = task_template['source'].format(user_id, source, country, cate_list_seq, query_trigger)
                target_text = task_template['target'].format(target)
            elif task_template['id'] in ['5-9', '5-10', '5-11']:
                source_text = task_template['source'].format(query_trigger, user_id, country, cate_list_seq, source)
                target_text = task_template['target'].format(target)
            elif task_template['id'] in ['5-12', '5-13']:
                source_text = task_template['source'].format(query_trigger, user_id, country, cate_list_seq)
                target_text = task_template['target'].format(target)
            else:
                raise NotImplementedError
        elif task_name == 'I2Q':
            seq_data = self.i2q_data.iloc[idx - bias]
            prod_id = seq_data['prod_id']
            prod_title = seq_data['prod_title']
            prod_keywords = seq_data['prod_keywords']
            target = seq_data['target_query_text']

            task_candidates = self.task_list[task_name]
            # 随机抽一个prompt
            task_idx = random.randint(0,
                                      len(task_candidates) - 1)  # random choose the task index for task_candidates
            task_template = self.all_tasks['I2Q'][task_candidates[task_idx]]
            assert task_template['task'] == 'I2Q'

            if task_template['id'] in ['6-1', '6-2', '6-3']:
                source_text = task_template['source'].format(prod_id, prod_title, prod_keywords)
                target_text = task_template['target'].format(target)
            elif task_template['id'] in ['6-4', '6-5', '6-6']:
                source_text = task_template['source'].format(prod_id, prod_title, prod_keywords)
                target_text = task_template['target'].format(target)
            else:
                raise NotImplementedError
        elif task_name == 'Q2C':
            seq_data = self.q2c_data.iloc[idx - bias]

            query_trigger = seq_data['query']
            target = seq_data['predict_cate_list']
            cate_list_seq = seq_data['predict_cate_list']

            task_candidates = self.task_list[task_name]
            # 随机抽一个prompt
            task_idx = random.randint(0,
                                      len(task_candidates) - 1)  # random choose the task index for task_candidates
            task_template = self.all_tasks['Q2C'][task_candidates[task_idx]]
            assert task_template['task'] == 'Q2C'

            if task_template['id'] in ['8-1', '8-2']:
                source_text = task_template['source'].format(query_trigger)
                target_text = task_template['target'].format(target)
            elif task_template['id'] in ['8-3', '8-4']:
                source_text = task_template['source'].format(query_trigger)
                target_text = task_template['target'].format(target)
            elif task_template['id'] in ['8-5']:
                if self.cateSet is None:
                    print('Cate set is None')
                    return None
                candidate_samples = []
                candidate_num = 5
                while len(candidate_samples) < candidate_num:
                    sample_ids = np.random.choice(self.cateSet, candidate_num, replace=False)
                    candidate_samples.extend(sample_ids)
                candidate_samples = candidate_samples[:candidate_num]
                candidate_samples.append(target)
                random.shuffle(candidate_samples)

                source_text = task_template['source'].format(query_trigger, ', '.join(candidate_samples))
                target_text = task_template['target'].format(target)
            elif task_template['id'] in ['8-6']:
                if self.cold:
                    label = seq_data['target']
                    source_text = task_template['source'].format(query_trigger, target)
                    if label == '1':
                        target_text = task_template['target'].format('yes')
                    else:
                        target_text = task_template['target'].format('no')
                else:
                    if self.cateSet is None:
                        print('Cate set is None')
                        return None
                    if random.random() > 0.5:
                        # positive
                        query_list = target.split(', ')
                        rand_idx = random.randint(0, len(query_list)-1)
                        source_text = task_template['source'].format(query_trigger, query_list[rand_idx])
                        target_text = task_template['target'].format('yes')
                    else:
                        candidate_num = 1
                        sample_ids = np.random.choice(self.cateSet, candidate_num, replace=False)

                        source_text = task_template['source'].format(query_trigger, sample_ids[0])
                        target_text = task_template['target'].format('no')
            else:
                raise NotImplementedError
        elif task_name == 'U2C':
            seq_data = self.u2c_data.iloc[idx - bias]

            user_id = seq_data['utdid']
            target = seq_data['cate_list_seq']

            task_candidates = self.task_list[task_name]
            # 随机抽一个prompt
            task_idx = random.randint(0,
                                      len(task_candidates) - 1)  # random choose the task index for task_candidates
            task_template = self.all_tasks['U2C'][task_candidates[task_idx]]
            assert task_template['task'] == 'U2C'

            if task_template['id'] in ['9-1', '9-2']:
                source_text = task_template['source'].format(user_id)
                target_text = task_template['target'].format(target)
            elif task_template['id'] in ['9-3', '9-4']:
                # 从target中随机mask掉一个，剩下的放到输入里
                tmp_cate_list = target.split(', ')
                mask_idx = random.randint(0, len(tmp_cate_list)-1)
                tmp = tmp_cate_list[mask_idx]
                tmp_cate_list[mask_idx] = '[M]'
                source_text = task_template['source'].format(user_id, ', '.join(tmp_cate_list))
                target_text = task_template['target'].format(tmp)
            else:
                raise NotImplementedError
        elif task_name == 'traditional':
            seq_data = self.getTraditionalIdx(idx - bias)
            user_id = seq_data['utdid']
            cate_list_seq = seq_data['cate_list_seq']
            country = seq_data['county']

            target = seq_data['target_seq']

            task_candidates = self.task_list[task_name]
            # 随机抽一个prompt
            task_idx = random.randint(0,
                                      len(task_candidates) - 1)  # random choose the task index for task_candidates
            task_template = self.all_tasks['traditional'][task_candidates[task_idx]]
            assert task_template['task'] == 'traditional'

            if task_template['id'] in ['10-1', '10-2']:
                # 判断题
                if random.random() > 0.5:
                    # positive
                    source_text = task_template['source'].format(target, user_id, country, cate_list_seq)
                    target_text = task_template['target'].format('yes')
                else:
                    candidate_num = 1
                    sample_ids = np.random.choice(self.querySet, candidate_num, replace=False)
                    source_text = task_template['source'].format(sample_ids[0], user_id, country, cate_list_seq)
                    target_text = task_template['target'].format('no')
            elif task_template['id'] in ['10-3', '10-4']:
                # 选择题
                candidate_samples = []
                candidate_num = 4
                while len(candidate_samples) < candidate_num:
                    sample_ids = np.random.choice(self.querySet, candidate_num, replace=False)
                    candidate_samples.extend(sample_ids)
                candidate_samples = candidate_samples[:candidate_num]

                # 随机idx
                candidate_samples.append(target)
                randidx = random.randint(0, candidate_num)
                tmp = candidate_samples[randidx]
                candidate_samples[randidx] = candidate_samples[candidate_num]
                candidate_samples[candidate_num] = tmp

                source_text = task_template['source'].format(', '.join(candidate_samples), user_id, country, cate_list_seq)
                target_text = task_template['target'].format(randidx)
            else:
                raise NotImplementedError
        elif task_name == 'U2QC':
            seq_data = self.u2qc_data.iloc[idx - bias]

            user_id = seq_data['utdid']
            query_trigger = seq_data['query_trigger']
            cate_list_seq = seq_data['cate_list_seq']
            country = seq_data['county']
            time_seq = seq_data['time_seq']

            source = seq_data['source_seq']
            target = seq_data['target_seq']

            task_candidates = self.task_list[task_name]
            # 随机抽一个prompt
            task_idx = random.randint(0,
                                      len(task_candidates) - 1)  # random choose the task index for task_candidates
            task_template = self.all_tasks['U2QC'][task_candidates[task_idx]]
            assert task_template['task'] == 'U2QC'

            if task_template['id'] in ['7-4', ]:
                source_text = task_template['source'].format(query_trigger, user_id, country, cate_list_seq)
                target_text = task_template['target'].format(target)
            elif task_template['id'] in ['7-5', ]:
                source_text = task_template['source'].format(query_trigger, user_id, source)
                target_text = task_template['target'].format(target)
            else:
                raise NotImplementedError
        elif task_name == 'Q2T' or task_name == 'Q2TR' or task_name == 'Q2TC':
            seq_data = self.q2t_data.iloc[idx - bias]

            query_trigger = seq_data['query']
            prod_id = seq_data['prod_id']
            prod_title = seq_data['title']
            cate_list_seq = seq_data['category']

            target = seq_data['result']

            task_candidates = self.task_list[task_name]
            # 随机抽一个prompt
            task_idx = random.randint(0,
                                      len(task_candidates) - 1)  # random choose the task index for task_candidates
            # prompt还是Q2T的prompt
            task_template = self.all_tasks['Q2T'][task_candidates[task_idx]]
            assert task_template['task'] == 'Q2T'

            if task_template['id'] in ['11-1', ]:
                source_text = task_template['source'].format(query_trigger, prod_id, prod_title)
                target_text = task_template['target'].format(target)
            elif task_template['id'] in ['11-2', ]:
                source_text = task_template['source'].format(query_trigger, prod_id, prod_title, cate_list_seq)
                target_text = task_template['target'].format(target)
            elif task_template['id'] in ['11-3', ]:
                source_text = task_template['source'].format(query_trigger, prod_id, prod_title)
                target_text = task_template['target'].format(target, cate_list_seq)
            elif task_template['id'] in ['11-4', ]:
                source_text = task_template['source'].format(query_trigger, prod_id, prod_title)
                target_text = task_template['target'].format(cate_list_seq)
            else:
                raise NotImplementedError
        elif task_name == 'RM':
            seq_data = self.rm_data.iloc[idx - bias]

            user_id = seq_data['utdid']
            query_trigger = seq_data['query']
            cate_list_seq = seq_data['cate_list_seq']
            country = seq_data['county']
            time_seq = seq_data['time_seq']

            source = seq_data['source_seq']
            target = seq_data['target']

            task_candidates = self.task_list[task_name]
            # 随机抽一个prompt
            task_idx = random.randint(0,
                                      len(task_candidates) - 1)  # random choose the task index for task_candidates
            # prompt还是Q2T的prompt
            task_template = self.all_tasks['RM'][task_candidates[task_idx]]
            assert task_template['task'] == 'RM'

            if task_template['id'] in ['12-1', '12-2']:
                source_text = task_template['source'].format(query_trigger, country, cate_list_seq, source)
                if int(target) == 0:
                    target_text = task_template['target'].format(0)
                else:
                    target_text = task_template['target'].format(1)
            elif task_template['id'] in ['12-3', ]:
                source_text = task_template['source'].format(query_trigger, )
                if int(target) == 0:
                    target_text = task_template['target'].format(0)
                else:
                    target_text = task_template['target'].format(1)
            else:
                raise NotImplementedError
        elif task_name == 'PRO':
            seq_data = self.pro_data.iloc[idx - bias]

            user_id = seq_data['utdid']
            source = seq_data['source_seq']
            cate_list_seq = seq_data['cate_list_seq']
            country = seq_data['county']
            # time_seq = seq_data['time_seq']

            query_trigger_neg = seq_data['query1']
            query_trigger = seq_data['query2']

            reward_neg = seq_data['reward1']
            reward_pos = seq_data['reward2']

            task_candidates = self.task_list[task_name]
            # 随机抽一个prompt
            task_idx = random.randint(0,
                                      len(task_candidates) - 1)  # random choose the task index for task_candidates
            # prompt还是Q2T的prompt
            task_template = self.all_tasks['PRO'][task_candidates[task_idx]]
            assert task_template['task'] == 'PRO'

            if task_template['id'] in ['13-1', '13-2', '13-3']:
                source_text = task_template['source'].format(user_id, country, cate_list_seq, source)
                target_text = task_template['target'].format(query_trigger)
            elif task_template['id'] in ['13-6', ]:
                source_text = task_template['source'].format(query_trigger)
                target_text = task_template['target'].format(1)
                source_text_neg = task_template['source'].format(query_trigger_neg)
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

        # 如果做文本直接生成，需不需要whole word？
        input_ids = self.tokenizer.encode(
            source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
        out_dict['input_ids'] = torch.LongTensor(input_ids)
        if source_text_neg is not None:
            input_ids_neg = self.tokenizer.encode(
                source_text_neg, padding=True, truncation=True, max_length=self.args.max_text_length)
            out_dict['input_ids_neg'] = torch.LongTensor(input_ids_neg)
            out_dict['input_length_neg'] = len(input_ids_neg)

        # 怎么由这个text判断whole word
        tokenized_text = self.tokenizer.tokenize(source_text)
        whole_word_ids = self.calculate_whole_word_ids(tokenized_text, input_ids)
        # 为什么要做这个判断
        assert len(whole_word_ids) == len(input_ids)

        time_ids = self.calculate_time_ids(tokenized_text, input_ids, time_seq, get_time_ids=True if time_seq is not None else False, utdid=user_id)
        assert len(time_ids) == len(input_ids)

        target_ids = self.tokenizer.encode(
            target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

        out_dict['input_length'] = len(input_ids)
        out_dict['whole_word_ids'] = torch.LongTensor(whole_word_ids)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        out_dict['target_length'] = len(target_ids)

        if query_trigger is not None:
            out_dict['query_trigger'] = query_trigger
            query_trigger_ids = self.tokenizer.encode(
                query_trigger, padding=True, truncation=True, max_length=self.args.gen_max_length)
            out_dict['query_trigger_ids'] = torch.LongTensor(query_trigger_ids)
            out_dict['query_trigger_length'] = len(query_trigger_ids)

        if query_trigger_neg is not None:
            out_dict['query_trigger_neg'] = query_trigger_neg
            query_trigger_neg_ids = self.tokenizer.encode(
                query_trigger_neg, padding=True, truncation=True, max_length=self.args.gen_max_length)
            out_dict['query_trigger_neg_ids'] = torch.LongTensor(query_trigger_neg_ids)
            out_dict['query_trigger_neg_length'] = len(query_trigger_neg_ids)

        if time_ids is not None:
            out_dict['time_ids'] = torch.LongTensor(time_ids)

        if user_id is not None:
            out_dict['user_id'] = user_id
        if source is not None:
            out_dict['source'] = source
        if prefix is not None:
            out_dict['prefix'] = prefix
        if country is not None:
            out_dict['county'] = country
        if cate_list_seq is not None:
            out_dict['cate_list_seq'] = cate_list_seq
        if prod_id is not None:
            out_dict['prod_id'] = prod_id
        if prod_title is not None:
            out_dict['prod_title'] = prod_title
        if prod_keywords is not None:
            out_dict['prod_keywords'] = prod_keywords

        if online_imps_querylist is not None:
            out_dict['online_imps_querylist'] = online_imps_querylist
        if online_imps_poslist is not None:
            out_dict['online_imps_poslist'] = online_imps_poslist

        if source_text_neg is not None:
            out_dict['source_text_neg'] = source_text_neg
        if reward_neg is not None:
            out_dict['reward_neg'] = reward_neg
        if reward_pos is not None:
            out_dict['reward_pos'] = reward_pos

        out_dict['source_text'] = source_text
        out_dict['tokenized_text'] = tokenized_text
        out_dict['target_text'] = target_text

        out_dict['task'] = task_template['task']

        out_dict['loss_weight'] = loss_weight

        return out_dict

    # 后续可能要处理成时间序列
    def calculate_whole_word_ids(self, tokenized_text, input_ids):
        whole_word_ids = []
        curr = 0
        for i in range(len(tokenized_text)):
            if tokenized_text[i].startswith('▁'):
                curr += 1
                whole_word_ids.append(curr)
            else:
                whole_word_ids.append(curr)
        last_item = whole_word_ids[len(input_ids) - 2]
        return whole_word_ids[:len(input_ids) - 1] + [0]  ## the added [0] is for </s>

    def calculate_time_ids(self, tokenized_text, input_ids, time_seq, get_time_ids=False, utdid=None):
        # 取值为0～101
        default = 101
        time_ids = [default for _ in range(len(tokenized_text))]

        # 对于不需要time embedding的任务
        if not get_time_ids:
            return time_ids[:len(input_ids) - 1] + [default]  ## for </s>

        time_seq_list = time_seq.split(', ')
        time_seq_idx = 0
        query_flag = False
        # 先找Query category preferences的位置，然后按照','的位置来赋值
        for i in range(len(tokenized_text)):
            # 按照, 划分不同的query
            if query_flag:
                if tokenized_text[i].endswith(','):
                    # u2q数据跑这个会有bug，还没有修
                    if time_seq_idx < len(time_seq_list) - 1:
                        time_seq_idx += 1
                else:
                    time_ids[i] = int(time_seq_list[time_seq_idx])

            # Purchase history:
            if i > 2 and tokenized_text[i] == ':' and tokenized_text[i-1] == '▁history' and tokenized_text[i-2] == '▁Purchase':
                query_flag = True
        # return whole_word_ids[:len(input_ids) - 1] + [0]  ## the added [0] is for </s>
        # if time_seq_idx != len(time_seq_list) - 1:
            # raise Exception('idx != len time seq', time_seq_idx, query_flag, tokenized_text, time_seq, utdid)
        return time_ids[:len(input_ids)-1] + [default]  ## for </s>

    def collate_fn(self, batch):
        batch_entry = {}

        B = len(batch)

        args = self.args

        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        whole_word_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        time_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        loss_weights = torch.ones(B, dtype=torch.float)

        if 'input_ids_neg' in batch[0]:
            S_W_L_neg = max(entry['input_length_neg'] for entry in batch)
            input_ids_neg = torch.ones(B, S_W_L_neg, dtype=torch.long) * self.tokenizer.pad_token_id

        if 'query_trigger' in batch[0]:
            Q_W_L = max(entry['query_trigger_length'] for entry in batch)
            query_trigger_ids = torch.ones(B, Q_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        if 'query_trigger_neg' in batch[0]:
            Q_W_L_neg = max(entry['query_trigger_neg_length'] for entry in batch)
            query_trigger_neg_ids = torch.ones(B, Q_W_L_neg, dtype=torch.long) * self.tokenizer.pad_token_id

        if 'reward_pos' in batch[0]:
            reward_pos = torch.zeros(B, dtype=torch.float)

        if 'reward_pos' in batch[0]:
            reward_neg = torch.zeros(B, dtype=torch.float)

        tasks = []
        source_text = []
        tokenized_text = []
        target_text = []
        source = []
        user_id = []
        prefix = []
        query_trigger = []
        query_trigger_neg = []
        cate_list_seq = []
        country = []
        prod_id = []
        prod_title = []
        prod_keywords = []
        online_imps_poslist = []
        online_imps_querylist = []
        # reward_pos = []
        # reward_neg = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            whole_word_ids[i, :entry['input_length']] = entry['whole_word_ids']
            time_ids[i, :entry['input_length']] = entry['time_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']

            if 'input_ids_neg' in entry:
                input_ids_neg[i, :entry['input_length_neg']] = entry['input_ids_neg']

            if 'task' in entry:
                tasks.append(entry['task'])

            if 'source' in entry:
                source.append(entry['source'])

            if 'prefix' in entry:
                prefix.append(entry['prefix'])

            if 'query_trigger' in entry:
                query_trigger.append(entry['query_trigger'])
                query_trigger_ids[i, :entry['query_trigger_length']] = entry['query_trigger_ids']
            if 'query_trigger_neg' in entry:
                query_trigger_neg.append(entry['query_trigger_neg'])
                query_trigger_neg_ids[i, :entry['query_trigger_neg_length']] = entry['query_trigger_neg_ids']

            if 'reward_pos' in entry:
                reward_pos[i] = float(entry['reward_pos'])
            if 'reward_neg' in entry:
                reward_neg[i] = float(entry['reward_neg'])

            if 'cate_list_seq' in entry:
                cate_list_seq.append(entry['cate_list_seq'])
            if 'county' in entry:
                country.append(entry['county'])

            if 'prod_id' in entry:
                prod_id.append(entry['prod_id'])
            if 'prod_title' in entry:
                prod_title.append(entry['prod_title'])
            if 'prod_keywords' in entry:
                prod_keywords.append(entry['prod_keywords'])

            if 'online_imps_querylist' in entry:
                online_imps_querylist.append(entry['online_imps_querylist'])
            if 'online_imps_poslist' in entry:
                online_imps_poslist.append(entry['online_imps_poslist'])

            if 'user_id' in entry:
                user_id.append(entry['user_id'])

            if 'source_text' in entry:
                source_text.append(entry['source_text'])

            if 'tokenized_text' in entry:
                tokenized_text.append(entry['tokenized_text'])

            if 'target_text' in entry:
                target_text.append(entry['target_text'])

            if 'loss_weight' in entry:
                loss_weights[i] = entry['loss_weight']

        assert 't5' in args.backbone
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['task'] = tasks

        batch_entry['user_id'] = user_id
        batch_entry['source'] = source

        if 'input_ids_neg' in batch[0]:
            batch_entry['input_ids_neg'] = input_ids_neg
        if len(prefix) > 0:
            batch_entry['prefix'] = prefix
        if len(query_trigger) > 0:
            batch_entry['query_trigger'] = query_trigger
            batch_entry['query_trigger_ids'] = query_trigger_ids
        if len(query_trigger_neg) > 0:
            batch_entry['query_trigger_neg'] = query_trigger_neg
            batch_entry['query_trigger_neg_ids'] = query_trigger_neg_ids

        if 'reward_pos' in batch[0]:
            batch_entry['reward_pos'] = reward_pos
        if 'reward_neg' in batch[0]:
            batch_entry['reward_neg'] = reward_neg

        if len(cate_list_seq) > 0:
            batch_entry['cate_list_seq'] = cate_list_seq
        if len(country) > 0:
            batch_entry['county'] = country
        if len(prod_id) > 0:
            batch_entry['prod_id'] = prod_id
        if len(prod_title) > 0:
            batch_entry['prod_title'] = prod_title
        if len(prod_keywords) > 0:
            batch_entry['prod_keywords'] = prod_keywords
        if len(online_imps_querylist) > 0:
            batch_entry['online_imps_querylist'] = online_imps_querylist
        if len(online_imps_poslist) > 0:
            batch_entry['online_imps_poslist'] = online_imps_poslist

        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text

        batch_entry['input_ids'] = input_ids
        batch_entry['whole_word_ids'] = whole_word_ids
        batch_entry['time_ids'] = time_ids
        batch_entry['target_ids'] = target_ids

        batch_entry['loss_weights'] = loss_weights

        return batch_entry


def get_loader(args, task_list, sample_numbers, split='icbu', mode='train',
               batch_size=16, workers=4, distributed=False, multi_dataset=True, datasplitmap=None,
               cold=False, online=False):
    if 't5' in args.backbone:
        tokenizer = P5Tokenizer.from_pretrained(
            args.path,
            max_length=args.max_text_length,
            do_lower_case=args.do_lower_case)
    else:
        raise NotImplementedError

    if split in ['test', 'icbu', 'icbu_text', 'QAC', 'Q2Q', 'I2Q', 'Q2C', 'U2C', 'traditional', 'U2QC', 'Q2T',
                 'Q2TR', 'Q2TC', 'RM', 'PRO'] \
            or split in ['icbu_seqtext', 'icbu_seqtextqac', 'icbu_seqtextqacq2q', 'icbu_u2qqacq2qi2q',
                         'icbu_u2qqacq2qi2qq2c', 'icbu_u2qqacq2qi2qq2cu2c', 'icbu_u2qqacq2qi2qq2ctra',
                         'icbu_u2qqacq2qi2qq2cu2ctra', 'icbu_u2qqacq2qi2qq2cu2cq2ttra',
                         'icbu_u2qqacq2qi2qq2cu2cq2trq2tctra']\
            or split in ['Q2QI2Q', 'icbu_texttra']:
        # from all_icbu_templates import all_tasks as task_templates
        from icbu_templates import all_tasks as task_templates
        dataset = P5_ICBU_Dataset(
            task_templates,
            task_list,
            tokenizer,
            args,
            sample_numbers,
            mode=mode,
            split=split,
            rating_augment=False,
            cold=cold,
            online=online,
        )
    else:
        raise NotImplementedError

    if multi_dataset:
        each_dataset_len = dataset.getLengthAllDataset()
        sampler = DisMultiSetSampler(sample_len_map={each: math.floor(each_dataset_len[each]*datasplitmap[each])
                                                     for each in each_dataset_len},
                                     dataset=dataset, shuffle=True if mode == 'train' else False)
    elif distributed:
        sampler = MyDistributedSampler(dataset, shuffle=True if mode == 'train' else False)
    else:
        sampler = None

    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    return loader
