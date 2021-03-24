from .base import AbstractDataloader

import torch
import torch.utils.data as data_utils
import numpy as np


class SrgnnDataloader(AbstractDataloader):
    @classmethod
    def code(cls):
        return 'srgnn'

    def _get_dataset(self, mode):
        if mode == 'train':
            return self._get_train_dataset()
        elif mode == 'val':
            return self._get_eval_dataset('val')
        else:
            return self._get_eval_dataset('test')

    def _get_train_dataset(self):
        train_ranges = self.train_targets
        dataset = SrgnnTrainDataset(self.args, self.dataset,
                                    self.train_negative_samples,
                                    self.rng, train_ranges)
        return dataset

    def _get_eval_dataset(self, mode):
        positions = self.validation_targets\
            if mode == 'val' else self.test_targets
        dataset = SrgnnEvalDataset(self.args, self.dataset,
                                   self.test_negative_samples, positions)
        return dataset


class SrgnnTrainDataset(data_utils.Dataset):
    def __init__(self, args, dataset, neg_samples, rng, train_ranges):
        self.args = args
        # session dict, assumption that 1 user_dict = 1 session
        self.user2dict = dataset['user2dict']  
        self.users = sorted(self.user2dict.keys())
        self.train_window = args.train_window  # 100
        
        # max_len determines training sequence max length.
        # if 30, we select 2~30(random) length randomly selected item sequence
        # from 1 user_dict as a 1 session.
        self.max_len = args.max_len
        self.num_users = len(dataset['umap'])
        self.num_items = len(dataset['smap'])
        self.rng = rng
        # we set 1~n vector -> 1~(n-2) training range
        # n-1th : val, nth : test.
        self.train_ranges = train_ranges

        # calculate session
        self.index2user_and_offsets = self.populate_indices()

        self.output_timestamps = args.dataloader_output_timestamp
        self.output_days = args.dataloader_output_days
        self.output_user = args.dataloader_output_user

        # is it needed?
        # self.neg_samples = neg_samples

    def get_rng_state(self):
        return self.rng.getstate()

    def set_rng_state(self, state):
        return self.rng.setstate(state)

    # session length tokenizing
    def populate_indices(self):
        index2user_and_offsets = {}
        i = 0
        T = self.max_len
        W = self.train_window

        # offset is exclusive
        # if W = 0, we use 1 full session as a training set.
        # if W > 0, we use tokenized some sessions from 1 full session as a training set
        # ex) 100 len session, 30 max_len, 10 train window
        # => (1000,100), (1000, 90), ... , (1000, 30). 8 sessions
        for user, pos in self.train_ranges:
            if W is None or W == 0:
                offsets = [pos]
            else:
                offsets = list(range(pos, T-1, -W))  # pos ~ T
                if len(offsets) == 0:
                    offsets = [pos]
            for offset in offsets:
                index2user_and_offsets[i] = (user, offset)
                i += 1
        return index2user_and_offsets

    def __len__(self):
        return len(self.index2user_and_offsets)

    def __getitem__(self, index):
        user, offset = self.index2user_and_offsets[index]
        seq = self.user2dict[user]['items']
        beg = max(0, offset-self.max_len)
        end = offset
        seq = seq[beg:end]
        # get session beg~end 끝에서부터하여 중복 없애기)
        # diff from bert
        # first, generate n-1 sessions and select randolmly one session.
        max_item_len = len(seq)
        curr = np.random.randint(2, max_item_len)  # session limit
        inputs = seq[:curr]
        labels = [seq[curr]]
        # second, pad all inputs and generate mask.
        # to match input dimension. we don't calculate 0s input in actual model.
        item_lens = len(inputs)
        inputs = inputs + [0] * (self.max_len - item_lens)
        masks = [1] * item_lens + [0] * (self.max_len - item_lens)

        # third, make graph with inputs
        node = np.unique(inputs)
        item_indices = node.tolist() + (self.max_len - len(node)) * [0]
        u_A = np.zeros((self.max_len, self.max_len))
        for i in np.arange(len(inputs) - 1):
            if inputs[i + 1] == 0:
                break
            u = np.where(node == inputs[i])[0][0]
            v = np.where(node == inputs[i + 1])[0][0]
            u_A[u][v] = 1
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        graph_A = u_A
        alias_inputs = [np.where(node == i)[0][0] for i in inputs]
        # tokens : graph node에 따른 세션 시퀀스. max_len 길이 벡터.
        # 모델에서 latent vector 생성 시 사용
        # labels : 예측해야할 ground truth, tokens 시퀀스 다음 시점 item
        # masks : max_len길이로 tokens의 padding 지우기 위함.
        # graph : GNN에 넣을 양방향 graph. max_len x 2*max_len
        # item : 실제 item index를 token에 대응. max_len 길이
        d = {'tokens': torch.LongTensor(alias_inputs),
             'labels': torch.LongTensor(labels),
             'masks': torch.LongTensor(masks),
             'graph': torch.FloatTensor(graph_A),
             'item': torch.LongTensor(item_indices)}
        return d


class SrgnnEvalDataset(data_utils.Dataset):
    def __init__(self, args, dataset, neg_samples, positions):
        self.user2dict = dataset['user2dict']
        self.positions = positions
        self.max_len = args.max_len
        self.num_items = len(dataset['smap'])

        self.output_timestamps = args.dataloader_output_timestamp
        self.output_days = args.dataloader_output_days
        self.output_user = args.dataloader_output_user

        self.neg_samples = neg_samples
        
    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        user, pos = self.positions[index]
        seq = self.user2dict[user]['items']

        # 역시 끝에서 부터. 앞서 말한대로 처음에 0~n 세션이면
        # val set은 n-1번째, test set은 n번째가 target pos이다.
        beg = max(0, pos + 1 - self.max_len)
        end = pos + 1
        seq = seq[beg:end]
        # answer : the last item (in position)
        answer = [seq[-1]]

        # make candidate
        candidates = answer + self.neg_samples[user]
        labels = [1] + [0] * len(self.neg_samples[user])

        # we use 5~10 length session to predict test target.
        # 너무 긴 세션으로 예측 시 성능 저하
        curr = np.random.randint(5, 10)
        inputs = seq[-1 - curr:-1] + [0] * (curr + 1)
        masks = [1] * (len(inputs) - curr-1) + [0] * (curr + 1)
        # masking
        n_node = len(inputs)
        inputs = inputs + [0] * (self.max_len - n_node)
        masks = masks + [0] * (self.max_len - n_node)
        
        # make graph (same as in training dataloader above)
        node = np.unique(inputs)
        item_indices = node.tolist() + [0] * (self.max_len - len(node))
        u_A = np.zeros((self.max_len, self.max_len))
        for i in np.arange(len(inputs) - 1):
            if inputs[i + 1] == 0:
                break
            u = np.where(node == inputs[i])[0][0]
            v = np.where(node == inputs[i + 1])[0][0]
            u_A[u][v] = 1
        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()
        graph_A = u_A
        alias_inputs = ([np.where(node == i)[0][0] for i in inputs])

        # train과 의미가 같다.
        d = {'tokens': torch.LongTensor(alias_inputs),
             'labels': torch.LongTensor(labels),
             'masks': torch.LongTensor(masks),
             'graph': torch.FloatTensor(graph_A),
             'item': torch.LongTensor(item_indices),
             'candidates': torch.LongTensor(candidates)}
        return d
