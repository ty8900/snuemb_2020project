from .base import AbstractDataloader

import torch
import torch.utils.data as data_utils


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
        dataset = SrgnnTrainDataset(self.args, self.dataset, self.train_negative_samples, self.rng, train_ranges)
        return dataset

    def _get_eval_dataset(self, mode):
        positions = self.validation_targets if mode=='val' else self.test_targets
        dataset = SrgnnEvalDataset(self.args, self.dataset, self.test_negative_samples, positions)
        return dataset

class SrgnnTrainDataset(data_utils.Dataset):
    def __init__(self, args, dataset, neg_samples, rng, train_ranges):
        self.args = args
        self.user2dict = dataset['user2dict'] # session dict
        self.users = sorted(self.user2dict.keys())
        self.train_window = args.train_window # 100
        self.max_len = args.max_len
        self.num_users = len(dataset['umap'])
        self.num_items = len(dataset['smap'])
        self.rng = rng
        self.train_ranges = train_ranges

        ## calculate session
        self.index2user_and_offsets = self.populate_indices()

        self.output_timestamps = args.dataloader_output_timestamp
        self.output_days = args.dataloader_output_days
        self.output_user = args.dataloaer_output_user

        ## is it needed?
        self.neg_samples = neg_samples

    def get_rng_state(self):
        return self.rng.getstate()

    def set_rng_state(self, state):
        return self.rng.setstate(state)

    ## maybe session length tokenizing
    def populate_indices(self):
        index2user_and_offsets = {}
        i = 0
        T = self.max_len
        W = self.train_window

        # offset is exclusive
        for user, pos in self.train_ranges:
            if W is None or W == 0:
                offsets = [pos]
            else:
                offsets = list(range(pos, T-1, -W)) # pos ~ T
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
        ## get session beg~end
        ## diff from bert
        ## first, generate n-1 sessions
        inputs, labels = [], []
        max_item_len = len(seq) - 1
        for i in range(1, len(seq)):
            i_label = seq[-i]
            labels += [i_label]
            inputs += [seq[:-i]]

        ## second, pad all inputs and generate mask.
        item_lens = [len(input) for input in inputs]
        inputs = [input + [0] * (max_item_len - le) for input, le in zip(inputs,item_lens)]
        masks = [[1] * le + [0] * (max_item_len - le) for le in item_lens]

        ## third, make graph with inputs
        item_indices, n_node, graph_A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            graph_A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])

        alias_inputs = torch.LongTensor(alias_inputs)
        d = {'tokens': torch.LongTensor(alias_inputs), 'labels': torch.LongTensor(labels), 'masks': torch.LongTensor(masks),
             'graph': torch.LongTensor(graph_A), 'index': torch.LongTensor(item_indices)}

        if self.output_timestamps:
            timestamps = self.user2dict[user]['timestamps']
            timestamps = timestamps[beg:end]
            d['timestamps'] = torch.LongTensor(timestamps)

        if self.output_days:
            days = self.user2dict[user]['days']
            days = days[beg:end]
            d['days'] = torch.LongTensor(days)

        return d

    class SrgnnEvalDataset(data_utils.Dataset):
        def __init__(self, args, dataset, neg_samples, positions):
            self.user2dict = dataset['user2dict']
            self.positions = positions
            self.max_len = args.max_len
            self.num_items = len(dataset['smap'])

            self.output_timestamps = args.dataloader_output_timestamp
            self.output_days = args.dataloader_output_days
            self.output_user = args.dataloaer_output_user

            ## is it needed?
            self.neg_samples = neg_samples

        def __len__(self):
            return len(self.positions)

        def __getitem__(self, index):
            user, pos = self.positions[index]
            seq = self.user2dict[user]['items']

            beg = max(0, pos + 1 - self.max_len)
            end = pos + 1
            seq = seq[beg:end]

            ## answer : the last item (in position)
            labels = [seq[-1]]

            inputs = seq[:-1] + [0]
            masks = [1] * (len(inputs) - 1) + [0]
            n_node = len(inputs)
            node = np.unique(inputs)
            item_indices = (node.tolist())
            u_A = np.zeros((n_node, n_node))
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

            d = {'tokens': torch.LongTensor([alias_inputs]), 'labels': torch.LongTensor([labels]),
                 'masks': torch.LongTensor([masks]),
                 'graph': torch.LongTensor([graph_A]), 'index': torch.LongTensor([item_indices])}

            if self.output_timestamps:
                timestamps = self.user2dict[user]['timestamps']
                timestamps = timestamps[beg:end]
                timestamps = [0] * padding_len + timestamps
                d['timestamps'] = torch.LongTensor(timestamps)

            if self.output_days:
                days = self.user2dict[user]['days']
                days = days[beg:end]
                days = [0] * padding_len + days
                d['days'] = torch.LongTensor(days)

            return d



