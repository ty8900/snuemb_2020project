from .srgnn import SrgnnBaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class GNN(nn.Module):
    def __init__(self, args):
        super(GNN, self).__init__()
        self.step = args.gnn_step
        self.hidden_units = args.hidden_units
        self.input_size = args.hidden_units * 2
        self.gate_size = args.hidden_units * 3

        # Eq (1) in paper
        self.b_in = nn.Parameter(torch.Tensor(self.hidden_units))
        self.b_out = nn.Parameter(torch.Tensor(self.hidden_units))
        self.linear_in_edge = nn.Linear(self.hidden_units,
                                        self.hidden_units, bias=True)
        self.linear_out_edge = nn.Linear(self.hidden_units,
                                         self.hidden_units, bias=True)
        # Eq (2,3)
        self.b_input = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hidden = nn.Parameter(torch.Tensor(self.gate_size))
        self.w_input = nn.Parameter(torch.Tensor(self.gate_size,
                                                 self.input_size))
        self.w_hidden = nn.Parameter(torch.Tensor(self.gate_size,
                                                  self.hidden_units))

    # A : B * L * 2L(i,o edge) graph
    # hidden : B * L * H       prev hidden (from main net)
    def GNNcell(self, A, hidden):
        # All implementation based on SR-GNN paper's equations.
        # Eq (1) : pass graph into linear layer
        edge_size = A.shape[1]
        input_in = torch.matmul(A[:, :, :edge_size],
                                self.linear_in_edge(hidden)) + self.b_in
        input_out = torch.matmul(A[:, :, edge_size:(2 * edge_size)],
                                 self.linear_out_edge(hidden)) + self.b_out
        input = torch.cat([input_in, input_out], 2)
        # Eq (2,3) : caculate gate states.
        # and get reset / update gate recursively(if cell_size>1)
        gate_input = F.linear(input, self.w_input, self.b_input)
        gate_hidden = F.linear(hidden, self.w_hidden, self.b_hidden)
        i_reset, i_update, i_next = gate_input.chunk(3, 2)
        h_reset, h_update, h_next = gate_hidden.chunk(3, 2)
        reset_gate = torch.sigmoid(i_reset + h_reset)
        update_gate = torch.sigmoid(i_update + h_update)
        # Eq (4,5) : get output / get next hidden
        next_gate = torch.tanh(i_next + reset_gate * h_next)
        next_hidden = next_gate + update_gate * (hidden - next_gate)
        return next_hidden

    # A : B * L * 2L(i,o edge) graph
    # hidden : B * L * H       prev hidden (from main net)
    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNcell(A, hidden)
        return hidden


class SrgnnModel(SrgnnBaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.output_info = args.output_info
        self.hidden_units = args.hidden_units
        self.n_node = args.num_items + 1
        self.batch_size = args.train_batch_size
        self.hybrid_vector = args.hybrid_vector  # if local+global vector. true
        self.embedding = nn.Embedding(self.n_node, self.hidden_units)
        self.gnn = GNN(args)

        self.init_weights()
        self.reset_parameters()

        self.w1 = nn.Linear(self.hidden_units, self.hidden_units, bias=True)
        self.w2 = nn.Linear(self.hidden_units, self.hidden_units, bias=True)
        self.wa = nn.Linear(self.hidden_units, 1, bias=False)
        self.ws = nn.Linear(self.hidden_units * 2,
                            self.hidden_units, bias=True)

    @classmethod
    def code(cls):
        return 'srgnn'

    # from original code, have not test yet
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_units)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    # get logits.
    def get_logits(self, d):
        # d = B x dict('token', 'label', 'mask', 'graph', 'item')
        # make item embedding B x max_len(L) x hidden_size(H)
        hidden = self.embedding(d['item'])
        # get item feature by pass graphs into GNN
        hidden = self.gnn(d['graph'], hidden)

        if torch.LongTensor([-1] * self.hidden_units).cuda() in hidden[:][0]:
            print(d['graph'])
            exit()
        # get latent vector of input
        # token, mask = B x max_len
        token = d['tokens']
        mask = d['masks']
        # translate token session into latent vector by learned item embedding
        # seq_hidden = B x max_len x H
        get_latent = lambda i: hidden[i][token[i]]
        seq_hidden = torch.stack([get_latent(i)
                                  for i in torch.arange(len(token)).long()])

        # caculate session embedding / attention Eq (6)
        # use the last token hidden vector as a local session vector.
        # local_emb = B x H
        local_emb = seq_hidden[torch.arange(mask.shape[0]).long(),
                               torch.sum(mask, 1) - 1]
        # get global session embedding by concat local embeddings
        q1 = self.w1(local_emb).view(local_emb.shape[0], 1,
                                     local_emb.shape[1])
        q2 = self.w2(seq_hidden)
        # alpha = B x L x 1
        # global_emb = B x H (with masking)
        alpha = self.wa(torch.sigmoid(q1 + q2))
        global_emb = torch.sum(
            alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        # local + global vector
        if self.hybrid_vector:
            global_emb = self.ws(torch.cat([global_emb, local_emb], 1))
        vi = self.embedding.weight[1:].transpose(1, 0)

        # get final output B x I(item #), prob of next item
        # train : B x I
        # test : B x C
        if 'candidates' in d.keys():
            c_emb = self.embedding(d['candidates'])
            return (global_emb.unsqueeze(1) * c_emb).sum(-1)
        else:
            return torch.matmul(global_emb, vi)

    # get scores(preds) with hit, mrr
    # not used actually, I used NDCG, Recall where calculated in Trainer, not here
    # for debugging only
    def get_scores(self, d, logits):
        # logits : B x I

        scores = logits.topk(20)[1].cpu()
        targets = d['labels'].cpu()
        masks = d['masks'].cpu()
        hit, mrr = [], []
        for score, target, mask in zip(scores, targets, masks):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
        hit = np.mean(hit) * 100
        mrr = np.mean(mrr) * 100
        return [hit, mrr]
