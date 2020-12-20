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

        ## Eq (1)
        self.b_in = nn.Parameter(torch.Tensor(self.hidden_units))
        self.b_out = nn.Parameter(torch.Tensor(self.hidden_units))
        self.linear_in_edge = nn.Linear(self.hidden_units, self.hidden_units, bias=True)
        self.linear_out_edge = nn.Linear(self.hidden_units, self.hidden_units, bias=True)
        ## Eq (2,3)
        self.b_input = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hidden = nn.Parameter(torch.Tensor(self.gate_size))
        self.w_input = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hidden = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_units))

    # A : batch * d * 2d(i,o edge)
    # H : prev hidden (from main net)
    def GNNcell(self, A, hidden):
        ## Eq (1) : caculate input
        edge_size = A.shape[1]
        input_in = torch.matmul(A[:,:,:edge_size], self.linear_in_edge(hidden)) + self.b_in
        input_out = torch.matmul(A[:, :, edge_size:(2 * edge_size)], self.linear_out_edge(hidden)) + self.b_out
        input = torch.cat([input_in,input_out],2)
        ## Eq (2,3) : caculate gate states. and get reset / update gate
        gate_input = F.linear(input, self.w_input, self.b_input)
        gate_hidden = F.linear(hidden, self.w_hidden, self.b_hidden)
        i_reset, i_update, i_next = gate_input.chunk(3,2)
        h_reset, h_update, h_next = gate_hidden.chunk(3, 2)
        reset_gate = torch.sigmoid(i_reset + h_reset)
        update_gate = torch.sigmoid(i_update + h_update)
        ## Eq (4,5) : get output / get next hidden
        next_gate = torch.tanh(i_next + reset_gate * h_next)
        next_hidden = next_gate + update_gate * (hidden - next_gate)
        return next_hidden

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
        self.hybrid_vector = args.hybrid_vector # if local+global vector. true
        self.embedding = nn.Embedding(self.n_node, self.hidden_units)
        self.gnn = GNN(args)

        self.init_weights()
        self.reset_parameters()

        self.w1 = nn.Linear(self.hidden_units, self.hidden_units, bias=True)
        self.w2 = nn.Linear(self.hidden_units, self.hidden_units, bias=True)
        self.wa = nn.Linear(self.hidden_units, 1, bias=False)
        self.ws = nn.Linear(self.hidden_units * 2, self.hidden_units, bias=True)

    @classmethod
    def code(cls):
        return 'srgnn'

    ## from original code, have not test yet
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_units)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def get_logits(self, d):
        ## make embedding
        hidden = self.embedding(d['item'])
        hidden = self.gnn(d['graph'], hidden) ## batch, len, hidden_size

        if torch.LongTensor([-1] * self.hidden_units).cuda() in hidden[:][0] :
            print(d['graph'])
            exit()
        ## get latent vector of input
        token = d['tokens']
        mask = d['masks']
        get_latent = lambda i: hidden[i][token[i]]
        seq_hidden = torch.stack([get_latent(i) for i in torch.arange(len(token)).long()])
        ## caculate session embedding/ attention Eq (6)
        local_emb = seq_hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask,1) - 1]
        q1 = self.w1(local_emb).view(local_emb.shape[0], 1, local_emb.shape[1])
        q2 = self.w2(seq_hidden)
        alpha = self.wa(torch.sigmoid(q1 + q2))
        global_emb = torch.sum(alpha * hidden * mask.view(mask.shape[0],-1,1).float(), 1)
        ## local + global vector
        if self.hybrid_vector:
            global_emb = self.ws(torch.cat([global_emb,local_emb], 1))
        ## get final output
        vi = self.embedding.weight[1:].transpose(1,0)

        return torch.matmul(global_emb, vi)



