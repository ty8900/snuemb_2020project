from meantime.models.base import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU4recModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        batch_size = args.train_batch_size
        input_size = args.input_size
        hidden = args.hidden_units
        num_layers = args.num_layers
        output_size = args.output_size
        dropout = args.dropout
        embedding_size = args.embedding_size

        self.one_hot_buffer = torch.FloatTensor(batch_size,output_size)
    
        if embedding_size > 0:
            self.embedding = nn.Embedding(input_size, embedding_size)
            self.GRU = nn.GRU(embedding_size, hidden, num_layers, dropout)
        else:
            self.GRU = nn.GRU(input_size, hidden, num_layers, dropout)
        self.ff_layer = nn.linear(hidden, output_size)
        self.ReLU = nn.ReLU()

    @classmethod
    def code(cls):
        return 'gru4rec'

    # input : (B,) current item indices 
    # target : (B,) next item indices
    # logits : (B,C) logits for next item indices
    # hidden : hidden
    def forward(self, input, target):
        if self.embedding_size > 0:
            emb_input = input.unsqueeze(0)
            emb_input = self.embedding(emb_input)
        else:
            emb_input = self.one_hot_encode(input)
            emb_input = emb_input.unsqueeze(0)

        logits, hidden = self.GRU(emb_input, target)
        logits = logits.view(-1,logits.size(-1))
        logits = self.ReLU(self.ff_layer(logits))

        return logits, hidden

    def one_hot_encode(self, x):
        self.one_hot_buffer.zero_()
        index = x.view(-1,1)
        return self.one_hot_buffer.scatter_(1, index, 1)

    def get_loss(self):
        #todo

    def get_scores(self):
        #todo

        
