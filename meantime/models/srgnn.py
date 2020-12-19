from meantime.models.base import BaseModel

import torch.nn as nn
from abc import *

class SrgnnBaseModel(BaseModel, metaclass=ABCMeta):
    def __init__(self, args):
        super().__init__(args)
        self.ce = nn.CrossEntropyLoss()

    ## d : batch * dict
    def forward(self, d):
        logits = self.get_logits(d)
        ret = {'logits': logits}
        if self.training:
            labels = d['labels']
            loss = self.get_loss(logits, labels)
            ret['loss'] = loss
            #ret['loss_cnt'] = loss_cnt
        else:
            ret['hit'], ret['mrr'] = self.get_scores(d, logits)
            ret['labels'] = d['labels']
        return ret

    @abstractmethod
    def get_logits(self, d):
        pass

    @abstractmethod
    def get_scores(self, d, logits):  # logits : B x H or M x H, returns B x C or M x V
        pass

    def get_loss(self, logits, labels):
        loss = self.ce(logits, (labels - 1).squeeze()) # why -1?
        return loss