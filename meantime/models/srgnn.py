from meantime.models.base import BaseModel

import torch.nn as nn
from abc import *

class SrgnnModel(BaseModel, metaclass=ABCMeta):
    def __init__(self, args):
        super().__init__(args)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, d):
        logits = self.get_logits(d)
        ret = {''}