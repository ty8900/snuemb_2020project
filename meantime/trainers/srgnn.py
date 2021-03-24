from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks
import matplotlib.pyplot as plt
import numpy as np

class SrgnnTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)

    @classmethod
    def code(cls):
        return 'srgnn'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        d = self.model(batch)

        """ scores = d['logits']
        p = [np.argmax(score) for score in scores.detach().cpu().numpy()]
        plt.plot([p.count(i) for i in range(scores.shape[1])])
        plt.savefig('test/test2.png')
        exit()"""

        loss = d['loss']
        return loss

    def calculate_metrics(self, batch):
        d = self.model(batch)
        scores = d['logits']
        labels = d['labels']

        """p = [np.argmax(score) for score in scores.detach().cpu().numpy()]
        plt.plot([p.count(i) for i in range(scores.shape[1])])
        plt.savefig('test/test_c.png')
        exit()"""
        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)
        return metrics
