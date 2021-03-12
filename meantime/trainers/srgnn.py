from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks_sr


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
        loss = d['loss']
        return loss

    def calculate_metrics(self, batch):
        d = self.model(batch)
        scores = d['logits']
        labels = d['labels']
        metrics = recalls_and_ndcgs_for_ks_sr(scores, labels, self.metric_ks)
        return metrics
