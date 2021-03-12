import torch
import numpy as np


# define recall@20 :
# By top k(=20) prediction, recall@20 = hits / answer#(=batch_size)
def recall(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = []
    for score, target in zip(cut, labels):
        hit.append(np.isin(target - 1, score))
    # hit = labels.gather(1, cut)
    # return (hit.sum(1).float() / labels.sum(1).float()).mean().item()
    return np.mean(hit)


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = []
    for score, target in zip(cut, labels):
        hit.append(1 if np.isin(target - 1, score) else 0)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hit.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(n, k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}
    scores = scores.cpu()
    labels = labels.cpu()
    answer_count = labels.sum(1)
    answer_count_float = answer_count.float()
    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
       cut = cut[:, :k]
       hits = labels_float.gather(1, cut)
       metrics['Recall@%d' % k] = (hits.sum(1) / answer_count_float).mean().item()
       position = torch.arange(2, 2+k)
       weights = 1 / torch.log2(position.float())
       dcg = (hits * weights).sum(1)
       idcg = torch.Tensor([weights[:min(n, k)].sum() for n in answer_count])
       ndcg = (dcg / idcg).mean().item()
       metrics['NDCG@%d' % k] = ndcg

    return metrics


# I used only this function.
# for BERT4Rec, recalls calculated by (hits / label's answer #)
# for SRGNN, recalls calculated by hits' mean, cuz label's answer # always 1.
# for BERT4Rec, ndcg calculated by (dcg / idcg).mean()
# for SRGNN, ndcg calculated by dcg.mean(), cuz idcg always 1. (answer# = 1)
def recalls_and_ndcgs_for_ks_sr(scores, labels, ks):
    metrics = {}

    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
       cut = cut[:, :k]
       hit = []
       hits = []
       for score, target in zip(cut, labels):
           hit.append(np.isin(target - 1, score))
           if np.isin(target - 1, score):
            hits.append(np.where(score == (target - 1))[0][0])
           else:
            hits.append(-1)
       metrics['Recall@%d' % k] = np.mean(hit)
       position = torch.arange(2, 2+k)
       weights = torch.cat([1 / torch.log2(position.float()), torch.FloatTensor([0])])
       dcg = weights[hits]
       ndcg = dcg.mean().item()
       metrics['NDCG@%d' % k] = ndcg

    return metrics