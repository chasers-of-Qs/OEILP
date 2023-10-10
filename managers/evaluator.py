import os
import numpy as np
import torch
import pdb
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Evaluator():
    def __init__(self, params, graph_classifier, data, is_onto=False):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data
        self.is_onto = is_onto

    def eval(self, save=False):
        pos_scores = []
        pos_labels = []
        neg_scores = []
        neg_labels = []

        pos_type_scores = []
        pos_type_labels = []
        neg_type_scores = []
        neg_type_labels = []

        if self.is_onto:
            dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False,
                                    num_workers=self.params.num_workers, collate_fn=self.params.collate_fn_onto)
        else:
            dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False,
                                    num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in enumerate(dataloader):
                if self.is_onto:
                    data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device_onto(batch,
                                                                                                         self.params.device)
                    score_pos = self.graph_classifier(data_pos, self.is_onto)
                    score_neg = self.graph_classifier(data_neg, self.is_onto)
                    pos_scores += score_pos.detach().cpu().tolist()
                    neg_scores += score_neg.detach().cpu().tolist()
                else:
                    data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch,
                                                                                                    self.params.device)
                    score_pos, score_type_pos, score_type_neg, score_idx = self.graph_classifier(data_pos, cal_type=True)
                    score_neg = self.graph_classifier(data_neg)
                    pos_scores += score_pos.squeeze(1).detach().cpu().tolist()
                    neg_scores += score_neg.squeeze(1).detach().cpu().tolist()

                    if len(score_idx) != 0:
                        pos_scores_list = score_type_pos.detach().cpu().tolist()
                        neg_scores_list = score_type_neg.detach().cpu().tolist()
                        add_pos = []
                        add_neg = []
                        for index in range(len(pos_scores_list)):
                            if pos_scores_list[index] < 1e-6:
                                continue
                            add_pos.append(pos_scores_list[index])
                            add_neg.append(neg_scores_list[index])
                        pos_type_scores += add_pos
                        neg_type_scores += add_neg
                        pos_type_labels += [0] * len(add_pos)
                        neg_type_labels += [1] * len(add_neg)

                pos_labels += targets_pos.tolist()
                neg_labels += targets_neg.tolist()

        auc = metrics.roc_auc_score(pos_labels + neg_labels, pos_scores + neg_scores)
        auc_pr = metrics.average_precision_score(pos_labels + neg_labels, pos_scores + neg_scores)

        if not self.is_onto:
            auc_type = metrics.roc_auc_score(pos_type_labels + neg_type_labels, pos_type_scores + neg_type_scores)
            auc_type_pr = metrics.average_precision_score(pos_type_labels + neg_type_labels,
                                                          pos_type_scores + neg_type_scores)

        if save:
            pos_test_triplets_path = os.path.join(self.params.main_dir,
                                                  'data/{}/{}.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_test_triplets_path) as f:
                pos_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            pos_file_path = os.path.join(self.params.main_dir,
                                         'data/{}/grail_{}_predictions.txt'.format(self.params.dataset,
                                                                                   self.data.file_name))
            with open(pos_file_path, "w") as f:
                for ([s, r, o], score) in zip(pos_triplets, pos_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

            neg_test_triplets_path = os.path.join(self.params.main_dir,
                                                  'data/{}/neg_{}_0.txt'.format(self.params.dataset,
                                                                                self.data.file_name))
            with open(neg_test_triplets_path) as f:
                neg_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            neg_file_path = os.path.join(self.params.main_dir,
                                         'data/{}/grail_neg_{}_{}_predictions.txt'.format(self.params.dataset,
                                                                                          self.data.file_name,
                                                                                          self.params.constrained_neg_prob))
            with open(neg_file_path, "w") as f:
                for ([s, r, o], score) in zip(neg_triplets, neg_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

        if self.is_onto:
            return {'auc': auc, 'auc_pr': auc_pr}
        else:
            return {'auc': auc, 'auc_pr': auc_pr, 'auc_type': auc_type, 'auc_pr_type': auc_type_pr}
