from .rgcn_model import RGCN
from dgl import mean_nodes
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class GraphClassifier(nn.Module):
    def __init__(self, params, relation2id, onto2id,
                 meta2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id
        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)

        self.rel_emb = nn.Embedding(self.params.num_rels + 1, self.params.inp_dim, sparse=False,
                                    padding_idx=self.params.num_rels)
        self.w_onto2ent = nn.Linear(self.params.onto_emb_dim, self.params.sem_dim)
        self.weight_emb = nn.Embedding(1, self.params.onto_emb_dim, sparse=False)
        self.onto2id = onto2id
        self.meta2id = meta2id
        self.data2id = [self.relation2id, self.onto2id, self.meta2id]
        self.onto_emb = nn.Embedding(self.params.num_ontos + 1, self.params.onto_emb_dim, sparse=False,
                                     padding_idx=self.params.num_ontos)
        self.meta_rel_emb = nn.Embedding(self.params.num_meta_rels, self.params.onto_emb_dim, sparse=False)
        self.sigmoid = nn.Sigmoid()
        self.onto_dropout = nn.Dropout(self.params.nei_onto_dropout)
        self.dropout = nn.Dropout(self.params.dropout)
        self.softmax = nn.Softmax(dim=1)
        # self.w_ent2onto = nn.Linear(self.params.emb_dim, self.params.onto_emb_dim)
        self.w_ent2onto_jk = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim, self.params.onto_emb_dim)
        self.tan = nn.Tanh()

        if self.params.add_ht_emb:
            self.fc_layer = nn.Linear(3 * self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim, 1)
        else:
            self.fc_layer = nn.Linear(self.params.num_gcn_layers * self.params.emb_dim + self.params.rel_emb_dim, 1)

    def init_ent_emb_matrix(self, g):
        """ Entity feature initialization with ontology information """
        ent_onto = g.ndata['onto']
        ent_onto_emb = self.onto_emb(ent_onto)
        weight_emb = self.weight_emb.weight.clone().unsqueeze(2)

        onto_atts = self.softmax(self.onto_dropout(torch.matmul(ent_onto_emb, weight_emb).squeeze(2)))
        onto_feats = torch.matmul(onto_atts.unsqueeze(1), ent_onto_emb).squeeze(1)
        ent_feats = self.sigmoid(self.w_onto2ent(onto_feats))

        g.ndata['init'] = torch.cat([g.ndata['feat'], ent_feats], dim=1)

    def get_onto_emb(self, triples):
        onto_triples = torch.tensor(triples)
        head = self.onto_emb(onto_triples[:, 0])
        tail = self.onto_emb(onto_triples[:, 1])
        meta_relation = self.meta_rel_emb(onto_triples[:, 2])
        score = head + meta_relation - tail
        score = torch.norm(score, p=2, dim=1)
        return score

    def get_mapping_constraint(self, g, head_ids, tail_ids, separate=False):
        if separate:
            sub_g_head_pos = g.ndata['onto_pos'][head_ids]
            ent_types_head = (sub_g_head_pos != self.params.num_ontos).nonzero()
            if ent_types_head.shape[0] != 0:
                ontos_head = torch.tensor([int(sub_g_head_pos[i[0]][i[1]]) for i in ent_types_head])
                # entity_emb_head = g.ndata['h'][head_ids][ent_types_head[:, 0]]
                entity_emb_head = g.ndata['repr'][head_ids][ent_types_head[:, 0]].view(-1,
                                                                                       self.params.num_gcn_layers * self.params.emb_dim)
                onto_emb_head = self.onto_emb(ontos_head)
                # ent2onto_emb_head = self.tan(self.w_ent2onto(entity_emb_head))
                ent2onto_emb_head = self.tan(self.w_ent2onto_jk(entity_emb_head))
                output_pos_head = torch.norm(ent2onto_emb_head - onto_emb_head, p=2, dim=1).view(-1,1)

                sub_g_head_neg = g.ndata['onto_neg'][head_ids]
                ent_types_neg_head = (sub_g_head_neg != self.params.num_ontos).nonzero()
                ontos_neg_head = torch.tensor([int(sub_g_head_neg[i[0]][i[1]]) for i in ent_types_neg_head])
                # entity_emb_neg_head = g.ndata['h'][head_ids][ent_types_neg_head[:, 0]]
                entity_emb_neg_head = g.ndata['repr'][head_ids][ent_types_neg_head[:, 0]].view(-1,
                                                                                       self.params.num_gcn_layers * self.params.emb_dim)
                onto_emb_neg_head = self.onto_emb(ontos_neg_head)
                # ent2onto_emb_neg_head = self.tan(self.w_ent2onto(entity_emb_neg_head))
                ent2onto_emb_neg_head = self.tan(self.w_ent2onto_jk(entity_emb_neg_head))
                output_neg_head = torch.norm(ent2onto_emb_neg_head - onto_emb_neg_head, p=2, dim=1).view(-1,1)
            else:
                output_pos_head = torch.tensor([])
                output_neg_head = torch.tensor([])

            sub_g_tail_pos = g.ndata['onto_pos'][tail_ids]
            ent_types_tail = (sub_g_tail_pos != self.params.num_ontos).nonzero()
            if ent_types_tail.shape[0] != 0:
                ontos_tail = torch.tensor([int(sub_g_tail_pos[i[0]][i[1]]) for i in ent_types_tail])
                # entity_emb_tail = g.ndata['h'][tail_ids][ent_types_tail[:, 0]]
                entity_emb_tail = g.ndata['repr'][tail_ids][ent_types_tail[:, 0]].view(-1,
                                                                                       self.params.num_gcn_layers * self.params.emb_dim)
                onto_emb_tail = self.onto_emb(ontos_tail)
                # ent2onto_emb_tail = self.tan(self.w_ent2onto(entity_emb_tail))
                ent2onto_emb_tail = self.tan(self.w_ent2onto_jk(entity_emb_tail))
                output_pos_tail = torch.norm(ent2onto_emb_tail - onto_emb_tail, p=2, dim=1).view(-1,1)

                sub_g_tail_neg = g.ndata['onto_neg'][tail_ids]
                ent_types_neg_tail = (sub_g_tail_neg != self.params.num_ontos).nonzero()
                ontos_neg_tail = torch.tensor([int(sub_g_tail_neg[i[0]][i[1]]) for i in ent_types_neg_tail])
                # entity_emb_neg_tail = g.ndata['h'][tail_ids][ent_types_neg_tail[:, 0]]
                entity_emb_neg_tail = g.ndata['repr'][tail_ids][ent_types_neg_tail[:, 0]].view(-1,
                                                                                               self.params.num_gcn_layers * self.params.emb_dim)
                onto_emb_neg_tail = self.onto_emb(ontos_neg_tail)
                # ent2onto_emb_neg_tail = self.tan(self.w_ent2onto(entity_emb_neg_tail))
                ent2onto_emb_neg_tail = self.tan(self.w_ent2onto_jk(entity_emb_neg_tail))
                output_neg_tail = torch.norm(ent2onto_emb_neg_tail - onto_emb_neg_tail, p=2, dim=1).view(-1,1)
            else:
                output_pos_tail = torch.tensor([])
                output_neg_tail = torch.tensor([])

            return output_pos_head, output_pos_tail, output_neg_head, output_neg_tail

        batch = len(head_ids)
        ids = torch.cat([head_ids, tail_ids], dim=0)
        sub_g = g.ndata['onto'][ids]
        sub_g_neg = g.ndata['onto_neg'][ids]
        ent_types = (sub_g != self.params.num_ontos).nonzero()
        if ent_types.shape[0] != 0:
            ontos = torch.tensor([int(sub_g[i[0]][i[1]]) for i in ent_types]).to(device=self.params.device)
            ontos_neg = torch.tensor([int(sub_g_neg[i[0]][i[1]]) for i in ent_types]).to(device=self.params.device)
            # entity_emb = g.ndata['h'][ids][ent_types[:, 0]]
            entity_emb = g.ndata['repr'][ids][ent_types[:, 0]].view(-1,
                                                                    self.params.num_gcn_layers * self.params.emb_dim)
            onto_emb = self.onto_emb(ontos)
            onto_emb_neg = self.onto_emb(ontos_neg)
            # ent2onto_emb = self.tan(self.w_ent2onto(entity_emb))
            ent2onto_emb = self.tan(self.w_ent2onto_jk(entity_emb))
            result_pos = torch.norm(ent2onto_emb - onto_emb, p=2, dim=1).squeeze()
            result_neg = torch.norm(ent2onto_emb - onto_emb_neg, p=2, dim=1).squeeze()
            output_pos = torch.tensor(np.zeros([batch])).to(device=self.params.device)
            output_neg = torch.tensor(np.ones([batch]) * self.params.margin3).to(device=self.params.device)
            output_idx = [0 for _ in range(batch)]
            idxs = ent_types[:, 0].cpu().numpy()
            for num in range(len(idxs)):
                idx = idxs[num] % batch
                output_pos[idx] = (output_pos[idx] * output_idx[idx] + result_pos[num]) / (output_idx[idx] + 1)
                output_neg[idx] = (output_neg[idx] * output_idx[idx] + result_neg[num]) / (output_idx[idx] + 1)
                output_idx[idx] += 1
        else:
            output_pos = torch.tensor([]).to(device=self.params.device)
            output_neg = torch.tensor([]).to(device=self.params.device)
            output_idx = []
        return output_pos, output_neg, output_idx

    def forward(self, data, cal_onto=False, cal_type=False, separate=False):
        if cal_onto:
            output = self.get_onto_emb(data)
            return output
        else:
            g, rel_labels = data

            if self.params.init_onto_use:
                self.init_ent_emb_matrix(g)
            else:
                g.ndata['init'] = g.ndata['feat'].clone()
            # r: Embedding of relation
            r = self.rel_emb.weight.clone()

            # Input graph into GNN to get embeddings.
            g.ndata['h'], r_emb_out = self.gnn(g, r)

            out_dim = self.params.num_gcn_layers * self.params.emb_dim
            g_out = mean_nodes(g, 'repr').view(-1, out_dim)  # [n, hidden_dim]

            head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
            head_embs = g.ndata['repr'][head_ids]

            tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
            tail_embs = g.ndata['repr'][tail_ids]

            if self.params.add_ht_emb:
                g_rep = torch.cat([g_out,
                                   head_embs.view(-1, out_dim),
                                   tail_embs.view(-1, out_dim),
                                   F.embedding(rel_labels, r_emb_out, padding_idx=-1)], dim=1)
            else:
                g_rep = torch.cat(
                    [g_out, self.rel_emb(rel_labels)], dim=1)

            self.r_emb_out = r_emb_out

            output = self.fc_layer(g_rep)

            if cal_type:
                if separate:
                    output_pos_head, output_pos_tail, output_neg_head, output_neg_tail = self.get_mapping_constraint(g,
                                                                                                                     head_ids,
                                                                                                                     tail_ids,
                                                                                                                     separate)
                    return output_pos_head, output_pos_tail, output_neg_head, output_neg_tail
                else:
                    output_type_pos, output_type_neg, output_idx = self.get_mapping_constraint(g, head_ids, tail_ids)
                    return output, output_type_pos, output_type_neg, output_idx
            else:
                return output
