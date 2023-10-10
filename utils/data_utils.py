import os
import pdb
import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt


def plot_rel_dist(adj_list, filename):
    rel_count = []
    for adj in adj_list:
        rel_count.append(adj.count_nonzero())

    fig = plt.figure(figsize=(12, 8))
    plt.plot(rel_count)
    fig.savefig(filename, dpi=fig.dpi)


def process_files(files, onto_files, type_files, saved_data2id=None):
    '''
    files: Dictionary map of file paths to read the triplets from.
    onto_files: Dictionary map of file paths to read the ontology triplets from.
    type_file: Dictionary map of file paths to read the type information from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
    entity2id = {}
    relation2id = {} if saved_data2id is None else saved_data2id[0]

    triplets = {}

    ent = 0
    rel = 0

    for file_type, file_path in files.items():

        data = []
        with open(file_path, 'r', encoding='UTF-8') as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
            if not saved_data2id and triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8),
                                    (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))),
                                   shape=(len(entity2id), len(entity2id))))

    onto2id = {} if saved_data2id is None else saved_data2id[1]
    meta2id = {} if saved_data2id is None else saved_data2id[2]
    triplets_onto = {}

    onto = 0
    meta = 0

    for file_type, file_path in onto_files.items():

        data_onto = []
        with open(file_path, 'r', encoding='UTF-8') as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if not saved_data2id:
                if triplet[0] not in onto2id:
                    onto2id[triplet[0]] = onto
                    onto += 1
                if triplet[2] not in onto2id:
                    onto2id[triplet[2]] = onto
                    onto += 1
                if triplet[1] not in meta2id:
                    meta2id[triplet[1]] = meta
                    meta += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in meta2id and triplet[0] in onto2id and triplet[2] in onto2id:
                data_onto.append([onto2id[triplet[0]], onto2id[triplet[2]], meta2id[triplet[1]]])

        triplets_onto[file_type] = np.array(data_onto)

    id2onto = {v: k for k, v in onto2id.items()}
    id2meta = {v: k for k, v in meta2id.items()}

    # Construct the list of adjacency matrix each corresponding to each meta relation. Note that this is constructed only from the onto data.
    adj_list_onto = []
    for i in range(len(meta2id)):
        idx = np.argwhere(triplets_onto['onto'][:, 2] == i)
        adj_list_onto.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (
            triplets_onto['onto'][:, 0][idx].squeeze(1), triplets_onto['onto'][:, 1][idx].squeeze(1))),
                                        shape=(len(onto2id), len(onto2id))))

    # Establish the mapping relationship between entities and their corresponding concepts. Based on all data.
    entity2onto = {}
    for file_type, file_path in type_files.items():
        with open(file_path, 'r', encoding='UTF-8') as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]
        for triplet in file_data:
            if triplet[0] in entity2id and triplet[2] in onto2id:
                ent_id = entity2id[triplet[0]]
                onto_id = onto2id[triplet[2]]
                if ent_id in entity2onto:
                    if onto_id not in entity2onto[ent_id]:
                        entity2onto[ent_id].append(onto_id)
                else:
                    entity2onto[ent_id] = [onto_id]
    m_e2o = np.ones([len(entity2id), len(id2onto)]) * len(id2onto)
    m_e2o_neg = np.ones([len(entity2id), len(id2onto)]) * len(id2onto)
    for enti, ont in entity2onto.items():
        ont_list = [j for j in range(len(id2onto))]
        for i in ont:
            ont_list.remove(i)
        ont = np.array(ont)
        ont_neg = np.random.choice(ont_list, len(ont), replace=False)
        m_e2o[enti][: ont.shape[0]] = ont
        m_e2o_neg[enti][: ont.shape[0]] = ont_neg
    print("Construct matrix of entity2onto done!")

    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation, adj_list_onto, triplets_onto, onto2id, id2onto, meta2id, id2meta, m_e2o, m_e2o_neg


def save_to_file(directory, file_name, triplets, id2entity, id2relation):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w", encoding='UTF-8') as f:
        for s, o, r in triplets:
            f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')
