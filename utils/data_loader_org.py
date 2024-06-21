import pdb
import random
import warnings
from collections import defaultdict
from time import time

import networkx as nx
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from itertools import combinations


# warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
n_entities = 0
n_relations = 0
n_nodes = 0
#defaultdict: global
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)


def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    pos = []
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]

        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        pos.extend(pos_ids)

        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
    pos=set(pos)

    maxs = max(pos)
    lens = len(pos)
    return np.array(inter_mat)


def remap_item(train_data, test_data):
    def _id_range(train_mat, test_mat, idx):
        min_id = min(min(train_mat[:, idx]), min(test_mat[:, idx]))
        max_id = max(max(train_mat[:, idx]), max(test_mat[:, idx]))


        n_id = max_id - min_id + 1
        return (min_id, max_id), n_id

    global n_users, n_items, item_range
    n_users = max(max(train_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(test_data[:, 1])) + 1
    item_range, n_items = _id_range(train_data, test_data, idx=1)
    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))


def read_triplets(file_name):
    global n_entities, n_relations, n_nodes, n_users, n_items

    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)  # can_triplets_np is data read

    can_triplets_np = np.unique(can_triplets_np, axis=0)

    if args.inverse_r:
        # get triplets with inverse direction like <entity, is-aspect-of, item>

        inv_triplets_np = can_triplets_np.copy()
        inv_triplets_np[:, 0] = can_triplets_np[:, 2]
        inv_triplets_np[:, 2] = can_triplets_np[:, 0]


        inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
        # consider two additional relations --- 'interact' and 'be interacted'
        # why add one to every relation
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
        # get full version of knowledge graph
        triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)
    else:
        # consider two additional relations --- 'interact'.
        can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
        triplets = can_triplets_np.copy()

    en1 = set(sorted(triplets[:, 0]))
    en2 = set(sorted(triplets[:, 2]))
    res = en1.union(en2)
    x = len(res)
    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets


def build_graph(train_data, triplets):
    ckg_graph = nx.MultiDiGraph()
    rd = defaultdict(list)

    print("Begin to load interaction triples ...")
    for u_id, i_id in tqdm(train_data, ascii=True):
        rd[0].append([u_id, i_id])  # 0 stands for interaction
        # why interaction is not added to ckg_graph

    print("\nBegin to load knowledge graph triples ...")
    for h_id, r_id, t_id in tqdm(triplets, ascii=True):
        ckg_graph.add_edge(h_id, t_id, key=r_id)
        rd[r_id].append([h_id, t_id])
        #KEY is unique

    return ckg_graph, rd


def build_sparse_relational_graph(relation_dict):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))  # sum each row

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        # bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []
    print("Begin to build sparse relation matrix ...")
    for r_id in tqdm(relation_dict.keys()):
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:
            cf = np_mat.copy()
            cf[:,1] = cf[:,1] + n_users  # [0, n_items) -> [n_users, n_users+n_items) !!!!!
            #vals: long list
            vals = [1.] * len(cf)
            adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])),shape=(n_nodes, n_nodes))
        else:
            vals = [1.] * len(np_mat)
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])),shape=(n_nodes, n_nodes))
        adj_mat_list.append(adj)

    norm_mat_list = [_bi_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
    # interaction: user->item, [n_users, n_entities]
    norm_mat_list[0] = norm_mat_list[0].tocsr()[:n_users, n_users:].tocoo()
    mean_mat_list[0] = mean_mat_list[0].tocsr()[:n_users, n_users:].tocoo()

    return adj_mat_list, norm_mat_list, mean_mat_list


def load_data_org_top_bottom(model_args):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'
    print('reading train and test user-item set ...')
    # ?? the meanning of tops and bottoms
    # train,test.txt with tops and bottoms only: train_remap_reorg.txt, test_remap_reorg.txt
    # train,test.txt with all types of items: train.txt, test.txt
    # tuple of (uid, item_id)


    train_cf = read_cf(directory + 'train_remap_reorg.txt')
    test_cf = read_cf(directory + 'test_remap_reorg.txt')
    # train_cf = read_cf(directory + 'train.txt')
    # test_cf = read_cf(directory + 'test.txt')

    remap_item(train_cf, test_cf)

    print('combinating train_cf and kg data ...')
    # kg with tops and bottoms: kg_final_top_bottom_remap.txt
    triplets = read_triplets(directory + 'kg_final_top_bottom_remap.txt')
    # kg with all types of items: kg_final.txt
    # triplets = read_triplets(directory + 'kg_final.txt')

    if args.load_rules:
        print("load rules as triplets")
        if args.which_rule == 1:
            triplets = load_rules1(
                directory, '11Aug_outfit_id_with_rules_chinese.txt', 'entity_list.txt', triplets)
        elif args.which_rule == 2:
            triplets = load_rules2(
                directory, '11Aug_outfit_id_with_rules_chinese.txt', 'entity_list.txt', triplets)
        elif args.which_rule == 3:
            triplets = load_rules3(
                directory, '11Aug_outfit_id_with_rules_chinese.txt', 'entity_list.txt', triplets)
        elif args.which_rule == 4:
            triplets = load_rules4(
                directory, '11Aug_outfit_id_with_rules_chinese.txt', 'entity_list.txt', triplets)
        elif args.which_rule == 5:
            triplets = load_rules5(
                directory, '11Aug_outfit_id_with_rules_chinese.txt', 'entity_list.txt', triplets)
        elif args.which_rule == 6:
            triplets = load_rules6(
                directory, '11Aug_outfit_id_with_rules_chinese.txt', 'OutfitRealid_2_entityID_remap.txt', triplets)
        elif args.which_rule == 7:
            triplets = load_rules7(
                directory, '11Aug_outfit_id_with_rules_chinese.txt', 'OutfitRealid_2_entityID_remap.txt', triplets)
        elif args.which_rule == 8:
            triplets = load_rules8(
                directory, '11Aug_outfit_id_with_rules_chinese.txt', 'OutfitRealid_2_entityID_remap.txt', triplets)
        elif args.which_rule == 9:
            triplets = test(
                directory, '11Aug_outfit_id_with_rules_chinese.txt', 'OutfitRealid_2_entityID_remap.txt', triplets)
        elif args.which_rule == 10:
            triplets = load_rules10(
                directory, '11Aug_outfit_id_with_rules_chinese.txt', 'OutfitRealid_2_entityID_remap.txt', triplets)
        elif args.which_rule == 11:
            triplets = load_rules11(
                directory, '11Aug_outfit_id_with_rules_chinese.txt', 'OutfitRealid_2_entityID_remap.txt', triplets)
        elif args.which_rule == 12:
            triplets = load_rules12(
                directory, '11Aug_outfit_id_with_rules_top500.txt', 'OutfitRealid_2_entityID_remap.txt', triplets)
        elif args.which_rule == 13:
            triplets = load_rules13(
                directory, '11Aug_outfit_id_with_rules_chinese.txt', 'OutfitRealid_2_entityID_remap.txt', triplets)
        elif args.which_rule == 14:
            triplets = load_rules14(
                directory, '11Aug_outfit_id_with_rules_chinese.txt', 'OutfitRealid_2_entityID_remap.txt', triplets)
        elif args.which_rule == 15:
            triplets = load_rules15(
                directory, '11Aug_outfit_id_with_rules_chinese.txt', 'OutfitRealid_2_entityID_remap.txt', triplets)
        elif args.which_rule == 16:
            triplets = load_rules16(
                directory, '11Aug_outfit_id_with_rules_chinese.txt', 'OutfitRealid_2_entityID_remap.txt', triplets)
        elif args.which_rule == 17:
            triplets = load_rules17(
                directory, '11Aug_outfit_id_with_rules_chinese.txt', 'OutfitRealid_2_entityID_remap.txt', triplets)
        elif args.which_rule == 18:
            triplets = load_rules18(
                directory, '11Aug_outfit_id_with_rules_chinese.txt', 'OutfitRealid_2_entityID_remap.txt', triplets)
        elif args.which_rule == 19:
            triplets = load_rules19(
                directory, '11Aug_outfit_id_with_rules_chinese.txt', 'OutfitRealid_2_entityID_remap.txt', triplets)
        elif args.which_rule == 20:
            triplets = load_rules20(
                directory, '11Aug_outfit_id_with_rules_chinese.txt', 'OutfitRealid_2_entityID_remap.txt', triplets)
        elif args.which_rule == 21:
            triplets = load_rules21(
                directory, '11Aug_outfit_id_with_rules_chinese.txt', 'OutfitRealid_2_entityID_remap.txt', triplets)
    print('building the graph ...')
    # relation_dict is relation to a list of (head, tail)
    graph, relation_dict = build_graph(train_cf, triplets)

    print('building the adj mat ...')
    adj_mat_list, norm_mat_list, mean_mat_list = build_sparse_relational_graph(
        relation_dict)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations),
        "item_range": item_range,
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }

    return train_cf, test_cf, user_dict, n_params, graph, [
        adj_mat_list, norm_mat_list, mean_mat_list
    ]


def load_rules1(directory, rule_file, entity_file, triplets):
    """load rules as triplets, inverse is not created"""
    rule_file = directory + rule_file
    entity_file = directory + entity_file
    # create uuid to id dictionary for later lookup
    entities_uuid2id = {}
    with open(entity_file) as f:
        skip_first = True
        # skip org_id and remap_id
        for line in f:
            if skip_first:
                skip_first = False
            else:
                entity_uuid, entity_id = line.strip().split()
                entities_uuid2id[entity_uuid] = entity_id
    # get current maximum node id and relation id
    #0,2 node 1 :relation
    maximum_node_id = max(max(triplets[:, 0]), max(triplets[:, 2]))
    maximum_relation_id = max(triplets[:, 1])
    # load in rules as triplets
    cur_node_id = maximum_node_id + 1
    cur_relation_id = maximum_relation_id + 1
    tail_id_remap_table = {}
    new_triplets = []
    with open(rule_file) as f:
        for line in f:
            head_uuid, tail_ids = line.split(':')
            if head_uuid in entities_uuid2id.keys():
                head_id = entities_uuid2id[head_uuid]
                tail_ids = eval(tail_ids)
                for tail_id in tail_ids:
                    if tail_id not in tail_id_remap_table:
                        tail_id_remap_table[tail_id] = cur_node_id
                        cur_node_id += 1
                    remapped_tail_id = tail_id_remap_table[tail_id]
                    new_triplets.append(
                        (head_id, cur_relation_id, remapped_tail_id))
    can_triplets_np = np.array(new_triplets, dtype=np.int32)

    can_triplets_np = np.row_stack([triplets, can_triplets_np])

    inv_triplets_np = can_triplets_np.copy()
    inv_triplets_np[:, 0] = can_triplets_np[:, 2]
    inv_triplets_np[:, 2] = can_triplets_np[:, 0]
    inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(
        can_triplets_np[:, 1]) + 1
    # consider two additional relations --- 'interact' and 'be interacted'
    # why add one to every relation
    can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
    inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
    # get full version of knowledge graph

    triplets = np.row_stack([can_triplets_np, inv_triplets_np])



    # update global variables
    global n_entities, n_nodes, n_relations
    n_entities = max(max(triplets[:, 0]), max(
        triplets[:, 2])) + 1
    n_nodes = n_entities + n_users + n_items
    n_relations = max(triplets[:, 1]) + 1

    return triplets


def load_rules2(directory, rule_file, entity_file, triplets):
    """load rules as triplets, inverse is not created"""
    rule_file = directory + rule_file
    entity_file = directory + entity_file
    # create uuid to id dictionary for later lookup
    entities_uuid2id = {}
    with open(entity_file) as f:
        skip_first = True
        for line in f:
            if skip_first:
                skip_first = False
            else:
                entity_uuid, entity_id = line.strip().split()
                entities_uuid2id[entity_uuid] = entity_id
    # get current maximum node id and relation id
    maximum_node_id = max(max(triplets[:, 0]), max(triplets[:, 2]))
    maximum_relation_id = max(triplets[:, 1])
    # load in rules as triplets
    cur_node_id = maximum_node_id + 1
    cur_relation_id = maximum_relation_id + 1
    tail_id_remap_table = {}
    new_triplets = []
    with open(rule_file) as f:
        for line in f:
            head_uuid, tail_ids = line.split(':')
            if head_uuid in entities_uuid2id.keys():
                head_id = entities_uuid2id[head_uuid]
                tail_ids = eval(tail_ids)
                for tail_id in tail_ids:
                    if tail_id not in tail_id_remap_table:
                        tail_id_remap_table[tail_id] = cur_node_id
                        cur_node_id += 1
                    remapped_tail_id = tail_id_remap_table[tail_id]
                    new_triplets.append(
                        (head_id, cur_relation_id, remapped_tail_id))
    new_triplets_array = np.array(new_triplets, dtype=np.int32)
    triplets = np.row_stack([triplets, new_triplets_array])
    # update global variables
    global n_entities, n_nodes, n_relations
    n_entities = max(max(triplets[:, 0]), max(
        triplets[:, 2])) + 1
    n_nodes = n_entities + n_users+ n_items
    n_relations = max(triplets[:, 1]) + 1

    return triplets

def load_rules3(directory, rule_file, entity_file, triplets):
    """load rules as triplets, inverse is not created"""
    rule_file = directory + rule_file
    entity_file = directory + entity_file
    # create uuid to id dictionary for later lookup
    entities_uuid2id = {}
    with open(entity_file) as f:
        skip_first = True
        for line in f:
            if skip_first:
                skip_first = False
            else:
                entity_uuid, entity_id = line.strip().split()
                entities_uuid2id[entity_uuid] = entity_id
    # get current maximum node id and relation id
    maximum_node_id = max(max(triplets[:, 0]), max(triplets[:, 2]))
    maximum_relation_id = max(triplets[:, 1])
    # load in rules as triplets
    cur_node_id = maximum_node_id + 1
    cur_relation_id = maximum_relation_id + 1
    tail_id_remap_table = {}
    new_triplets = []
    with open(rule_file) as f:
        for line in f:
            head_uuid, tail_ids = line.split(':')
            if head_uuid in entities_uuid2id.keys():
                head_id = entities_uuid2id[head_uuid]
                tail_ids = eval(tail_ids)
                for tail_id in tail_ids:
                    if tail_id not in tail_id_remap_table:
                        tail_id_remap_table[tail_id] = cur_node_id
                        cur_node_id += 1
                    remapped_tail_id = tail_id_remap_table[tail_id]
                    new_triplets.append(
                        (head_id, cur_relation_id, remapped_tail_id))


    new_triplets = np.array(new_triplets, dtype=np.int32)
    new_triplets = inverse(new_triplets)
    triplets = inverse(triplets)
    # get full version of knowledge graph

    triplets = np.row_stack([triplets, new_triplets])

    # update global variables

    global n_entities, n_nodes, n_relations
    n_entities = max(max(triplets[:, 0]), max(
        triplets[:, 2])) + 1
    n_nodes = n_entities + n_users + n_items
    n_relations = max(triplets[:, 1]) + 1

    return triplets




def load_rules4(directory, rule_file, entity_file, triplets):
    """load rules as triplets, inverse is not created"""
    rule_file = directory + rule_file
    entity_file = directory + entity_file
    # create uuid to id dictionary for later lookup
    entities_uuid2id = {}
    with open(entity_file) as f:
        skip_first = True
        for line in f:
            if skip_first:
                skip_first = False
            else:
                entity_uuid, entity_id = line.strip().split()
                entities_uuid2id[entity_uuid] = entity_id
    # get current maximum node id and relation id
    maximum_node_id = max(max(triplets[:, 0]), max(triplets[:, 2]))
    maximum_relation_id = max(triplets[:, 1])
    # load in rules as triplets
    cur_node_id = maximum_node_id + 1
    cur_relation_id = maximum_relation_id + 1
    tail_id_remap_table = {}
    new_triplets = []
    with open(rule_file) as f:
        for line in f:
            head_uuid, tail_ids = line.split(':')
            if head_uuid in entities_uuid2id.keys():
                head_id = entities_uuid2id[head_uuid]
                tail_ids = eval(tail_ids)
                for tail_id in tail_ids:
                    if tail_id not in tail_id_remap_table:
                        tail_id_remap_table[tail_id] = cur_node_id
                        cur_node_id += 1
                    remapped_tail_id = tail_id_remap_table[tail_id]
                    new_triplets.append(
                        (head_id, cur_relation_id, remapped_tail_id))
                cur_relation_id +=1
    new_triplets_array = np.array(new_triplets, dtype=np.int32)
    triplets = np.row_stack([triplets, new_triplets_array])
    # update global variables
    global n_entities, n_nodes, n_relations
    n_entities = max(max(triplets[:, 0]), max(
        triplets[:, 2])) + 1
    n_nodes = n_entities + n_users + n_items
    n_relations = max(triplets[:, 1]) + 1

    return triplets




def load_rules5(directory, rule_file, entity_file, triplets):
    """load rules as triplets, inverse is not created"""
    rule_file = directory + rule_file
    entity_file = directory + entity_file
    # create uuid to id dictionary for later lookup
    entities_uuid2id = {}
    with open(entity_file) as f:
        skip_first = True
        # skip org_id and remap_id
        for line in f:
            if skip_first:
                skip_first = False
            else:
                entity_uuid, entity_id = line.strip().split()
                entities_uuid2id[entity_uuid] = entity_id
    # get current maximum node id and relation id
    #0,2 node 1 :relation
    maximum_node_id = max(max(triplets[:, 0]), max(triplets[:, 2]))
    maximum_relation_id = max(triplets[:, 1])
    # load in rules as triplets
    cur_node_id = maximum_node_id + 1
    cur_relation_id = maximum_relation_id + 1
    tail_id_remap_table = {}
    new_triplets = []
    with open(rule_file) as f:
        for line in f:
            head_uuid, tail_ids = line.split(':')
            if head_uuid in entities_uuid2id.keys():
                head_id = entities_uuid2id[head_uuid]
                tail_ids = eval(tail_ids)
                for tail_id in tail_ids:
                    if tail_id not in tail_id_remap_table:
                        tail_id_remap_table[tail_id] = cur_node_id
                        cur_node_id += 1
                    remapped_tail_id = tail_id_remap_table[tail_id]
                    new_triplets.append(
                        (head_id, cur_relation_id, remapped_tail_id))
    can_triplets_np = np.array(new_triplets, dtype=np.int32)

    can_triplets_np = np.row_stack([triplets, can_triplets_np])

    inv_triplets_np = can_triplets_np.copy()
    inv_triplets_np[:, 0] = can_triplets_np[:, 2]
    inv_triplets_np[:, 2] = can_triplets_np[:, 0]
    inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(
        can_triplets_np[:, 1]) + 1
    # consider two additional relations --- 'interact' and 'be interacted'
    # why add one to every relation
    can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
    inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
    # get full version of knowledge graph

    triplets = np.row_stack([can_triplets_np, inv_triplets_np])
    np.random.shuffle(triplets)


    # update global variables
    global n_entities, n_nodes, n_relations
    n_entities = max(max(triplets[:, 0]), max(
        triplets[:, 2])) + 1
    n_nodes = n_entities + n_users + n_items
    n_relations = max(triplets[:, 1]) + 1

    return triplets





def load_rules6(directory, rule_file, entity_file, triplets):
    """load rules as triplets, inverse is not created"""
    rule_file = directory + rule_file
    entity_file = directory + entity_file
    # create uuid to id dictionary for later lookup
    entities_uuid2id = {}
    with open(entity_file) as f:
        skip_first = True
        for line in f:
            if skip_first:
                skip_first = False
            else:
                org_id = line.strip().split(' ')[0]
                remap_id = line.strip().split(' ')[2]
                entities_uuid2id[org_id] = remap_id
    # get current maximum node id and relation id
    maximum_node_id = max(max(triplets[:, 0]), max(triplets[:, 2]))
    maximum_relation_id = max(triplets[:, 1])
    # load in rules as triplets
    cur_node_id = maximum_node_id + 1
    cur_relation_id = maximum_relation_id + 1
    tail_id_remap_table = {}
    new_triplets = []
    with open(rule_file) as f:
        for line in f:
            head_uuid, tail_ids = line.split(':')
            if head_uuid in entities_uuid2id.keys():
                head_id = entities_uuid2id[head_uuid]
                tail_ids = eval(tail_ids)
                for tail_id in tail_ids:
                    if tail_id not in tail_id_remap_table:
                        tail_id_remap_table[tail_id] = cur_node_id
                        cur_node_id += 1
                    remapped_tail_id = tail_id_remap_table[tail_id]
                    new_triplets.append((head_id, cur_relation_id, remapped_tail_id))
    new_triplets_array = np.array(new_triplets, dtype=np.int32)
    triplets = np.row_stack([triplets, new_triplets_array])
    # update global variables
    en1 = set(sorted(triplets[:, 0]))
    en2 = set(sorted(triplets[:, 2]))
    res = en1.union(en2)
    x = len(res)
    global n_entities, n_nodes, n_relations
    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets




def load_rules7(directory, rule_file, entity_file, triplets):
    """load rules as triplets, inverse is not created"""
    rule_file = directory + rule_file
    entity_file = directory + entity_file
    # create uuid to id dictionary for later lookup
    entities_uuid2id = {}
    with open(entity_file) as f:
        skip_first = True
        for line in f:
            if skip_first:
                skip_first = False
            else:
                org_id = line.strip().split(' ')[0]
                remap_id = line.strip().split(' ')[2]
                entities_uuid2id[org_id] = remap_id
    # get current maximum node id and relation id
    maximum_node_id = max(max(triplets[:, 0]), max(triplets[:, 2]))
    maximum_relation_id = max(triplets[:, 1])
    # load in rules as triplets
    cur_node_id = maximum_node_id + 1
    cur_relation_id = maximum_relation_id + 1
    tail_id_remap_table = {}
    new_triplets = []
    with open(rule_file) as f:
        for line in f:
            head_uuid, tail_ids = line.split(':')
            if head_uuid in entities_uuid2id.keys():
                head_id = entities_uuid2id[head_uuid]
                tail_ids = eval(tail_ids)
                for tail_id in tail_ids:
                    if tail_id not in tail_id_remap_table:
                        tail_id_remap_table[tail_id] = cur_node_id
                        cur_node_id += 1
                    remapped_tail_id = tail_id_remap_table[tail_id]
                    new_triplets.append(
                        (head_id, cur_relation_id, remapped_tail_id))

    can_triplets_np = np.array(new_triplets, dtype=np.int32)

    can_triplets_np = np.row_stack([triplets, can_triplets_np])

    inv_triplets_np = can_triplets_np.copy()
    inv_triplets_np[:, 0] = can_triplets_np[:, 2]
    inv_triplets_np[:, 2] = can_triplets_np[:, 0]
    inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(
        can_triplets_np[:, 1]) + 1
    # consider two additional relations --- 'interact' and 'be interacted'
    # why add one to every relation
    can_triplets_np[:, 1] = can_triplets_np[:, 1] + 1
    inv_triplets_np[:, 1] = inv_triplets_np[:, 1] + 1
    # get full version of knowledge graph
    triplets = np.row_stack([can_triplets_np, inv_triplets_np])

    # update global variables
    global n_entities, n_nodes, n_relations
    n_entities = max(max(triplets[:, 0]), max(
        triplets[:, 2])) + 1
    n_nodes = n_entities + n_items + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets




def load_rules8(directory, rule_file, entity_file, triplets):
    """load rules as triplets, inverse is not created"""
    rule_file = directory + rule_file
    entity_file = directory + entity_file
    # create uuid to id dictionary for later lookup
    entities_uuid2id = {}
    with open(entity_file) as f:
        skip_first = True
        for line in f:
            if skip_first:
                skip_first = False
            else:
                org_id = line.strip().split(' ')[0]
                remap_id = line.strip().split(' ')[2]
                entities_uuid2id[org_id] = remap_id
    # get current maximum node id and relation id
    maximum_node_id = max(max(triplets[:, 0]), max(triplets[:, 2]))
    maximum_relation_id = max(triplets[:, 1])
    # load in rules as triplets
    cur_node_id = maximum_node_id + 1
    cur_relation_id = maximum_relation_id + 1
    tail_id_remap_table = {}
    new_triplets = []
    with open(rule_file) as f:
        for line in f:
            head_uuid, tail_ids = line.split(':')
            if head_uuid in entities_uuid2id.keys():
                head_id = entities_uuid2id[head_uuid]
                tail_ids = eval(tail_ids)
                for tail_id in tail_ids:
                    if tail_id not in tail_id_remap_table:
                        tail_id_remap_table[tail_id] = cur_node_id
                        cur_node_id += 1
                    remapped_tail_id = tail_id_remap_table[tail_id]
                    new_triplets.append(
                        (head_id, cur_relation_id, remapped_tail_id))
    new_triplets_array = np.array(new_triplets, dtype=np.int32)
    triplets = np.row_stack([triplets, new_triplets_array])
    # update global variables
    global n_entities, n_nodes, n_relations
    n_entities = max(max(triplets[:, 0]), max(
        triplets[:, 2])) + 1
    n_nodes = n_entities
    n_relations = max(triplets[:, 1]) + 1

    return triplets





def test(directory, rule_file, entity_file, triplets):
    """load rules as triplets, inverse is not created"""
    rule_file = directory + rule_file
    entity_file = directory + entity_file
    # create uuid to id dictionary for later lookup
    entities_uuid2id = {}
    max_tail = 0
    min_tail = 0
    with open(entity_file) as f:
        skip_first = True
        for line in f:
            if skip_first:
                skip_first = False
            else:
                org_id = line.strip().split(' ')[0]
                remap_id = line.strip().split(' ')[2]
                entities_uuid2id[org_id] = remap_id
    # get current maximum node id and relation id
    maximum_node_id = max(max(triplets[:, 0]), max(triplets[:, 2]))
    maximum_relation_id = max(triplets[:, 1])
    # load in rules as triplets
    cur_node_id = maximum_node_id + 1
    cur_relation_id = maximum_relation_id + 1
    tail_id_remap_table = {}
    new_triplets = []
    fid_write = open('./repeat' + '.txt', 'w')
    with open(rule_file) as f:
        for line in f:
            head_uuid, tail_ids = line.split(':')
            if head_uuid in entities_uuid2id.keys():
                head_id = entities_uuid2id[head_uuid]
                tail_ids = eval(tail_ids)


                fid_write.write('{},{}\n'.format(head_uuid,tail_ids))
                fid_write.flush()
                print('successfully writen')
                for tail_id in tail_ids:
                    # if max_tail < tail_id:
                    #     max_tail = tail_id
                    #     x = tail_ids
                    # if min_tail > tail_id:
                    #     min_tail = tail_id
                    if tail_id not in tail_id_remap_table:
                        tail_id_remap_table[tail_id] = cur_node_id
                        cur_node_id += 1
                    remapped_tail_id = tail_id_remap_table[tail_id]
                    new_triplets.append(
                        (head_id, cur_relation_id, remapped_tail_id))

        fid_write.close()

    new_triplets_array = np.array(new_triplets, dtype=np.int32)
    triplets = np.row_stack([triplets, new_triplets_array])
    # update global variables
    global n_entities, n_nodes, n_relations
    n_entities = max(max(triplets[:, 0]), max(
        triplets[:, 2])) + 1
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets



def load_rules10(directory, rule_file, entity_file, triplets):
    """load rules as triplets, inverse is not created"""
    rule_file = directory + rule_file
    entity_file = directory + entity_file
    # create uuid to id dictionary for later lookup
    entities_uuid2id = {}
    with open(entity_file) as f:
        skip_first = True
        for line in f:
            if skip_first:
                skip_first = False
            else:
                org_id = line.strip().split(' ')[0]
                remap_id = line.strip().split(' ')[2]
                entities_uuid2id[org_id] = remap_id
    # get current maximum node id and relation id
    maximum_node_id = max(max(triplets[:, 0]), max(triplets[:, 2]))
    maximum_relation_id = max(triplets[:, 1])
    # load in rules as triplets
    cur_node_id = maximum_node_id + 1
    cur_relation_id = maximum_relation_id + 1
    cur_relation_id_toge = cur_relation_id + 1
    tail_id_remap_table = {}
    new_triplets = []

    with open(rule_file) as f:
        for line in f:
            head_uuid, tail_ids = line.split(':')
            if head_uuid in entities_uuid2id.keys():
                head_id = entities_uuid2id[head_uuid]
                tail_ids = eval(tail_ids)
                for tail_id in tail_ids:
                    if tail_id not in tail_id_remap_table:
                        tail_id_remap_table[tail_id] = cur_node_id
                        cur_node_id += 1
                    remapped_tail_id = tail_id_remap_table[tail_id]
                    new_triplets.append((head_id, cur_relation_id, remapped_tail_id))
                if len(tail_ids) > 1:
                    for i in range(len(tail_ids)-1):
                        new_triplets.append((tail_id_remap_table[tail_ids[i]], cur_relation_id_toge, tail_id_remap_table[tail_ids[i+1]]))
                    new_triplets.append((tail_id_remap_table[tail_ids[-1]], cur_relation_id_toge, tail_id_remap_table[tail_ids[0]]))
    new_triplets_array = np.array(new_triplets, dtype=np.int32)
    triplets = np.row_stack([triplets, new_triplets_array])
    # update global variables
    global n_entities, n_nodes, n_relations
    n_entities = max(max(triplets[:, 0]), max(
        triplets[:, 2])) + 1
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets



def load_rules11(directory, rule_file, entity_file, triplets):
    """load rules as triplets, inverse is not created"""
    rule_file = directory + rule_file
    entity_file = directory + entity_file
    # create uuid to id dictionary for later lookup
    entities_uuid2id = {}
    with open(entity_file) as f:
        skip_first = True
        for line in f:
            if skip_first:
                skip_first = False
            else:
                org_id = line.strip().split(' ')[0]
                remap_id = line.strip().split(' ')[2]
                entities_uuid2id[org_id] = remap_id
    # get current maximum node id and relation id
    maximum_node_id = max(max(triplets[:, 0]), max(triplets[:, 2]))
    maximum_relation_id = max(triplets[:, 1])
    # load in rules as triplets
    cur_node_id = maximum_node_id + 1
    cur_relation_id = maximum_relation_id + 1
    cur_relation_id_toge = cur_relation_id + 1
    tail_id_remap_table = {}
    new_triplets = []
    with open(rule_file) as f:
        for line in f:
            head_uuid, tail_ids = line.split(':')
            if head_uuid in entities_uuid2id.keys():
                head_id = entities_uuid2id[head_uuid]
                tail_ids = eval(tail_ids)
                for tail_id in tail_ids:
                    if tail_id not in tail_id_remap_table:
                        tail_id_remap_table[tail_id] = cur_node_id
                        cur_node_id += 1
                    remapped_tail_id = tail_id_remap_table[tail_id]
                    new_triplets.append((head_id, cur_relation_id, remapped_tail_id))
                if len(tail_ids) > 1:
                    for i in range(len(tail_ids)-1):
                        new_triplets.append((tail_id_remap_table[tail_ids[i]], cur_relation_id_toge, tail_id_remap_table[tail_ids[i+1]]))
                    new_triplets.append((tail_id_remap_table[tail_ids[-1]], cur_relation_id_toge, tail_id_remap_table[tail_ids[0]]))
    new_triplets_array = np.array(new_triplets, dtype=np.int32)
    new_triplets_array = inverse(new_triplets_array)
    triplets = np.row_stack([triplets, new_triplets_array])
    # update global variables
    global n_entities, n_nodes, n_relations
    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets



def load_rules12(directory, rule_file, entity_file, triplets):
    """load rules as triplets, inverse is not created"""
    rule_file = directory + rule_file
    entity_file = directory + entity_file
    # create uuid to id dictionary for later lookup
    entities_uuid2id = {}
    with open(entity_file) as f:
        skip_first = True
        for line in f:
            if skip_first:
                skip_first = False
            else:
                org_id = line.strip().split(' ')[0]
                remap_id = line.strip().split(' ')[2]
                entities_uuid2id[org_id] = remap_id
    # get current maximum node id and relation id
    maximum_node_id = max(max(triplets[:, 0]), max(triplets[:, 2]))
    maximum_relation_id = max(triplets[:, 1])
    # load in rules as triplets
    cur_node_id = maximum_node_id + 1
    cur_relation_id = maximum_relation_id + 1
    tail_id_remap_table = {}
    new_triplets = []
    with open(rule_file) as f:
        for line in f:
            head_uuid, tail_ids = line.split(':')
            if head_uuid in entities_uuid2id.keys():
                head_id = entities_uuid2id[head_uuid]
                tail_ids = eval(tail_ids)
                for tail_id in tail_ids:
                    if tail_id not in tail_id_remap_table:
                        tail_id_remap_table[tail_id] = cur_node_id
                        cur_node_id += 1
                    remapped_tail_id = tail_id_remap_table[tail_id]
                    new_triplets.append((head_id, cur_relation_id, remapped_tail_id))
    new_triplets_array = np.array(new_triplets, dtype=np.int32)
    triplets = np.row_stack([triplets, new_triplets_array])
    # update global variables
    global n_entities, n_nodes, n_relations
    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets



def load_rules13(directory, rule_file, entity_file, triplets):
    """load rules as triplets, inverse is not created"""
    rule_file = directory + rule_file
    entity_file = directory + entity_file
    # create uuid to id dictionary for later lookup
    entities_uuid2id = {}
    with open(entity_file) as f:
        skip_first = True
        for line in f:
            if skip_first:
                skip_first = False
            else:
                org_id = line.strip().split(' ')[0]
                remap_id = line.strip().split(' ')[2]
                entities_uuid2id[org_id] = remap_id
    # get current maximum node id and relation id
    maximum_node_id = max(max(triplets[:, 0]), max(triplets[:, 2]))
    maximum_relation_id = max(triplets[:, 1])
    # load in rules as triplets
    cur_node_id = maximum_node_id + 1
    cur_relation_id = maximum_relation_id + 1
    tail_id_remap_table = {}
    new_triplets = []
    with open(rule_file) as f:
        for line in f:
            head_uuid, tail_ids = line.split(':')
            if head_uuid in entities_uuid2id.keys():
                head_id = entities_uuid2id[head_uuid]
                tail_ids = eval(tail_ids)
                for tail_id in tail_ids:
                    if tail_id not in tail_id_remap_table:
                        tail_id_remap_table[tail_id] = cur_node_id
                        cur_node_id += 1
                    remapped_tail_id = tail_id_remap_table[tail_id]
                    new_triplets.append((head_id, cur_relation_id, remapped_tail_id))
    new_triplets_array = np.array(new_triplets, dtype=np.int32)

    triplets = np.row_stack([triplets, new_triplets_array])
    triplets = inverse(triplets)
    # update global variables
    global n_entities, n_nodes, n_relations
    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets


def load_rules14(directory, rule_file, entity_file, triplets):
    """load rules as triplets, inverse is not created"""
    rule_file = directory + rule_file
    entity_file = directory + entity_file
    # create uuid to id dictionary for later lookup
    entities_uuid2id = {}
    with open(entity_file) as f:
        skip_first = True
        for line in f:
            if skip_first:
                skip_first = False
            else:
                org_id = line.strip().split(' ')[0]
                remap_id = line.strip().split(' ')[2]
                entities_uuid2id[org_id] = remap_id
    # get current maximum node id and relation id
    maximum_node_id = max(max(triplets[:, 0]), max(triplets[:, 2]))
    maximum_relation_id = max(triplets[:, 1])
    # load in rules as triplets
    cur_node_id = maximum_node_id + 1
    cur_relation_id = maximum_relation_id + 1
    cur_relation_id_toge = cur_relation_id + 1
    tail_id_remap_table = {}
    new_triplets = []
    similar_triplets = []
    similar = []
    with open(rule_file) as f:
        for line in f:
            head_uuid, tail_ids = line.split(':')
            if head_uuid in entities_uuid2id.keys():
                head_id = entities_uuid2id[head_uuid]
                tail_ids = eval(tail_ids)
                for tail_id in tail_ids:
                    if tail_id not in tail_id_remap_table:
                        tail_id_remap_table[tail_id] = cur_node_id
                        cur_node_id += 1
                    remapped_tail_id = tail_id_remap_table[tail_id]
                    new_triplets.append((head_id, cur_relation_id, remapped_tail_id))
                if len(tail_ids) > 1:
                    similar.append(tuple(tail_ids))
    similar = list(unique(similar))
    for i in range(len(similar)):
        for j in range(len(similar[i])-1):
            similar_triplets.append((tail_id_remap_table[similar[i][j]], cur_relation_id_toge, tail_id_remap_table[similar[i][j+1]]))
        similar_triplets.append((tail_id_remap_table[similar[i][-1]], cur_relation_id_toge, tail_id_remap_table[similar[i][0]]))
    similar_triplets = np.array(similar_triplets, dtype=np.int32)
    new_triplets_array = np.array(new_triplets, dtype=np.int32)
    triplets = np.row_stack([triplets, new_triplets_array,similar_triplets])
    # update global variables
    global n_entities, n_nodes, n_relations
    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets




def load_rules15(directory, rule_file, entity_file, triplets):
    """load rules as triplets, inverse is not created"""
    rule_file = directory + rule_file
    entity_file = directory + entity_file
    # create uuid to id dictionary for later lookup
    entities_uuid2id = {}
    with open(entity_file) as f:
        skip_first = True
        for line in f:
            if skip_first:
                skip_first = False
            else:
                org_id = line.strip().split(' ')[0]
                remap_id = line.strip().split(' ')[2]
                entities_uuid2id[org_id] = remap_id
    # get current maximum node id and relation id
    maximum_node_id = max(max(triplets[:, 0]), max(triplets[:, 2]))
    maximum_relation_id = max(triplets[:, 1])
    # load in rules as triplets
    cur_node_id = maximum_node_id + 1
    cur_relation_id = maximum_relation_id + 1
    cur_relation_id_toge = cur_relation_id + 1
    tail_id_remap_table = {}
    new_triplets = []
    similar_triplets = []
    similar = []
    with open(rule_file) as f:
        for line in f:
            head_uuid, tail_ids = line.split(':')
            if head_uuid in entities_uuid2id.keys():
                head_id = entities_uuid2id[head_uuid]
                tail_ids = eval(tail_ids)
                for tail_id in tail_ids:
                    if tail_id not in tail_id_remap_table:
                        tail_id_remap_table[tail_id] = cur_node_id
                        cur_node_id += 1
                    remapped_tail_id = tail_id_remap_table[tail_id]
                    new_triplets.append((head_id, cur_relation_id, remapped_tail_id))
                if len(tail_ids) > 1:
                    similar.append(tuple(tail_ids))
    similar = list(unique(similar))

    for i in range(len(similar)):
        similar_list = list(combinations(similar[i],2))
        for j in similar_list:
            similar_triplets.append((tail_id_remap_table[j[0]], cur_relation_id_toge, tail_id_remap_table[j[1]]))

    similar_triplets = np.array(similar_triplets, dtype=np.int32)
    new_triplets_array = np.array(new_triplets, dtype=np.int32)
    triplets = np.row_stack([triplets, new_triplets_array,similar_triplets])
    # update global variables
    global n_entities, n_nodes, n_relations
    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets




def load_rules16(directory, rule_file, entity_file, triplets):
    """load rules as triplets, inverse is not created"""
    rule_file = directory + rule_file
    entity_file = directory + entity_file
    # create uuid to id dictionary for later lookup
    entities_uuid2id = {}
    with open(entity_file) as f:
        skip_first = True
        for line in f:
            if skip_first:
                skip_first = False
            else:
                org_id = line.strip().split(' ')[0]
                remap_id = line.strip().split(' ')[2]
                entities_uuid2id[org_id] = remap_id
    # get current maximum node id and relation id
    maximum_node_id = max(max(triplets[:, 0]), max(triplets[:, 2]))
    maximum_relation_id = max(triplets[:, 1])
    # load in rules as triplets
    cur_node_id = maximum_node_id + 1
    cur_relation_id = maximum_relation_id + 1
    cur_relation_id_toge = cur_relation_id + 1
    tail_id_remap_table = {}
    new_triplets = []
    similar_triplets = []
    similar = []
    with open(rule_file) as f:
        for line in f:
            head_uuid, tail_ids = line.split(':')
            if head_uuid in entities_uuid2id.keys():
                head_id = entities_uuid2id[head_uuid]
                tail_ids = eval(tail_ids)
                for tail_id in tail_ids:
                    if tail_id not in tail_id_remap_table:
                        tail_id_remap_table[tail_id] = cur_node_id
                        cur_node_id += 1
                    remapped_tail_id = tail_id_remap_table[tail_id]
                    new_triplets.append((head_id, cur_relation_id, remapped_tail_id))
                if len(tail_ids) > 1:
                    similar.append(tuple(tail_ids))
    similar = list(unique(similar))
    for i in range(len(similar)):
        for j in range(len(similar[i])-1):
            similar_triplets.append((tail_id_remap_table[similar[i][j]], cur_relation_id_toge, tail_id_remap_table[similar[i][j+1]]))
        similar_triplets.append((tail_id_remap_table[similar[i][-1]], cur_relation_id_toge, tail_id_remap_table[similar[i][0]]))
    similar_triplets = np.array(similar_triplets, dtype=np.int32)
    similar_triplets = inverse(similar_triplets)
    new_triplets_array = np.array(new_triplets, dtype=np.int32)
    triplets = np.row_stack([triplets, new_triplets_array,similar_triplets])
    # update global variables
    global n_entities, n_nodes, n_relations
    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets



def load_rules17(directory, rule_file, entity_file, triplets):
    """load rules as triplets, inverse is not created"""
    rule_file = directory + rule_file
    entity_file = directory + entity_file
    # create uuid to id dictionary for later lookup
    entities_uuid2id = {}
    with open(entity_file) as f:
        skip_first = True
        for line in f:
            if skip_first:
                skip_first = False
            else:
                org_id = line.strip().split(' ')[0]
                remap_id = line.strip().split(' ')[2]
                entities_uuid2id[org_id] = remap_id
    # get current maximum node id and relation id
    maximum_node_id = max(max(triplets[:, 0]), max(triplets[:, 2]))
    maximum_relation_id = max(triplets[:, 1])
    # load in rules as triplets
    cur_node_id = maximum_node_id + 1
    cur_relation_id = maximum_relation_id + 1

    tail_id_remap_table = {}
    new_triplets = []
    similar_triplets = []
    similar = []
    with open(rule_file) as f:
        for line in f:
            head_uuid, tail_ids = line.split(':')
            if head_uuid in entities_uuid2id.keys():
                head_id = entities_uuid2id[head_uuid]
                tail_ids = eval(tail_ids)
                for tail_id in tail_ids:
                    if tail_id not in tail_id_remap_table:
                        tail_id_remap_table[tail_id] = cur_node_id
                        cur_node_id += 1
                    remapped_tail_id = tail_id_remap_table[tail_id]
                    new_triplets.append((head_id, cur_relation_id, remapped_tail_id))
                if len(tail_ids) > 1:
                    similar.append(tuple(tail_ids))
    similar = list(unique(similar))
    new_triplets_array = np.array(new_triplets, dtype=np.int32)
    new_triplets_array = inverse(new_triplets_array)
    cur_relation_id_toge = cur_relation_id + 2
    for i in range(len(similar)):
        similar_list = list(combinations(similar[i], 2))
        for j in similar_list:
            similar_triplets.append((tail_id_remap_table[j[0]], cur_relation_id_toge, tail_id_remap_table[j[1]]))

    similar_triplets = np.array(similar_triplets, dtype=np.int32)
    similar_triplets = inverse(similar_triplets)

    triplets = np.row_stack([triplets, new_triplets_array, similar_triplets])
    # update global variables
    global n_entities, n_nodes, n_relations
    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets

def load_rules18(directory, rule_file, entity_file, triplets):
    """load rules as triplets, inverse is not created"""
    rule_file = directory + rule_file
    entity_file = directory + entity_file
    # create uuid to id dictionary for later lookup
    entities_uuid2id = {}
    with open(entity_file) as f:
        skip_first = True
        for line in f:
            if skip_first:
                skip_first = False
            else:
                org_id = line.strip().split(' ')[0]
                remap_id = line.strip().split(' ')[2]
                entities_uuid2id[org_id] = remap_id
    # get current maximum node id and relation id
    maximum_node_id = max(max(triplets[:, 0]), max(triplets[:, 2]))
    maximum_relation_id = max(triplets[:, 1])
    # load in rules as triplets
    cur_node_id = maximum_node_id + 1
    cur_relation_id = maximum_relation_id + 1

    tail_id_remap_table = {}
    new_triplets = []
    similar_triplets = []
    similar = []
    with open(rule_file) as f:
        for line in f:
            head_uuid, tail_ids = line.split(':')
            if head_uuid in entities_uuid2id.keys():
                head_id = entities_uuid2id[head_uuid]
                tail_ids = eval(tail_ids)
                for tail_id in tail_ids:
                    if tail_id not in tail_id_remap_table:
                        tail_id_remap_table[tail_id] = cur_node_id
                        cur_node_id += 1
                    remapped_tail_id = tail_id_remap_table[tail_id]
                    new_triplets.append((head_id, cur_relation_id, remapped_tail_id))
                if len(tail_ids) > 1:
                    similar.append(tuple(tail_ids))
    similar = list(unique(similar))
    new_triplets_array = np.array(new_triplets, dtype=np.int32)
    #new_triplets_array = inverse(new_triplets_array)
    cur_relation_id_toge = cur_relation_id + 1
    for i in range(len(similar)):
        similar_list = list(combinations(similar[i], 2))
        for j in similar_list:
            similar_triplets.append((tail_id_remap_table[j[0]], cur_relation_id_toge, tail_id_remap_table[j[1]]))

    similar_triplets = np.array(similar_triplets, dtype=np.int32)
    similar_triplets = inverse(similar_triplets)

    triplets = np.row_stack([triplets, new_triplets_array, similar_triplets])
    # update global variables
    global n_entities, n_nodes, n_relations
    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets







def load_rules19(directory, rule_file, entity_file, triplets):
    """load rules as triplets, inverse is not created"""
    rule_file = directory + rule_file
    entity_file = directory + entity_file
    # create uuid to id dictionary for later lookup
    entities_uuid2id = {}
    with open(entity_file) as f:
        skip_first = True
        for line in f:
            if skip_first:
                skip_first = False
            else:
                org_id = line.strip().split(' ')[0]
                remap_id = line.strip().split(' ')[2]
                entities_uuid2id[org_id] = remap_id
    # get current maximum node id and relation id
    maximum_node_id = max(max(triplets[:, 0]), max(triplets[:, 2]))
    maximum_relation_id = max(triplets[:, 1])
    # load in rules as triplets
    cur_node_id = maximum_node_id + 1
    cur_relation_id = maximum_relation_id + 1

    tail_id_remap_table = {}
    new_triplets = []
    similar_triplets = []
    similar = []
    with open(rule_file) as f:
        for line in f:
            head_uuid, tail_ids = line.split(':')
            if head_uuid in entities_uuid2id.keys():
                head_id = entities_uuid2id[head_uuid]
                tail_ids = eval(tail_ids)
                for tail_id in tail_ids:
                    if tail_id not in tail_id_remap_table:
                        tail_id_remap_table[tail_id] = cur_node_id
                        cur_node_id += 1
                    remapped_tail_id = tail_id_remap_table[tail_id]
                    new_triplets.append((head_id, cur_relation_id, remapped_tail_id))
                if len(tail_ids) > 1:
                    similar.append(tuple(tail_ids))
    similar = list(unique(similar))
    new_triplets_array = np.array(new_triplets, dtype=np.int32)
    new_triplets_array = inverse(new_triplets_array)
    cur_relation_id_toge = cur_relation_id + 2
    for i in range(len(similar)):
        for j in range(len(similar[i])-1):
            similar_triplets.append((tail_id_remap_table[similar[i][j]], cur_relation_id_toge, tail_id_remap_table[similar[i][j+1]]))
        similar_triplets.append((tail_id_remap_table[similar[i][-1]], cur_relation_id_toge, tail_id_remap_table[similar[i][0]]))
    similar_triplets = np.array(similar_triplets, dtype=np.int32)
    similar_triplets = inverse(similar_triplets)

    triplets = np.row_stack([triplets, new_triplets_array,similar_triplets])
    # update global variables
    global n_entities, n_nodes, n_relations
    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets


def load_rules20(directory, rule_file, entity_file, triplets):
    """load rules as triplets, inverse is not created"""
    rule_file = directory + rule_file
    entity_file = directory + entity_file
    # create uuid to id dictionary for later lookup
    entities_uuid2id = {}
    with open(entity_file) as f:
        skip_first = True
        for line in f:
            if skip_first:
                skip_first = False
            else:
                org_id = line.strip().split(' ')[0]
                remap_id = line.strip().split(' ')[2]
                entities_uuid2id[org_id] = remap_id
    # get current maximum node id and relation id
    maximum_node_id = max(max(triplets[:, 0]), max(triplets[:, 2]))
    maximum_relation_id = max(triplets[:, 1])
    # load in rules as triplets
    cur_node_id = maximum_node_id + 1
    cur_relation_id = 1
    tail_id_remap_table = {}
    new_triplets = []
    with open(rule_file) as f:
        for line in f:
            head_uuid, tail_ids = line.split(':')
            if head_uuid in entities_uuid2id.keys():
                head_id = entities_uuid2id[head_uuid]
                tail_ids = eval(tail_ids)
                for tail_id in tail_ids:
                    if tail_id not in tail_id_remap_table:
                        tail_id_remap_table[tail_id] = cur_node_id
                        cur_node_id += 1
                    remapped_tail_id = tail_id_remap_table[tail_id]
                    new_triplets.append((remapped_tail_id, cur_relation_id, head_id))
    new_triplets_array = np.array(new_triplets, dtype=np.int32)
    triplets = np.row_stack([triplets, new_triplets_array])
    # update global variables
    en1 = set(sorted(triplets[:, 0]))
    en2 = set(sorted(triplets[:, 2]))
    res = en1.union(en2)
    x = len(res)
    global n_entities, n_nodes, n_relations
    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets




def load_rules21(directory, rule_file, entity_file, triplets):
    """load rules as triplets, inverse is not created"""
    rule_file = directory + rule_file
    entity_file = directory + entity_file
    # create uuid to id dictionary for later lookup
    entities_uuid2id = {}
    with open(entity_file) as f:
        skip_first = True
        for line in f:
            if skip_first:
                skip_first = False
            else:
                org_id = line.strip().split(' ')[0]
                remap_id = line.strip().split(' ')[2]
                entities_uuid2id[org_id] = remap_id
    # get current maximum node id and relation id
    maximum_node_id = max(max(triplets[:, 0]), max(triplets[:, 2]))
    maximum_relation_id = max(triplets[:, 1])
    # load in rules as triplets
    cur_node_id = maximum_node_id + 1
    #treat loaded rule relation as 'including' : 1
    cur_relation_id = 1
    tail_id_remap_table = {}
    new_triplets = []
    with open(rule_file) as f:
        for line in f:
            head_uuid, tail_ids = line.split(':')
            if head_uuid in entities_uuid2id.keys():
                head_id = entities_uuid2id[head_uuid]
                tail_ids = eval(tail_ids)
                for tail_id in tail_ids:
                    if tail_id not in tail_id_remap_table:
                        tail_id_remap_table[tail_id] = cur_node_id
                        cur_node_id += 1
                    remapped_tail_id = tail_id_remap_table[tail_id]
                    new_triplets.append((remapped_tail_id, cur_relation_id, head_id))
    new_triplets_array = np.array(new_triplets, dtype=np.int32)
    triplets = np.row_stack([triplets, new_triplets_array])
    triplets = inverse(triplets)
    # update global variables
    en1 = set(sorted(triplets[:, 0]))
    en2 = set(sorted(triplets[:, 2]))
    res = en1.union(en2)
    x = len(res)
    global n_entities, n_nodes, n_relations
    n_entities = max(max(triplets[:, 0]), max(triplets[:, 2])) + 1
    n_nodes = n_entities + n_users
    n_relations = max(triplets[:, 1]) + 1

    return triplets



def unique(data):
    obj = {}
    for item in data:
        obj[item] = item
    return obj.values()






def inverse(can_triplets_np):

    inv_triplets_np = can_triplets_np.copy()
    inv_triplets_np[:, 0] = can_triplets_np[:, 2]
    inv_triplets_np[:, 2] = can_triplets_np[:, 0]
    inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1])-min(can_triplets_np[:,1]) + 1
    # inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1


    triplets = np.row_stack([can_triplets_np, inv_triplets_np])

    return triplets



def load_data_org(model_args):
    global args
    args = model_args
    directory = args.data_path + args.dataset + '/'
    print('reading train and test user-item set ...')
    train_cf = read_cf(directory + 'train.txt')
    test_cf = read_cf(directory + 'test.txt')
    remap_item(train_cf, test_cf)

    print('combinating train_cf and kg data ...')
    triplets = read_triplets(
        directory + 'kg_final_org.txt')  # head, relation, tail triplets

    print('building the graph ...')
    graph, relation_dict = build_graph(train_cf, triplets)

    print('building the adj mat ...')
    adj_mat_list, norm_mat_list, mean_mat_list = build_sparse_relational_graph(
        relation_dict)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
        'n_entities': int(n_entities),
        'n_nodes': int(n_nodes),
        'n_relations': int(n_relations)
    }
    user_dict = {
        'train_user_set': train_user_set,
        'test_user_set': test_user_set
    }

    return train_cf, test_cf, user_dict, n_params, graph, [
        adj_mat_list, norm_mat_list, mean_mat_list
    ]
