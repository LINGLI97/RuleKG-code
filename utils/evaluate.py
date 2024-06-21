from .metrics import *
from .parser import parse_args

import torch
import numpy as np
import multiprocessing
import heapq
from time import time
from tqdm import tqdm
import pdb

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)
device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag

def cal_ndcg(topk, test_set, num_pos, k):
    n = min(num_pos, k)
    nrange = np.arange(n) + 2
    idcg = np.sum(1 / np.log2(nrange))

    dcg = 0
    for i, s in enumerate(topk):
        if s in test_set:
            dcg += 1 / np.log2(i + 2)

    ndcg = dcg / idcg

    return ndcg

def get_score_KGAT(model, edge_matrix, n_users, n_items, n_entities, train_user_dict, s, t):
    gcn_embedding = model.generate(edge_matrix)
    u_e, i_e, e_e = torch.split(gcn_embedding, [n_users, n_items, n_entities-n_items])
    #u_e, i_e, e_e = torch.split(model.all_embed, [n_users, n_items, n_entities-n_items])
    #u_e, i_e = torch.split(model.all_embed, [n_users, n_items])

    u_e = u_e[s:t, :]

    score_matrix = torch.matmul(u_e, i_e.t())
    for u in range(s, t):
        pos = train_user_dict[u]
        idx = pos.index(-1) if -1 in pos else len(pos)
        train_item_index=[gg-n_users for gg in pos[:idx]]
        score_matrix[u-s][train_item_index] = -1e5
        #score_matrix[u-s][pos[:idx] - n_users] = -1e5

    return score_matrix

def get_score(model, n_users, n_items, n_entities, train_user_dict, s, t):
    #u_e, i_e, e_e = torch.split(model.all_embed, [n_users, n_items, n_entities-n_items])
    u_e, i_e = torch.split(model.all_embed, [n_users, n_items])

    u_e = u_e[s:t, :]

    score_matrix = torch.matmul(u_e, i_e.t())
    for u in range(s, t):
        pos = train_user_dict[u]
        idx = pos.index(-1) if -1 in pos else len(pos)
        train_item_index=[gg-n_users for gg in pos[:idx]]
        score_matrix[u-s][train_item_index] = -1e5
        #score_matrix[u-s][pos[:idx] - n_users] = -1e5

    return score_matrix

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = AUC(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set
    try:
        training_items = train_user_set[u]
    except Exception:
        training_items = []
    # user u's items in the test set
    user_pos_test = test_user_set[u]

    all_items = set(range(0, n_items))

    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)

def test_v2(model, ks, ckg, user_dict, n_params, recom_type, edge_matrix, n_batchs=4):
    ks = eval(ks)
    train_user_dict, test_user_dict = user_dict['train_user_set'], user_dict['test_user_set']
    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']

    #train_user_dict, test_user_dict = ckg.train_user_dict, ckg.test_user_dict

    n_test_users = len(test_user_dict)

    n_k = len(ks)
    result = {
        "precision": np.zeros(n_k),
        "recall": np.zeros(n_k),
        "ndcg": np.zeros(n_k),
        "hit_ratio": np.zeros(n_k),
    }

    n_users = model.n_users
    batch_size = n_users // n_batchs
    for batch_id in tqdm(range(n_batchs), ascii=True, desc="Evaluate",disable=True):
        s = batch_size * batch_id
        t = batch_size * (batch_id + 1)
        if t > n_users:
            t = n_users
        if s == t:
            break
        if recom_type == 'MF':
            score_matrix = get_score(model, n_users, n_items, n_entities, train_user_dict, s, t)
        else:
            score_matrix = get_score_KGAT(model, edge_matrix, n_users, n_items, n_entities, train_user_dict, s, t)
        for i, k in enumerate(ks):
            precision, recall, ndcg, hr = 0, 0, 0, 0
            #pdb.set_trace()
            _, topk_index = torch.topk(score_matrix, k)
            topk_index = topk_index.cpu().numpy() + n_users

            for u in range(s, t):
                gt_pos = test_user_dict[u]
                topk = topk_index[u - s]
                num_pos = len(gt_pos)

                topk_set = set(topk)
                test_set = set(gt_pos)
                num_hit = len(topk_set & test_set)

                precision += num_hit / k
                recall += num_hit / num_pos
                hr += 1 if num_hit > 0 else 0

                ndcg += cal_ndcg(topk, test_set, num_pos, k)

            result["precision"][i] += precision / n_test_users
            result["recall"][i] += recall / n_test_users
            result["ndcg"][i] += ndcg / n_test_users
            result["hit_ratio"][i] += hr / n_test_users

    return result

def test(model, user_dict, n_params):
    result = {'precision': np.zeros(len(Ks)),
              'recall': np.zeros(len(Ks)),
              'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)),
              'auc': 0.}

    global n_users, n_items
    n_items = n_params['n_items']
    n_users = n_params['n_users']

    global train_user_set, test_user_set
    train_user_set = user_dict['train_user_set']
    test_user_set = user_dict['test_user_set']

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = list(test_user_set.keys())
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    entity_gcn_emb, user_gcn_emb = model.generate()

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_list_batch = test_users[start: end]
        user_batch = torch.LongTensor(np.array(user_list_batch)).to(device)
        u_g_embeddings = user_gcn_emb[user_batch]

        if batch_test_flag:
            # batch-item test
            n_item_batchs = n_items // i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), n_items))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, n_items)

                item_batch = torch.LongTensor(np.array(range(i_start, i_end))).view(i_end-i_start).to(device)
                i_g_embddings = entity_gcn_emb[item_batch]

                i_rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == n_items
        else:
            # all-item test
            item_batch = torch.LongTensor(np.array(range(0, n_items))).view(n_items, -1).to(device)
            i_g_embddings = entity_gcn_emb[item_batch]
            rate_batch = model.rating(u_g_embeddings, i_g_embddings).detach().cpu()

        user_batch_rating_uid = zip(rate_batch, user_list_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users

    assert count == n_test_users
    pool.close()
    return result
