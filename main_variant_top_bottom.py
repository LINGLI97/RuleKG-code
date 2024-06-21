'''
Created on July 1, 2020
@author: Tinglin Huang (tinglin.huang@zju.edu.cn)
'''
__author__ = "huangtinglin"

import random

import torch
import numpy as np

from time import time
from prettytable import PrettyTable

from utils.parser import parse_args
from utils.data_loader_org import load_data_org_top_bottom
from modules.KGIN_variant import Recommender
from utils.evaluate import test
from utils.helper import early_stopping
import pdb
import datetime
import os
from pathlib import Path


n_users = 0
n_items = 0
n_entities = 0  # entities are nodes that are not users and items
n_nodes = 0  # nodes = users + items + entities
n_relations = 0


def get_feed_dict(train_entity_pairs, start, end, train_user_set):
    #from non-interact item choose neg randomly
    def negative_sampling(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                # print(f"nitems is {n_items}")
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)
        return neg_items

    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,
                                                                train_user_set)).to(device)
    return feed_dict


def save_model(file_name, model, config):
    if not os.path.isdir(config.out_dir):
        os.mkdir(config.out_dir)

    model_file = Path(config.out_dir + file_name)
    model_file.touch(exist_ok=True)

    print("Saving model...")
    torch.save(model.state_dict(), model_file)


if __name__ == '__main__':
    """fix the random seed"""

    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args, device
    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")
    # write the results to KGIN_top&bottom...
    fid_write_t2 = open('./result/KGIN_top&bottom_' +
                        datetime.datetime.now().strftime("%m%d%H%M%S") + args.description +'.txt', 'w')
    # print the parametrs of the args
    fid_write_t2.write('#' * 20 + ' parameter settings ' + '#'*20 + '\n')
    for k in args.__dict__:
        if args.__dict__[k] is not None:
            fid_write_t2.write(str(k) + ':' + str(args.__dict__[k]) + '\n')
    fid_write_t2.flush()

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data_org_top_bottom(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']
    """cf data"""
    train_cf_pairs = torch.LongTensor(
        np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(
        np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

    """define model"""
    model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)

    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    fid_write_t2.write('#' * 20 + ' experimental settings ' + '#'*20 + '\n')
    fid_write_t2.write('0) Model: KGIN\n')
    fid_write_t2.write('1) recommender: GCN\n')
    fid_write_t2.write('2) sampler: random sampling\n')
    fid_write_t2.write('#' * 20 + ' start training ' + '#'*20 + '\n')
    fid_write_t2.flush()
    eval_metric_list = ['recall', 'ndcg', 'precision', 'hit_ratio']
    train_s_epoch = time()

    print("start training ...")
    for epoch in range(args.epoch):
        """training CF"""
        # shuffle training data

        print('processing epoch:{}/{}\n'.format(epoch, args.epoch))
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf_pairs = train_cf_pairs[index]

        """training"""
        loss, s, cor_loss = 0, 0, 0
        num = 1
        train_s_t = time()
        while s + args.batch_size <= len(train_cf):
            #batch中users，pos,neg
            batch = get_feed_dict(train_cf_pairs,
                                  s, s + args.batch_size,
                                  user_dict['train_user_set'])
            neg_item = batch['neg_items']
            batch_loss, _, _, batch_cor = model(neg_item, batch)

            batch_loss = batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            cor_loss += batch_cor
            s += args.batch_size
            num += 1

        train_e_t = time()

        if epoch % args.k_step == 0:
            """testing"""
            test_s_t = time()
            ret = test(model, user_dict, n_params)
            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision", "hit_ratio"]
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
            )
            print(train_res)
            #print("\n".join("{:10s}: {}".format(key, values) for key, values in ret.items()))

            fid_write_t2.write('Epoch:{}/{}, one epoch(train+test/hour): {}, training time: {}, testing time: {}\n'.format(
                epoch, args.epoch, (test_e_t-train_s_t)/3600, train_e_t-train_s_t, test_e_t-test_s_t))
            for key in eval_metric_list:
                #长度为10,
                print("{:10s}: {}\n".format(key, ret[key]))
                fid_write_t2.write("{:10s}: {}\n".format(key, ret[key]))
            fid_write_t2.flush()

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step = args.flag_step)
            if should_stop:
                break

            """save weight"""
            if ret['recall'][0] == cur_best_pre_0 and args.save:
                save_model('best_KGIN_top&bottom_' + args.dataset + '.ckpt', model, args)
        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            print('using time %.4f, training loss at epoch %d: %.4f, cor: %.6f' % (train_e_t - train_s_t, epoch, loss.item(), cor_loss.item()))

    test_s_epoch = time()
    print('\ntraining {%d} epoches takes {%.3f} seconds,{%.3f} hour' % (
        args.epoch, test_s_epoch-train_s_epoch, (test_s_epoch-train_s_epoch)/3600))
    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))
    fid_write_t2.write('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))
    fid_write_t2.write('\ntraining {%d} epoches takes {%.3f} seconds,{%.3f} hour' % (
        args.epoch, test_s_epoch - train_s_epoch, (test_s_epoch - train_s_epoch) / 3600))
    fid_write_t2.flush()
    fid_write_t2.close()
