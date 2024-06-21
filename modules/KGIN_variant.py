'''
Created on July 1, 2020
PyTorch Implementation of KGIN
@author: Tinglin Huang (tinglin.huang@zju.edu.cn)
'''
__author__ = "huangtinglin"

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
import pdb

class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """
    def __init__(self, n_users, n_factors):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_factors = n_factors

    def forward(self, entity_emb, user_emb, latent_emb,
                edge_index, edge_type, interact_mat,
                weight, disen_weight_att):

        n_entities = entity_emb.shape[0]
        channel = entity_emb.shape[1]
        n_users = self.n_users
        n_factors = self.n_factors

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        a = entity_emb[tail]
        #print(type(a))
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        """cul user->latent factor attention"""
        score_ = torch.mm(user_emb, latent_emb.t())
        score = nn.Softmax(dim=1)(score_).unsqueeze(-1)  # [n_users, n_factors, 1]

        """user aggregate"""
        user_agg = torch.sparse.mm(interact_mat, entity_emb)  # [n_users, channel]
        disen_weight = torch.mm(nn.Softmax(dim=-1)(disen_weight_att),
                                weight).expand(n_users, n_factors, channel)
        user_agg = user_agg * (disen_weight * score).sum(dim=1) + user_agg  # [n_users, channel]

        return entity_agg, user_agg


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_users,
                 n_factors, n_relations, interact_mat,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_factors = n_factors
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind

        self.temperature = 0.2
        initializer = nn.init.xavier_uniform_
        #channel = 64
        weight = initializer(torch.empty(n_relations - 1, channel))  # not include interact
        self.weight = nn.Parameter(weight)  # [n_relations - 1, in_channel]
        #n_factor = number of latent factor for user favour
        disen_weight_att = initializer(torch.empty(n_factors, n_relations - 1))
        self.disen_weight_att = nn.Parameter(disen_weight_att)

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users, n_factors=n_factors))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    # def _cul_cor_pro(self):
    #     # disen_T: [num_factor, dimension]
    #     disen_T = self.disen_weight_att.t()
    #
    #     # normalized_disen_T: [num_factor, dimension]
    #     normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)
    #
    #     pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
    #     ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)
    #
    #     pos_scores = torch.exp(pos_scores / self.temperature)
    #     ttl_scores = torch.exp(ttl_scores / self.temperature)
    #
    #     mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
    #     return mi_score

    def _cul_cor(self):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative
        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = torch.matmul(tensor_1, tensor_1.t()) * 2, \
                   torch.matmul(tensor_2, tensor_2.t()) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = torch.sqrt(torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8), \
                   torch.sqrt(torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel ** 2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel ** 2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel ** 2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)
        def MutualInformation():
            # disen_T: [num_factor, dimension]
            disen_T = self.disen_weight_att.t()

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / disen_T.norm(dim=1, keepdim=True)

            pos_scores = torch.sum(normalized_disen_T * normalized_disen_T, dim=1)
            ttl_scores = torch.sum(torch.mm(disen_T, self.disen_weight_att), dim=1)

            pos_scores = torch.exp(pos_scores / self.temperature)
            ttl_scores = torch.exp(ttl_scores / self.temperature)

            mi_score = - torch.sum(torch.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return MutualInformation()
        else:
            cor = 0
            for i in range(self.n_factors):
                for j in range(i + 1, self.n_factors):
                    if self.ind == 'distance':
                        cor += DistanceCorrelation(self.disen_weight_att[i], self.disen_weight_att[j])
                    else:
                        cor += CosineSimilarity(self.disen_weight_att[i], self.disen_weight_att[j])
        return cor

    def forward(self, user_emb, entity_emb, latent_emb, edge_index, edge_type,
                interact_mat, mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        cor = self._cul_cor()
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](entity_emb, user_emb, latent_emb,
                                                 edge_index, edge_type, interact_mat,
                                                 self.weight, self.disen_weight_att)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)

            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb, cor

class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat):
        super(Recommender, self).__init__()
        #node = entity + user
        #entity = items + outfits
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.decay = args_config.l2
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.n_factors = args_config.n_factors
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.ind = args_config.ind
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                                      else torch.device("cpu")

        self.adj_mat = adj_mat
        self.graph = graph
        self.edge_index, self.edge_type = self._get_edges(graph)

        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.latent_emb = nn.Parameter(self.latent_emb)

        self.gcn = self._init_model()

    def _init_weight(self):
        #xavier initializer
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.latent_emb = initializer(torch.empty(self.n_factors, self.emb_size))

        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         n_factors=self.n_factors,
                         interact_mat=self.interact_mat,
                         ind=self.ind,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        #torch.FloatTensor:32 bits，torch.LongTensor:64bits
        i = torch.LongTensor([coo.row, coo.col])
        #torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def forward(self, neg_item, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        #neg_item = batch['neg_items']

        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        entity_gcn_emb, user_gcn_emb, cor = self.gcn(user_emb,
                                                     item_emb,
                                                     self.latent_emb,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     self.interact_mat,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout)
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        #pdb.set_trace()
        return self.create_bpr_loss(u_e, pos_e, neg_e, cor)

    def generate(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        return self.gcn(user_emb,
                        item_emb,
                        self.latent_emb,
                        self.edge_index,
                        self.edge_type,
                        self.interact_mat,
                        mess_dropout=False, node_dropout=False)[:-1]

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items, cor):
        batch_size = users.shape[0]
        #pdb.set_trace()
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        cor_loss = self.sim_decay * cor

        return mf_loss + emb_loss, mf_loss, emb_loss, cor

class KGPolicy(nn.Module):
    """
    Dynamical negative item sampler based on Knowledge graph
    Input: user, postive item, knowledge graph embedding
    Ouput: qualified negative item
    """

    def __init__(self, dis, params, config, graph, adj_mat):
        super(KGPolicy, self).__init__()
        self.params = params
        self.config = config
        self.dis = dis
        self.adj_mat = adj_mat
        self.emb_size = config.dim
        self.context_hops = config.context_hops
        self.n_users = params['n_users']
        self.n_nodes = params['n_nodes']
        self.n_relations = params['n_relations']
        self.n_users = params['n_users']
        self.n_factors = config.n_factors
        self.device = torch.device("cuda:" + str(config.gpu_id)) if config.cuda else torch.device("cpu")        
        #self.device = torch.device("cpu")
        #self.device = torch.device("cuda:0")
        #self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)
        self.ind = config.ind
        self.node_dropout = config.node_dropout
        self.node_dropout_rate = config.node_dropout_rate
        self.mess_dropout = config.mess_dropout
        self.mess_dropout_rate = config.mess_dropout_rate
        in_channel = eval(config.in_channel)
        out_channel = eval(config.out_channel)
        #self.gcn = GraphConv(in_channel, out_channel, config)

        # original kg-policy structure 
        self.n_entities = params["n_nodes"]
        self.item_range = params["item_range"]
        self.input_channel = in_channel
        #self.entity_embedding = self._initialize_weight(
        #    self.n_entities, self.input_channel
        #)
        self.edge_index, self.edge_type = self._get_edges(graph)
        self._init_weight()
        self.all_embed = nn.Parameter(self.all_embed)
        self.latent_emb = nn.Parameter(self.latent_emb)
        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.latent_emb = initializer(torch.empty(self.n_factors, self.emb_size))

        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         n_factors=self.n_factors,
                         interact_mat=self.interact_mat,
                         ind=self.ind,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)


    def _initialize_weight(self, n_entities, input_channel):
        """entities includes items and other entities in knowledge graph"""
        if self.config.pretrain_s:
            pdb.set_trace()
            kg_embedding = self.params["kg_embedding"]
            entity_embedding = nn.Parameter(kg_embedding)
        else:
            entity_embedding = nn.Parameter(
                torch.FloatTensor(n_entities, input_channel[0])
            )
            nn.init.xavier_uniform_(entity_embedding)

        if self.config.freeze_s:
            entity_embedding.requires_grad = False

        return entity_embedding

    def forward(self, data_batch, adj_matrix, edge_matrix):
        users = data_batch["users"]
        pos = data_batch["pos_items"]

        self.edges = self.build_edge(edge_matrix)

        '''
        for _ in range(k):
            """sample candidate negative items based on knowledge graph"""
            one_hop, one_hop_logits = self.kg_step(pos, users, adj_matrix, step=1)
            candidate_neg, two_hop_logits = self.kg_step(
                one_hop, users, adj_matrix, step=2
            )
            candidate_neg = self.filter_entity(candidate_neg, self.item_range)
            good_neg, good_logits = self.prune_step(
                self.dis, candidate_neg, users, two_hop_logits
            )
            good_logits = good_logits + one_hop_logits

            neg_list = torch.cat([neg_list, good_neg.unsqueeze(0)])
            prob_list = torch.cat([prob_list, good_logits.unsqueeze(0)])

            pos = good_neg
        
        #sample negatives based on merely outfits. 
        for _ in range(k):
            """sample candidate negative items based on knowledge graph"""

            candidate_neg, two_hop_logits = self.kg_step(pos, users, adj_matrix, step=2)
            candidate_neg = self.filter_entity(candidate_neg, self.item_range)
            good_neg, good_logits = self.prune_step(
                self.dis, candidate_neg, users, two_hop_logits
            )
            good_logits = good_logits

            neg_list = torch.cat([neg_list, good_neg.unsqueeze(0)])
            prob_list = torch.cat([prob_list, good_logits.unsqueeze(0)])
            pos = good_neg
        return neg_list, prob_list

        neg_list_outfit = torch.tensor([], dtype=torch.long, device=adj_matrix.device)
        prob_list_outfit = torch.tensor([], device=adj_matrix.device)
        neg_list_user = torch.tensor([], dtype=torch.long, device=adj_matrix.device)
        prob_list_user = torch.tensor([], device=adj_matrix.device)
        k = self.config.k_step
        assert k > 0
        for _ in range(k):
            """sample candidate negative items based on knowledge graph"""

            candidate_neg_outfit, one_hop_logits = self.kg_step_outfit(pos, users, adj_matrix, step=2)
            candidate_neg_outfit = self.filter_entity_outfit(candidate_neg_outfit, self.item_range)
            good_neg_outfit, good_logits_outfit = self.prune_step(
                self.dis, candidate_neg_outfit, users, one_hop_logits
            )

            neg_list_outfit = torch.cat([neg_list_outfit, good_neg_outfit.unsqueeze(0)])
            prob_list_outfit = torch.cat([prob_list_outfit, good_logits_outfit.unsqueeze(0)])
        return neg_list_outfit, prob_list_outfit 
        #return neg_list, prob_list
        '''

        neg_list = torch.tensor([], dtype=torch.long, device=adj_matrix.device)
        prob_list = torch.tensor([], device=adj_matrix.device)
        neg_list_user = torch.tensor([], dtype=torch.long, device=adj_matrix.device)
        prob_list_user = torch.tensor([], device=adj_matrix.device)
        k = self.config.k_step
        assert k > 0
        for _ in range(k):
            """sample candidate negative items based on knowledge graph"""
            '''
            candidate_neg, one_hop_logits = self.kg_step(pos, users, adj_matrix, step=1)
            candidate_neg = self.filter_entity(candidate_neg, self.item_range)
            '''
            #OUTFIT 
            candidate_neg, two_hop_logits = self.kg_step_outfit(pos, users, adj_matrix, step=2)
            candidate_neg = self.filter_entity(candidate_neg, self.item_range)
            good_neg, good_logits = self.prune_step(
                self.dis, candidate_neg, users, two_hop_logits
            )
            good_logits = good_logits
            good_neg = good_neg

            neg_list = torch.cat([neg_list, good_neg.unsqueeze(0)])
            prob_list = torch.cat([prob_list, good_logits.unsqueeze(0)])
            
            #USER
            candidate_neg_user, one_hop_logits = self.kg_step_user(pos, users, adj_matrix, step=1)
            candidate_neg_user = self.filter_entity_user(candidate_neg_user, self.n_users)

            good_neg_user = candidate_neg_user
            good_logits_user = one_hop_logits

            neg_list_user = torch.cat([neg_list_user, good_neg_user.unsqueeze(0)])
            prob_list_user = torch.cat([prob_list_user, good_logits_user.unsqueeze(0)])

        return neg_list, prob_list, neg_list_user, prob_list_user

    def build_edge(self, adj_matrix):
        """build edges based on adj_matrix"""
        sample_edge = self.config.edge_threshold
        edge_matrix = adj_matrix

        n_node = edge_matrix.size(0)
        node_index = (
            torch.arange(n_node, device=edge_matrix.device)
            .unsqueeze(1)
            .repeat(1, sample_edge)
            .flatten()
        )
        neighbor_index = edge_matrix.flatten()
        #pdb.set_trace()
        edges = torch.cat((node_index.unsqueeze(1), neighbor_index.unsqueeze(1)), dim=1)
        return edges

    def kg_step_user(self, pos, user, adj_matrix, step):
        #x = self.entity_embedding
        edges = self.edges
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        entity_gcn_emb, user_gcn_emb, cor = self.gcn(user_emb,
                                                     item_emb,
                                                     self.latent_emb,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     self.interact_mat,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout)
        """knowledge graph embedding using gcn"""
        #gcn_embedding = torch.cat([entity_gcn_emb,user_gcn_emb], dim=0)
        gcn_embedding = torch.cat([user_gcn_emb,entity_gcn_emb], dim=0)
        #gcn_embedding = self.gcn(x, edges.t().contiguous())

        """use knowledge embedding to decide candidate negative items"""
        u_e = gcn_embedding[user]
        u_e = u_e.unsqueeze(dim=1)
        pos_e = gcn_embedding[pos]
        pos_e = pos_e.unsqueeze(dim=2)

        one_hop_user = adj_matrix[user]
        i_user_e = gcn_embedding[one_hop_user]

        p_entity = F.leaky_relu(u_e * i_user_e)
        p = torch.sum(p_entity,dim=-1)
        '''
        p = torch.matmul(p_entity, pos_e)
        p = p.squeeze()
        '''
        logits = F.softmax(p, dim=1)

        """sample negative items based on logits"""
        batch_size = logits.size(0)
        if step == 1:
            nid = torch.argmax(logits, dim=1, keepdim=True)
        else:
            n = self.config.num_sample
            _, indices = torch.sort(logits, descending=True)
            nid = indices[:, :n]
        row_id = torch.arange(batch_size, device=logits.device).unsqueeze(1)

        candidate_neg = one_hop_user[row_id, nid].squeeze()
        candidate_logits = torch.log(logits[row_id, nid]).squeeze()

        return candidate_neg, candidate_logits

    def kg_step_outfit(self, pos, user, adj_matrix, step):
        #x = self.entity_embedding
        edges = self.edges
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        entity_gcn_emb, user_gcn_emb, cor = self.gcn(user_emb,
                                                     item_emb,
                                                     self.latent_emb,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     self.interact_mat,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout)
        """knowledge graph embedding using gcn"""
        #gcn_embedding = torch.cat([entity_gcn_emb,user_gcn_emb], dim=0)
        gcn_embedding = torch.cat([user_gcn_emb,entity_gcn_emb], dim=0)
        #gcn_embedding = self.gcn(x, edges.t().contiguous())

        """use knowledge embedding to decide candidate negative items"""
        u_e = gcn_embedding[user]
        u_e = u_e.unsqueeze(dim=2)
        pos_e = gcn_embedding[pos]
        pos_e = pos_e.unsqueeze(dim=1)

        one_hop = adj_matrix[pos]
        i_e = gcn_embedding[one_hop]

        p_entity = F.leaky_relu(pos_e * i_e)
        p = torch.sum(p_entity, dim=-1)
        '''
        p = torch.matmul(p_entity, u_e)
        p = p.squeeze()
        '''
        logits = F.softmax(p, dim=1)

        """sample negative items based on logits"""
        batch_size = logits.size(0)
        if step == 1:
            nid = torch.argmax(logits, dim=1, keepdim=True)
        else:
            n = self.config.num_sample
            _, indices = torch.sort(logits, descending=True)
            nid = indices[:, :n]
        row_id = torch.arange(batch_size, device=logits.device).unsqueeze(1)

        candidate_neg = one_hop[row_id, nid].squeeze()
        candidate_logits = torch.log(logits[row_id, nid]).squeeze()

        return candidate_neg, candidate_logits

    def kg_step(self, pos, user, adj_matrix, step):
        #x = self.entity_embedding
        edges = self.edges
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        entity_gcn_emb, user_gcn_emb, cor = self.gcn(user_emb,
                                                     item_emb,
                                                     self.latent_emb,
                                                     self.edge_index,
                                                     self.edge_type,
                                                     self.interact_mat,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout)
        """knowledge graph embedding using gcn"""
        #gcn_embedding = torch.cat([entity_gcn_emb,user_gcn_emb], dim=0)
        gcn_embedding = torch.cat([user_gcn_emb,entity_gcn_emb], dim=0)
        #gcn_embedding = self.gcn(x, edges.t().contiguous())

        """use knowledge embedding to decide candidate negative items"""
        u_e = gcn_embedding[user]
        u_e = u_e.unsqueeze(dim=2)
        pos_e = gcn_embedding[pos]
        pos_e = pos_e.unsqueeze(dim=1)

        one_hop = adj_matrix[pos]
        i_e = gcn_embedding[one_hop]

        p_entity = F.leaky_relu(pos_e * i_e)
        p = torch.matmul(p_entity, u_e)
        p = p.squeeze()
        logits = F.softmax(p, dim=1)

        """sample negative items based on logits"""
        batch_size = logits.size(0)
        if step == 1:
            nid = torch.argmax(logits, dim=1, keepdim=True)
        else:
            n = self.config.num_sample
            _, indices = torch.sort(logits, descending=True)
            nid = indices[:, :n]
        row_id = torch.arange(batch_size, device=logits.device).unsqueeze(1)

        candidate_neg = one_hop[row_id, nid].squeeze()
        candidate_logits = torch.log(logits[row_id, nid]).squeeze()

        return candidate_neg, candidate_logits

    @staticmethod
    def prune_step(dis, negs, users, logits):
        with torch.no_grad():
            #pdb.set_trace()
            ranking = dis.rank(users, negs)

        """get most qualified negative item based on user-neg similarity"""
        indices = torch.argmax(ranking, dim=1)

        batch_size = negs.size(0)
        row_id = torch.arange(batch_size, device=negs.device).unsqueeze(1)
        indices = indices.unsqueeze(1)

        good_neg = negs[row_id, indices].squeeze()
        goog_logits = logits[row_id, indices].squeeze()

        return good_neg, goog_logits

    @staticmethod
    def filter_entity(neg, item_range):
        random_neg = torch.randint(
            int(item_range[0]), int(item_range[1] + 1), neg.size(), device=neg.device
        )
        neg[neg > item_range[1]] = random_neg[neg > item_range[1]]
        neg[neg < item_range[0]] = random_neg[neg < item_range[0]]
        return neg

    @staticmethod
    def filter_entity_outfit(neg, item_range):
        random_neg = torch.randint(
            int(item_range[0]), int(item_range[1] + 1), neg.size(), device=neg.device
        )
        neg[neg > item_range[1]] = random_neg[neg > item_range[1]]
        neg[neg < item_range[0]] = random_neg[neg < item_range[0]]
        return neg

    @staticmethod
    def filter_entity_user(neg, n_users):
        random_neg = torch.randint(0, n_users, neg.size(), device=neg.device
        )
        neg[neg > n_users] = random_neg[neg > n_users]
        neg[neg < 0] = random_neg[neg < 0]
        return neg


