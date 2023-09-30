import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class Sub_channel(nn.Module):

    def __init__(self, in_dim, out1_dim, out2_dim, K):
        super(Sub_channel, self).__init__()

        self.K = K

        self.linear_1 = nn.Linear(in_dim, out1_dim)
        nn.init.xavier_uniform_(self.linear_1.weight)
        self.activation_1 = nn.LeakyReLU()

        self.linear_2 = nn.Linear(out1_dim, 2 * out1_dim)
        nn.init.xavier_uniform_(self.linear_2.weight)
        self.activation_2 = nn.LeakyReLU()

        self.linear_3 = nn.Linear(2 * out1_dim, out1_dim)
        nn.init.xavier_uniform_(self.linear_3.weight)
        self.activation_3 = nn.LeakyReLU()



    def forward(self, node_init_embed, k_neighbor_adj, neighbor_size):

        # 公式（4-1）  获得各个实例的特征表示
        ent_side_embed = torch.matmul(k_neighbor_adj, node_init_embed)  # shape: (k, nm+nd, _dim)
        ent_embed = self.activation_1(self.linear_1(torch.div(ent_side_embed + node_init_embed, neighbor_size)))  # shape: (k, nm+nd, _dim)

        # 聚合K个实例的特征表示，得到该motif视角下，节点的高阶结构特征。
        ent_sum_embed = torch.div(ent_embed.sum(dim=0), self.K)  # shape: (nm+nd, out1_dim)
        # 两层FNN
        ent_sum_embed = self.activation_2(self.linear_2(ent_sum_embed))  # shape: (nm+nd, out1_dim * 2)

        ent_sum_embed = self.activation_3(self.linear_3(ent_sum_embed))  # shape: (nm+nd, out1_dim)

        return ent_sum_embed


class Attention_layer_2ch(nn.Module):
    def __init__(self, in_features, out_features):
        super(Attention_layer_2ch, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.leakyrelu = nn.LeakyReLU()

        self.linear_W = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear_W.weight)

        self.linear_a = nn.Linear(out_features, 1)
        nn.init.xavier_uniform_(self.linear_a.weight)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, c1_embed, c2_embed):

        wc1 = self.linear_W(c1_embed)
        wc2 = self.linear_W(c2_embed)
        node_num = c1_embed.shape[0]
        alpha_ij = self.softmax(torch.cat((self.leakyrelu(self.linear_a(wc1)), self.leakyrelu(self.linear_a(wc2))), 1))
        mixed_channel_embed = alpha_ij[:, 0].reshape(node_num, 1) * wc1 + alpha_ij[:, 1].reshape(node_num, 1) * wc2
        return mixed_channel_embed


class Attention_layer_3ch(nn.Module):
    def __init__(self, in_features, out_features):
        super(Attention_layer_3ch, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.leakyrelu = nn.LeakyReLU()

        self.linear_W = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear_W.weight)

        self.linear_a = nn.Linear(out_features, 1)
        nn.init.xavier_uniform_(self.linear_a.weight)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, c1_embed, c2_embed, c3_embed):

        wc1 = self.linear_W(c1_embed)
        wc2 = self.linear_W(c2_embed)
        wc3 = self.linear_W(c3_embed)

        node_num = c1_embed.shape[0]
        alpha_ij = self.softmax(torch.cat((self.leakyrelu(self.linear_a(wc1)), self.leakyrelu(self.linear_a(wc2)), self.leakyrelu(self.linear_a(wc3))), 1))
        mixed_channel_embed = alpha_ij[:, 0].reshape(node_num, 1) * wc1 + alpha_ij[:, 1].reshape(node_num, 1) * wc2 + alpha_ij[:, 2].reshape(node_num, 1) * wc3
        return mixed_channel_embed


class Attention_layer_4ch(nn.Module):
    def __init__(self, in_features, out_features):
        super(Attention_layer_4ch, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.leakyrelu = nn.LeakyReLU()

        self.linear_W = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear_W.weight)

        self.linear_a = nn.Linear(out_features, 1)
        nn.init.xavier_uniform_(self.linear_a.weight)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, c1_embed, c2_embed, c3_embed, c4_embed):

        wc1 = self.linear_W(c1_embed)
        wc2 = self.linear_W(c2_embed)
        wc3 = self.linear_W(c3_embed)
        wc4 = self.linear_W(c4_embed)

        node_num = c1_embed.shape[0]
        alpha_ij = self.softmax(torch.cat((self.leakyrelu(self.linear_a(wc1)), self.leakyrelu(self.linear_a(wc2)), self.leakyrelu(self.linear_a(wc3)), self.leakyrelu(self.linear_a(wc4))), 1))
        mixed_channel_embed = alpha_ij[:, 0].reshape(node_num, 1) * wc1 + alpha_ij[:, 1].reshape(node_num, 1) * wc2 + alpha_ij[:, 2].reshape(node_num, 1) * wc3 + alpha_ij[:, 3].reshape(node_num, 1) * wc4
        return mixed_channel_embed
