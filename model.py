import torch
from torch import nn
from layers import Sub_channel, Attention_layer_2ch, Attention_layer_3ch, Attention_layer_4ch
import numpy as np
from torch_geometric.nn import conv
import torch.nn.functional as F


# for 标准实验
class Model(nn.Module):
    def __init__(self, _config):
        super(Model, self).__init__()

        self.in_dim = _config.in_dim
        self.out1_dim = _config.out1_dim
        self.out2_dim = _config.out2_dim
        self.out3_dim = _config.out3_dim
        self.m_size = _config.m_size
        self.d_size = _config.d_size
        self.K = _config.K
        self.dropout_p = _config.dropout_p
        self.alpha = _config.alpha

        self.node_init_embed = nn.Embedding(self.m_size + self.d_size,
                                            self.in_dim)  # (nm+nd,in_dim) ,0-494:miRNA,495-877:disease
        nn.init.xavier_uniform_(self.node_init_embed.weight)

        self.ch1 = Sub_channel(self.in_dim, self.out1_dim, self.out2_dim, self.K)
        self.ch2 = Sub_channel(self.in_dim, self.out1_dim, self.out2_dim, self.K)
        self.ch3 = Sub_channel(self.in_dim, self.out1_dim, self.out2_dim, self.K)
        self.ch4 = Sub_channel(self.in_dim, self.out1_dim, self.out2_dim, self.K)

        self.gcn_lf = conv.GCNConv(self.in_dim, self.out2_dim)
        # self.gcn_lf_out1 = conv.GCNConv(self.in_dim, self.out1_dim)  # for 1 channel test

        self.at2 = Attention_layer_2ch(self.out1_dim, self.out2_dim)  # 2 channel inputs
        self.at3 = Attention_layer_3ch(self.out1_dim, self.out2_dim)  # 3 channel inputs
        self.at4 = Attention_layer_4ch(self.out1_dim, self.out2_dim)  # 4 channel inputs

        self.at2_out2_out2 = Attention_layer_2ch(self.out2_dim, self.out2_dim)
        # self.at2_out2_out3 = Attention_layer_2ch(self.out2_dim, self.out3_dim)
        # self.at2_out1_out1 = Attention_layer_2ch(self.out1_dim, self.out1_dim)  # for 1 channel test
        # self.at3_out1_out1 = Attention_layer_3ch(self.out1_dim, self.out1_dim)

        self.linear_1 = nn.Linear(self.out1_dim * 3, self.out1_dim)
        nn.init.xavier_uniform_(self.linear_1.weight)
        self.activation_1 = nn.LeakyReLU()
        self.linear_2 = nn.Linear(self.out1_dim, self.out2_dim)
        nn.init.xavier_uniform_(self.linear_2.weight)
        self.activation_2 = nn.LeakyReLU()

        self.outlayer_sigmoid = nn.Sigmoid()

    def forward(self, input):
        # Sub_channel的三个参数： 节点初始embed, 采样的k个实例的邻接矩阵，实例中节点的数量
        node_c1_ent_embed = self.ch1(self.node_init_embed.weight, input['channel_1_k_neighbors'], 3)
        node_c2_ent_embed = self.ch2(self.node_init_embed.weight, input['channel_2_k_neighbors'], 4)
        # node_c3_ent_embed = self.ch3(self.node_init_embed.weight, input['channel_3_k_neighbors'], 4)
        # node_c4_ent_embed = self.ch4(self.node_init_embed.weight, input['channel_4_k_neighbors'], 4)

        # 一层GCN得到节点的低阶结构特征
        node_gcn_embed = torch.relu(self.gcn_lf(self.node_init_embed.weight, input['md_adj_withsl_edge_index']))
        # 高阶结构特征聚合
        node_embed = self.at2(node_c1_ent_embed, node_c2_ent_embed)  # 2 channel (out1, out2)
        # 高、低聚合
        node_embed = self.at2_out2_out2(node_embed, node_gcn_embed)  # (out2, out2)

        miRNA_embed = node_embed[:self.m_size]
        disease_embed = node_embed[self.m_size:]
        score = torch.matmul(miRNA_embed, disease_embed.t())
        return self.outlayer_sigmoid(score)