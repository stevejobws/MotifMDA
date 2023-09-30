import csv
import torch as t
import random
import numpy as np
import math
import utility
from copy import deepcopy


# 得到每条边的index，
def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return t.LongTensor(edge_index)


# 邻接矩阵添加自环
def add_self_loop(matrix):
    temp_matrix = deepcopy(matrix)
    for i in range(temp_matrix.shape[0]):
        temp_matrix[i][i] = 1
    return temp_matrix


# 初始化邻接矩阵
def get_adj(opt):
    nm = 495  # number of miRNAs
    nd = 383  # number of diseases
    nc = 5430  # number of miRNA-disease associations
    nn = 495 * 383 - 5430  # number of unknown samples
    ConnectDate = np.loadtxt(opt.data_path + '/known disease-miRNA association number ID.txt', dtype=int) - 1
    adj = np.zeros((nm, nd))
    for i in range(nc):
        adj[ConnectDate[i, 0], ConnectDate[i, 1]] = 1
    return t.FloatTensor(adj)


# 去掉邻接矩阵中1个fold的正样本，用于训练
def adj_drop1Fold(opt, test_pos_index):
    adj = get_adj(opt)
    for i in range(len(test_pos_index)):
        adj[test_pos_index[i][0]][test_pos_index[i][1]] = 0
    return t.FloatTensor(adj)


# 计算miRNA高斯特征
def Getgauss_miRNA(adjacentmatrix, nm):
    """
    MiRNA Gaussian interaction profile kernels similarity
    """
    KM = np.zeros((nm, nm))
    gamaa = 1
    sumnormm = 0
    for i in range(nm):
        normm = np.linalg.norm(adjacentmatrix[i]) ** 2
        sumnormm = sumnormm + normm
    gamam = gamaa / (sumnormm / nm)
    for i in range(nm):
        for j in range(nm):
            KM[i, j] = math.exp(-gamam * (np.linalg.norm(adjacentmatrix[i] - adjacentmatrix[j]) ** 2))
    return KM


# 计算disease高斯特征
def Getgauss_disease(adjacentmatrix, nd):
    """
    Disease Gaussian interaction profile kernels similarity
    """
    KD = np.zeros((nd, nd))
    gamaa = 1
    sumnormd = 0
    for i in range(nd):
        normd = np.linalg.norm(adjacentmatrix[:, i]) ** 2
        sumnormd = sumnormd + normd
    gamad = gamaa / (sumnormd / nd)
    for i in range(nd):
        for j in range(nd):
            KD[i, j] = math.exp(-(gamad * (np.linalg.norm(adjacentmatrix[:, i] - adjacentmatrix[:, j]) ** 2)))
    return KD


# 整合disease相似性特征
def _getDS_integration(opt, AA):
    DS1 = np.loadtxt(opt.data_path + '/disease semantic similarity matrix 1.txt')
    DS2 = np.loadtxt(opt.data_path + '/disease semantic similarity matrix 2.txt')
    DS = (DS1 + DS2) / 2
    AA = np.array(AA)
    nd = DS.shape[0]
    KD = Getgauss_disease(AA, nd)
    DS_integration = np.zeros((nd, nd))
    for i in range(nd):
        for j in range(nd):
            if DS[i, j] > 0:
                DS_integration[i, j] = DS[i, j]
            else:
                DS_integration[i, j] = KD[i, j]
    return t.FloatTensor(DS_integration)


# 整合miRNA相似性特征
def _getFS_integration(opt, AA):
    FS = np.loadtxt(opt.data_path + '/miRNA functional similarity matrix.txt')
    AA = np.array(AA)
    nm = FS.shape[0]
    KM = Getgauss_miRNA(AA, nm)
    FS_integration = np.zeros((nm, nm))
    for i in range(nm):
        for j in range(nm):
            if FS[i, j] > 0:
                FS_integration[i, j] = FS[i, j]
            else:
                FS_integration[i, j] = KM[i, j]
    return t.FloatTensor(FS_integration)


# 整合miRNA、disease特征到md_adj（邻接矩阵）中
def generate_combined_adj(adj, mm, dd, T):
    # T is the threshold for similarity matrices
    nm = mm.shape[0]
    nd = dd.shape[0]
    md_adj = np.zeros((nm+nd, nm+nd))
    for i in range(adj.shape[0]):  # i for miRNA (0-494)
        for j in range(adj.shape[1]):  # j for disease 495+(0-382)
            if adj[i][j] > 0:
                md_adj[i][j + nm] = 1
                md_adj[j + nm][i] = 1

    for i in range(nm):
        for j in range(nm):
            if mm[i][j] >= T:
                md_adj[i][j] = 1
                md_adj[j][i] = 1
    for i in range(nd):
        for j in range(nd):
            if dd[i][j] >= T:
                md_adj[i + nm][j + nm] = 1
                md_adj[i + nm][j + nm] = 1

    # drop self_loop:
    for i in range(nm+nd):
        md_adj[i][i] = 0

    return t.FloatTensor(md_adj)


# 在邻域采样的实例存储在矩阵中，shape(k, i, j) k是采样的个数，i是目标节点index，j是实例节点的index
def fill_k_neighbors(entity_list, node_i, k_neighbors):
    for _k in range(len(entity_list)):
        for j in entity_list[_k]:
            k_neighbors[_k, node_i, j] = 1


# 采样k个实例
def sample_k_entities(k, entity_list, node_i, k_neighbors):
    if len(entity_list) >= k:
        # sample
        entity_list = random.sample(entity_list, k)
        # fill
        fill_k_neighbors(entity_list, node_i, k_neighbors)
    elif (len(entity_list) > 0) and (len(entity_list) < k):
        origin_list = deepcopy(entity_list)
        for _ in range(k-len(origin_list)):
            entity_list.append(random.choice(origin_list))
        fill_k_neighbors(entity_list, node_i, k_neighbors)    


def triangle_channel_entities(k, md_adj, opt):
    md_adj = np.array(md_adj)  # tensor transfer to np
    k_neighbors = np.zeros((k, md_adj.shape[0], md_adj.shape[1]))

    for i in range(md_adj.shape[0]):
        i_related_nodes = np.nonzero(md_adj[i])[0]
        i_related_entities = []
        for j in range(len(i_related_nodes) - 1):
            for l in range(j + 1, len(i_related_nodes)):
                # i_related_nodes[j], i_related_nodes[l] are connected or not
                if md_adj[i_related_nodes[j]][i_related_nodes[l]] == 1 or md_adj[i_related_nodes[l]][
                        i_related_nodes[j]] == 1:
                    i_related_entities.append((i_related_nodes[j], i_related_nodes[l]))
        sample_k_entities(k, i_related_entities, i, k_neighbors)
    # k_neighbors shape (k, nm+nd, nm+nd)
    md_adj = t.FloatTensor(md_adj)  # np transfer to tensor
    return t.FloatTensor(k_neighbors)


def L_channel_entities(k, md_adj, opt):
    md_adj = np.array(md_adj)  # tensor transfer to np
    k_neighbors = np.zeros((k, md_adj.shape[0], md_adj.shape[1]))

    for i in range(md_adj.shape[0]):
        i_related_nodes = np.nonzero(md_adj[i])[0]
        i_related_entities = []
        for j in i_related_nodes:
            for l in np.nonzero(md_adj[j])[0]:
                if l != i:
                    i_related_entities.append((j, l))
        sample_k_entities(k, i_related_entities, i, k_neighbors)
    # k_neighbors shape (k, nm+nd, nm+nd)
    md_adj = t.FloatTensor(md_adj)  # np transfer to tensor
    return t.FloatTensor(k_neighbors)


def xyyy_channel_entities(k, md_adj, opt):
    """
    不区分节点类型，中心节点为x，其他3个关联节点为y
    """
    md_adj = np.array(md_adj)  # tensor transfer to np
    k_neighbors = np.zeros((k, md_adj.shape[0], md_adj.shape[1]))

    for i in range(md_adj.shape[0]):
        i_related_nodes = np.nonzero(md_adj[i])[0]
        i_related_entities = []
        for j in random.sample([_ for _ in range(len(i_related_nodes) - 2)], (len(i_related_nodes) - 2) // 3):
            for l in range(j + 1, len(i_related_nodes) - 1):
                for m in range(l + 1, len(i_related_nodes)):
                    i_related_entities.append((i_related_nodes[j], i_related_nodes[l], i_related_nodes[m]))
        sample_k_entities(k, i_related_entities, i, k_neighbors)
    # k_neighbors shape (k, nm+nd, nm+nd)
    md_adj = t.FloatTensor(md_adj)  # np transfer to tensor
    return t.FloatTensor(k_neighbors)


def rectangle_channel_entities(k, md_adj, opt):
    """
    四边形的情况种类太多了。
    1. 全为x(y)类型, 1+1=2
    2. x:y=1:3(x:y=3:1), 1+1=2
    3. x:y=2:2, 2
    如果具体区分节点种类的话，共有六种。太复杂了，还是直接用rectangle把。
    """
    md_adj = np.array(md_adj)  # tensor transfer to np
    k_neighbors = np.zeros((k, md_adj.shape[0], md_adj.shape[1]))

    for i in range(md_adj.shape[0]):
        i_related_nodes = np.nonzero(md_adj[i])[0]
        i_related_entities = []
        # for j in range(len(i_related_nodes) - 1):
        for j in random.sample([_ for _ in range(len(i_related_nodes) - 1)], (len(i_related_nodes) - 1) // 3):
            for l in range(j + 1, len(i_related_nodes)):
                # j, l common related nodes
                j_related_nodes = np.nonzero(md_adj[i_related_nodes[j]])[0]
                l_related_nodes = np.nonzero(md_adj[i_related_nodes[l]])[0]
                common_list = list(set(j_related_nodes).intersection(set(l_related_nodes)))
                if len(common_list) > k:
                    common_list = random.sample(common_list, k)
                for m in common_list:
                    if m != i:
                        i_related_entities.append((i_related_nodes[j], i_related_nodes[l], m))
        sample_k_entities(k, i_related_entities, i, k_neighbors)
    # k_neighbors shape (k, nm+nd, nm+nd)
    md_adj = t.FloatTensor(md_adj)  # np transfer to tensor
    return t.FloatTensor(k_neighbors)


def mddd_channel_entities(k, md_adj, opt):
    """
    中心节点为m，其他3个关联节点为d
    """
    md_adj = np.array(md_adj)  # tensor transfer to np
    k_neighbors = np.zeros((k, md_adj.shape[0], md_adj.shape[1]))

    for i in range(md_adj.shape[0]):
        i_related_nodes = np.nonzero(md_adj[i])[0]
        i_related_miRNA = [node for node in i_related_nodes if node < opt.m_size]
        i_related_disease = [node for node in i_related_nodes if node >= opt.m_size]
        i_related_entities = []
        if i < opt.m_size:  # i is miRNA node
            # travelling i related 3 diseases noodes
            for d1 in range(len(i_related_disease) - 2):
                for d2 in range(d1 + 1, len(i_related_disease) - 1):
                    for d3 in range(d2 + 1, len(i_related_disease)):
                        i_related_entities.append((i_related_disease[d1], i_related_disease[d2], i_related_disease[d3]))
        else:               # i is disease node
            for m in i_related_miRNA:
                m_related_disease = [node for node in np.nonzero(md_adj[m])[0] if node >= opt.m_size]
                for d1 in range(len(m_related_disease) - 1):
                    if m_related_disease[d1] == i:
                        continue
                    for d2 in range(d1 + 1, len(m_related_disease)):
                        if m_related_disease[d2] == i:
                            continue
                        i_related_entities.append((m, m_related_disease[d1], m_related_disease[d2]))
        sample_k_entities(k, i_related_entities, i, k_neighbors)
    # k_neighbors shape (k, nm+nd, nm+nd)
    md_adj = t.FloatTensor(md_adj)  # np transfer to tensor
    return t.FloatTensor(k_neighbors)


def triangle2miRNA_channel_entities(k, md_adj, opt):
    md_adj = np.array(md_adj)  # tensor transfer to np
    k_neighbors = np.zeros((k, md_adj.shape[0], md_adj.shape[1]))

    for i in range(md_adj.shape[0]):
        i_related_nodes = np.nonzero(md_adj[i])[0]
        i_related_miRNA = [node for node in i_related_nodes if node < opt.m_size]
        i_related_disease = [node for node in i_related_nodes if node >= opt.m_size]
        i_related_entities = []
        if i < opt.m_size:  # i is miRNA node
            for j in i_related_miRNA:
                for l in i_related_disease:
                    if md_adj[j][l] == 1 or md_adj[l][j] == 1:
                        i_related_entities.append((j, l))
        else:               # i is disease node
            for j in range(len(i_related_miRNA) - 1):
                for l in range(j + 1, len(i_related_miRNA)):
                    if md_adj[i_related_miRNA[j]][i_related_miRNA[l]] == 1 or md_adj[i_related_miRNA[l]][i_related_miRNA[j]] == 1:
                        i_related_entities.append((i_related_miRNA[j], i_related_miRNA[l]))
        sample_k_entities(k, i_related_entities, i, k_neighbors)
    # k_neighbors shape (k, nm+nd, nm+nd)
    md_adj = t.FloatTensor(md_adj)  # np transfer to tensor
    return t.FloatTensor(k_neighbors)


def triangle2disease_channel_entities(k, md_adj, opt):
    md_adj = np.array(md_adj)  # tensor transfer to np
    k_neighbors = np.zeros((k, md_adj.shape[0], md_adj.shape[1]))

    for i in range(md_adj.shape[0]):
        i_related_nodes = np.nonzero(md_adj[i])[0]
        i_related_miRNA = [node for node in i_related_nodes if node < opt.m_size]
        i_related_disease = [node for node in i_related_nodes if node >= opt.m_size]
        i_related_entities = []
        if i < opt.m_size:  # i is miRNA node
            for j in range(len(i_related_disease) - 1):
                for l in range(j + 1, len(i_related_disease)):
                    if md_adj[i_related_disease[j]][i_related_disease[l]] == 1 or md_adj[i_related_disease[l]][i_related_disease[j]] == 1:
                        i_related_entities.append((i_related_disease[j], i_related_disease[l]))
        else:               # i is disease node
            for j in i_related_disease:
                for l in i_related_miRNA:
                    if md_adj[j][l] == 1 or md_adj[l][j] == 1:
                        i_related_entities.append((j, l))
        sample_k_entities(k, i_related_entities, i, k_neighbors)
    # k_neighbors shape (k, nm+nd, nm+nd)
    md_adj = t.FloatTensor(md_adj)  # np transfer to tensor
    return t.FloatTensor(k_neighbors)


def mdm_channel_entities(k, md_adj, opt):
    md_adj = np.array(md_adj)  # tensor transfer to np
    k_neighbors = np.zeros((k, md_adj.shape[0], md_adj.shape[1]))

    for i in range(md_adj.shape[0]):
        i_related_nodes = np.nonzero(md_adj[i])[0]
        i_related_miRNA = [node for node in i_related_nodes if node < opt.m_size]
        i_related_disease = [node for node in i_related_nodes if node >= opt.m_size]
        i_related_entities = []
        if i < opt.m_size:  # i is miRNA node
            for j in i_related_disease:
                for l in [m1 for m1 in np.nonzero(md_adj[j])[0] if m1 < opt.m_size]:
                    if l != i:
                        i_related_entities.append((j, l))
        else:               # i is disease node
            for j in range(len(i_related_miRNA) - 1):
                for l in range(j + 1, len(i_related_miRNA)):
                    i_related_entities.append((i_related_miRNA[j], i_related_miRNA[l]))
        sample_k_entities(k, i_related_entities, i, k_neighbors)
    # k_neighbors shape (k, nm+nd, nm+nd)
    md_adj = t.FloatTensor(md_adj)  # np transfer to tensor
    return t.FloatTensor(k_neighbors)


def dmd_channel_entities(k, md_adj, opt):
    md_adj = np.array(md_adj)  # tensor transfer to np
    k_neighbors = np.zeros((k, md_adj.shape[0], md_adj.shape[1]))

    for i in range(md_adj.shape[0]):
        i_related_nodes = np.nonzero(md_adj[i])[0]
        i_related_miRNA = [node for node in i_related_nodes if node < opt.m_size]
        i_related_disease = [node for node in i_related_nodes if node >= opt.m_size]
        i_related_entities = []
        if i < opt.m_size:  # i is miRNA node
            for j in range(len(i_related_disease) - 1):
                for l in range(j + 1, len(i_related_disease)):
                    i_related_entities.append((i_related_disease[j], i_related_disease[l]))
        else:  # i is disease node
            for j in i_related_miRNA:
                for l in [[d1 for d1 in np.nonzero(md_adj[j])[0] if d1 >= opt.m_size]]:
                    if l != i:
                        i_related_entities.append((j, l))
        sample_k_entities(k, i_related_entities, i, k_neighbors)
    # k_neighbors shape (k, nm+nd, nm+nd)
    md_adj = t.FloatTensor(md_adj)  # np transfer to tensor
    return t.FloatTensor(k_neighbors)


def rectangle1_channel_entities(k, md_adj, opt):
    """
    m:d=2:2, rectangle1 为相同类型的节点在对角线 (为HiSCMDA里的mmotif-6，是性能提升的那一类)
    """
    md_adj = np.array(md_adj)  # tensor transfer to np
    k_neighbors = np.zeros((k, md_adj.shape[0], md_adj.shape[1]))

    for i in range(md_adj.shape[0]):
        i_related_nodes = np.nonzero(md_adj[i])[0]
        i_related_miRNA = [node for node in i_related_nodes if node < opt.m_size]
        i_related_disease = [node for node in i_related_nodes if node >= opt.m_size]
        i_related_entities = []
        if i < opt.m_size:  # i is miRNA node
            for d1 in range(len(i_related_disease) - 1):
                for d2 in range(d1 + 1, len(i_related_disease)):
                    d1_related_nodes = [node for node in np.nonzero(md_adj[i_related_disease[d1]])[0] if node < opt.m_size]
                    d2_related_nodes = [node for node in np.nonzero(md_adj[i_related_disease[d2]])[0] if node < opt.m_size]
                    common_list = list(set(d1_related_nodes).intersection(set(d2_related_nodes)))
                    if len(common_list) > k:
                        common_list = random.sample(common_list, k)
                    for m1 in common_list:
                        if m1 != i:
                            i_related_entities.append((i_related_disease[d1], i_related_disease[d2], m1))
        else:              # i is disease node
            for m1 in range(len(i_related_miRNA) - 1):
                for m2 in range(m1 + 1, len(i_related_miRNA)):
                    m1_related_nodes = [node for node in np.nonzero(md_adj[i_related_miRNA[m1]])[0] if node >= opt.m_size]
                    m2_related_nodes = [node for node in np.nonzero(md_adj[i_related_miRNA[m2]])[0] if node >= opt.m_size]
                    common_list = list(set(m1_related_nodes).intersection(set(m2_related_nodes)))
                    if len(common_list) > k:
                        common_list = random.sample(common_list, k)
                    for d1 in common_list:
                        if d1 != i:
                            i_related_entities.append((i_related_miRNA[m1], i_related_miRNA[m2], d1))
        sample_k_entities(k, i_related_entities, i, k_neighbors)
    # k_neighbors shape (k, nm+nd, nm+nd)
    md_adj = t.FloatTensor(md_adj)  # np transfer to tensor
    return t.FloatTensor(k_neighbors)    


def rectangle2_channel_entities(k, md_adj, opt):
    """
    m:d=2:2, rectangle2 为相同类型的节点相连。
    """
    md_adj = np.array(md_adj)  # tensor transfer to np
    k_neighbors = np.zeros((k, md_adj.shape[0], md_adj.shape[1]))

    for i in range(md_adj.shape[0]):
        i_related_nodes = np.nonzero(md_adj[i])[0]
        i_related_miRNA = [node for node in i_related_nodes if node < opt.m_size]
        i_related_disease = [node for node in i_related_nodes if node >= opt.m_size]
        i_related_entities = []
        if i < opt.m_size:  # i is miRNA node
            for m1 in i_related_miRNA:
                for d1 in i_related_disease:
                    m1_related_nodes = [node for node in np.nonzero(md_adj[m1])[0] if node >= opt.m_size]   # m1 related d2
                    d1_related_nodes = [node for node in np.nonzero(md_adj[d1])[0] if node >= opt.m_size]   # d1 related d2    
                    common_list = list(set(m1_related_nodes).intersection(set(d1_related_nodes)))
                    if len(common_list) > k:
                        common_list = random.sample(common_list, k)
                    for d2 in common_list:
                        i_related_entities.append((m1, d1, d2))

        else:              # i is disease node
            for m1 in i_related_miRNA:
                for d1 in i_related_disease:
                    m1_related_nodes = [node for node in np.nonzero(md_adj[m1])[0] if node < opt.m_size]   # m1 related m2
                    d1_related_nodes = [node for node in np.nonzero(md_adj[d1])[0] if node < opt.m_size]   # d1 related m2    
                    common_list = list(set(m1_related_nodes).intersection(set(d1_related_nodes)))
                    if len(common_list) > k:
                        common_list = random.sample(common_list, k)
                    for m2 in common_list:
                        i_related_entities.append((m1, d1, m2))
        sample_k_entities(k, i_related_entities, i, k_neighbors)
    # k_neighbors shape (k, nm+nd, nm+nd)
    md_adj = t.FloatTensor(md_adj)  # np transfer to tensor
    return t.FloatTensor(k_neighbors)    


def prepare_data(opt):
    dataset = dict()
    # one_index[*][0]:miRAN_index 0-494, one_index[*][1]:disease_index 0-382
    train_id_all, test_id_all = utility.balanced_5fold_index(opt.SEED, opt.FOLD)
    # train data store
    one_index = train_id_all[train_id_all[:, 2] == 1][:, [0, 1]]
    zero_index = train_id_all[train_id_all[:, 2] == 0][:, [0, 1]]
    np.random.shuffle(one_index)
    np.random.shuffle(zero_index)
    one_tensor = t.LongTensor(one_index)
    zero_tensor = t.LongTensor(zero_index)
    dataset['train'] = [one_tensor, zero_tensor]

    # test data store  未改为tensor数据类型
    test_pos_index = test_id_all[test_id_all[:, 2] == 1][:, [0, 1]]
    test_neg_index = test_id_all[test_id_all[:, 2] == 0][:, [0, 1]]
    dataset['test'] = dict()
    dataset['test']['all'] = test_id_all
    dataset['test']['pos'] = test_pos_index
    dataset['test']['neg'] = test_neg_index

    # adj (nm, nd) matrix load
    dataset['adj_origin'] = get_adj(opt)  # 完整的邻接矩阵 (nm, nd), 从文件中读取
    dataset['adj_removed'] = adj_drop1Fold(opt, test_pos_index)  # 去掉测试集和验证集中正样本的邻接矩阵 (nm, nd)

    # md_adj (nm+nd, nm+nd) load
    dd_matrix = _getDS_integration(opt, dataset['adj_removed'])  # 相似性矩阵里是有自环的
    mm_matrix = _getFS_integration(opt, dataset['adj_removed'])
    dataset['md_adj'] = generate_combined_adj(dataset['adj_removed'], mm_matrix, dd_matrix, opt.Threshold)  # 按T筛选，去掉了自环

    # motif选择
    dataset['channel_1_k_neighbors'] = dmd_channel_entities(opt.K, dataset['md_adj'], opt)
    dataset['channel_2_k_neighbors'] = mddd_channel_entities(opt.K, dataset['md_adj'], opt)
    # dataset['channel_3_k_neighbors'] = rectangle1_channel_entities(opt.K, dataset['md_adj'], opt)
    # dataset['channel_4_k_neighbors'] = rectangle_channel_entities(opt.K, dataset['md_adj'], opt)

    # GCN需要自环
    dataset['md_adj_add_self_loof'] = add_self_loop(dataset['md_adj'])
    dataset['md_adj_withsl_edge_index'] = get_edge_index(dataset['md_adj_add_self_loof'])

    return dataset

