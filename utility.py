import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import KFold
import random


#  平衡五折交叉验证的正负样本划分
def balanced_5fold_index(seed, fold):
    random.seed(seed)
    np.random.seed(seed)
    all_associations = pd.read_csv('data/all_miRNA_disease_pairs.csv')
    known_associations = all_associations.loc[all_associations['label'] == 1]
    pos_df = known_associations
    pos_df.reset_index(drop=True, inplace=True)

    # 正样本进行五折划分
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    pos_train_id_all, pos_test_id_all = [], []
    for train_idx, test_idx in kf.split(pos_df): #train_index与test_index为下标
        pos_train_id_all.append(np.array(pos_df.iloc[train_idx][['miRNA', 'disease', 'label']]))
        pos_test_id_all.append(np.array(pos_df.iloc[test_idx][['miRNA', 'disease', 'label']]))

    # 随机取与正样本相同数量的负样本，然后五折划分
    unknown_associations = all_associations.loc[all_associations['label'] == 0]
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=seed, axis=0)
    neg_df = random_negative
    neg_df.reset_index(drop=True, inplace=True)
    neg_train_id_all, neg_test_id_all = [], []
    for train_idx, test_idx in kf.split(neg_df): #train_index与test_index为下标
        neg_train_id_all.append(np.array(neg_df.iloc[train_idx][['miRNA', 'disease', 'label']]))
        neg_test_id_all.append(np.array(neg_df.iloc[test_idx][['miRNA', 'disease', 'label']]))

    # 正负样本合并
    train_id_all, test_id_all = [], []
    for i in range(5):
        train_id_all.append(np.vstack((pos_train_id_all[i], neg_train_id_all[i])))
        test_id_all.append(np.vstack((pos_test_id_all[i], neg_test_id_all[i])))

    # fold 1-5
    # 0:miRNA, 1:disease, 2:label; miRNA,disease begin with 0
    return train_id_all[fold-1], test_id_all[fold-1]