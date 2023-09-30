from torch import nn, optim
import torch
import random
import config
from prepareData import prepare_data
# from save_scores import save_scores
from model import Model
import numpy as np
import utility
from sklearn import metrics
import time
import glob
import os
from math import cos, pi


"""
没有划分独立的验证集，直接使用测试集当作验证集。
对比模型和我们的标准一致。
"""


# BCELoss
class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, one_index, zero_index, target, output):
        loss_func = torch.nn.BCELoss()
        # output = torch.where(torch.isnan(output), torch.full_like(output, 0), output)
        loss_mean = loss_func(torch.cat((output[one_index], output[zero_index]), 0),
                             torch.cat((target[one_index], target[zero_index]), 0))
        return loss_mean


def calculate_metrics(y_score, y_predict, y_label):
    """
    score[0, 1], predict{0, 1}, label{0, 1}
    """
    accuracy_val = metrics.accuracy_score(y_label, y_predict)
    precision_val = metrics.precision_score(y_label, y_predict)
    recall_val = metrics.recall_score(y_label, y_predict)
    f1_val = metrics.f1_score(y_label, y_predict)
    val_auc = metrics.roc_auc_score(y_label, y_score)
    precision, recall, thresholds = metrics.precision_recall_curve(y_label, y_score)
    val_auprc = metrics.auc(recall, precision)
    return accuracy_val, precision_val, recall_val, f1_val, val_auc, val_auprc


def save_epoch_test_score(_config, epoch, score):
    np.savetxt('result_test/HiSHANMDA_balance5fold_SEED_{0}_FOLD_{1}_epoch_{2}.txt'.format(_config.SEED, _config.FOLD, epoch), score)


def delete_redundent_files(_config, best_epoch):
    files = glob.glob('result_test/HiSHANMDA_balance5fold_SEED_{0}_FOLD_{1}_epoch_*.txt'.format(_config.SEED, _config.FOLD))
    for file in files:
        epoch_nb = int(file.split('_')[-1].split('.')[0])
        if epoch_nb != best_epoch:
            os.remove(file)


def train(model, dataset, optimizer, _config):

    regression_crit = Myloss()
    one_index = dataset['train'][0].t().tolist()
    zero_index = dataset['train'][1].t().tolist()

    best_epoch = 0
    best_auc_plus_auprc = 0
    bad_counter = 0
    test_auc_values = []
    test_auprc_values = []
    test_auc_plus_auprc_values = []


    for epoch in range(1, _config.epoch+1):
        t_epoch_begin = time.time()
        model.train()
        model.zero_grad()
        score = model(dataset)
        loss = regression_crit(one_index, zero_index, dataset['adj_removed'], score)
        loss.backward()
        optimizer.step()

        test_auc, test_auprc, scores_save = training_perform_metrics(model, dataset, dataset['test'], epoch)
        test_auc_values.append(test_auc)
        test_auprc_values.append(test_auprc)
        test_auc_plus_auprc_values.append(test_auc + test_auprc)

        t_epoch_end = time.time()
        print('LOSS:  %.4f' % (loss.item()))
        print('epoch_cost_time: %.2f ' % (t_epoch_end - t_epoch_begin))

        if test_auc_plus_auprc_values[-1] > best_auc_plus_auprc:
            best_auc_plus_auprc = test_auc_plus_auprc_values[-1]
            best_epoch = epoch
            bad_counter = 0
            if epoch > 15:
                save_epoch_test_score(_config, epoch, scores_save)
        else:
            bad_counter += 1

        if bad_counter == _config.patience:
            break
    
    print('best epoch: ', best_epoch, ' best auc: ', test_auc_values[best_epoch-1], 'best auprc: ', test_auprc_values[best_epoch-1])
    best_auc_list.append(test_auc_values[best_epoch-1])
    best_auprc_list.append(test_auprc_values[best_epoch-1])
    best_epoch_list.append(best_epoch)
    delete_redundent_files(_config, best_epoch)



def training_perform_metrics(model, dataset, validate_set, epoch):
    model.eval()
    scorexx = model(dataset)
    scores_for_save = scorexx.detach().numpy()
    # scorexx = torch.where(torch.isnan(scorexx), torch.full_like(scorexx, 0), scorexx)
    vali_score = scorexx[validate_set['pos'].transpose()].tolist() + scorexx[validate_set['neg'].transpose()].tolist()
    vali_predict = [0 if j < 0.5 else 1 for j in vali_score]
    vali_label = [1 for _ in range(len(validate_set['pos']))] + [0 for _ in range(len(validate_set['neg']))]

    accuracy_val, precision_val, recall_val, f1_val, val_auc, val_auprc = calculate_metrics(
        vali_score, vali_predict, vali_label)

    # print('Epoch:', epoch,
    #     'validate: Acc: %.4f' % accuracy_val, 'Pre: %.4f' % precision_val, 'Recall: %.4f' % recall_val,
    #     'F1: %.4f' % f1_val, 'Val AUC: %.4f' % val_auc, 'Val AUPRC: %.4f' % val_auprc)

    # tradition metrics
    # tradi_test_pos_socre = scorexx[dataset['test']['pos'].transpose()].tolist()
    # mat = np.array(dataset['adj_origin'])
    # all_neg_index = np.where(mat == 0)
    # tradi_test_neg_score = scorexx[all_neg_index].tolist()
    # tra_test_score = tradi_test_pos_socre + tradi_test_neg_score
    # tra_test_label = [1 for _ in range(len(tradi_test_pos_socre))] + [0 for _ in range(len(tradi_test_neg_score))]
    #
    # tra_auc = metrics.roc_auc_score(tra_test_label, tra_test_score)
    # precision, recall, thresholds = metrics.precision_recall_curve(tra_test_label, tra_test_score)
    # tra_auprc = metrics.auc(recall, precision)

    print('Epoch:', epoch, 'validate AUC: %.4f' % val_auc, 'validate AUPRC: %.4f' % val_auprc)
    return val_auc, val_auprc, scores_for_save


_config = config.Config()
best_auc_list = []
best_auprc_list = []
best_epoch_list = []


torch.manual_seed(_config.torch_seed)


def main():

    for i in range(1, 6):
        t_begin = time.time()
        print('-'*50)
        _config.FOLD = i
        print('processing Fold: ', _config.FOLD)
        # 数据初始化
        dataset = prepare_data(_config)  # dataset这里是dict
        # 模型初始化
        model = Model(_config)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        train(model, dataset, optimizer, _config)

        t_end = time.time()
        print('FOLD', i, 'cost time %.2f ' % (t_end - t_begin))

    print('mean AUC: %.4f' % (sum(best_auc_list) / len(best_auc_list)), 'mean AUPRC: %.4f' %
          (sum(best_auprc_list) / len(best_auprc_list)))
    print('best epoch list: ', best_epoch_list)

    with open('result_record.txt', 'a') as f:
        f.write('SEED {0}, RS {1}, AUC = {2:.4f}, AUPRC = {3:.4f}'.format(_config.SEED, _config.torch_seed,
            (sum(best_auc_list) / len(best_auc_list)), (sum(best_auprc_list) / len(best_auprc_list))) + '\n')

if __name__ == "__main__":
    main()
