import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import glob
import json
import time
import os
import sys
sys.path.append('../../')

from sklearn.metrics import f1_score, roc_auc_score

import src.datasets.MURADataset as MURA
from src.models.networks.AE_ResNet18_net import AE_ResNet18, ResNet18_Encoder
from src.models.optim.autoencoder_trainer import AutoEncoderTrainer
from src.models.optim.CustomLosses import MaskedMSELoss
from src.utils.utils import get_best_threshold, print_progessbar

PATH_TO_OUTPUTS = r'../../../Outputs/DeepSAD_2020_02_25_11h12/'
DATA_PATH = r'../../../data/PROCESSED/'
DATA_INFO_PATH = r'../../../data/data_info.csv'


def test_AE(ae_net, dataset, batch_size=16, n_jobs_dataloader=4, device='cuda'):
    # make test dataloader using image and mask
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, \
                                    shuffle=True, num_workers=n_jobs_dataloader)

    # MSE loss without reduction --> MSE loss for each output pixels
    criterion = MaskedMSELoss(reduction='none')

    # set to device
    ae_net = ae_net.to(device)
    criterion = criterion.to(device)

    # Testing
    epoch_loss = 0.0
    n_batch = 0
    start_time = time.time()
    idx_label_score = []
    # put network in evaluation mode
    ae_net.eval()

    with torch.no_grad():
        for b, data in enumerate(loader):
            input, label, mask, _, idx = data
            # put inputs to device
            input, label = input.to(device).float(), label.to(device)
            mask, idx = mask.to(device), idx.to(device)

            rec = ae_net(input)
            rec_loss = criterion(rec, input, mask)
            score = torch.mean(rec_loss, dim=tuple(range(1, rec.dim()))) # mean over all dimension per batch

            # append scores and label
            idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                        label.cpu().data.numpy().tolist(),
                                        score.cpu().data.numpy().tolist()))
            # overall batch loss
            loss = torch.sum(rec_loss) / torch.sum(mask)
            epoch_loss += loss.item()
            n_batch += 1

            print_progessbar(b, loader.__len__(), Name='\t\tBatch', Size=20)

    test_time = time.time() - start_time
    scores = idx_label_score
    return test_time, scores

# restructure the output of the DeepSAD_2020_25_02 because of format change
def main(replicat_i):
    # final results file
    results = {
        'reconstruction':{
            'train':{
                'time': None,
                'loss': None
            },
            'scores_threshold':None,
            'valid':{
                'time':None,
                'auc':None,
                'f1':None,
                'scores':None
            },
            'test':{
                'time':None,
                'auc':None,
                'f1':None,
                'scores':None
            }
        },
        'embedding':{
            'train':{
                'time': None,
                'loss': None
            },
            'scores_threshold':None,
            'valid':{
                'time':None,
                'auc':None,
                'f1':None,
                'scores':None
            },
            'test':{
                'time':None,
                'auc':None,
                'f1':None,
                'scores':None
            }
        }
    }

    # %% load the AE_results file and DeepSAD results file
    with open(PATH_TO_OUTPUTS + f'results/AE_results_{replicat_i}.json') as f:
        ae_results = json.load(f)

    with open(PATH_TO_OUTPUTS + f'results/DeepSAD_valid_results_{replicat_i}.json') as f:
        sad_results_valid = json.load(f)

    with open(PATH_TO_OUTPUTS + f'results/DeepSAD_test_results_{replicat_i}.json') as f:
        sad_results_test = json.load(f)

    # %% Recover AE and SAD train time and train loss
    results['embedding']['train']['time'] = sad_results_test['train_time']
    results['embedding']['train']['loss'] = sad_results_test['train_loss']

    results['reconstruction']['train']['time'] = ae_results['train_time']
    results['reconstruction']['train']['loss'] = ae_results['train_loss']

    # %% recover test and valid AUC and time
    results['embedding']['valid']['time'] = sad_results_valid['test_time']
    results['embedding']['valid']['auc'] = sad_results_valid['test_auc']
    results['embedding']['valid']['scores'] = sad_results_valid['test_scores']

    results['embedding']['test']['time'] = sad_results_test['test_time']
    results['embedding']['test']['auc'] = sad_results_test['test_auc']
    results['embedding']['test']['scores'] = sad_results_test['test_scores']

    results['reconstruction']['valid']['time'] = ae_results['test_time']
    results['reconstruction']['valid']['auc'] = ae_results['test_auc']

    # %% compute f1-scores and threshold
    scores_v = np.array(results['embedding']['valid']['scores'])[:,2]
    labels_v = np.array(results['embedding']['valid']['scores'])[:,1]
    scores_t = np.array(results['embedding']['test']['scores'])[:,2]
    labels_t = np.array(results['embedding']['test']['scores'])[:,1]
    thres, f1_valid = get_best_threshold(scores_v, labels_v, metric=f1_score)
    f1_test = f1_score(np.where(scores_t > thres, 1, 0), labels_t)

    results['embedding']['scores_threshold'] = thres
    results['embedding']['valid']['f1'] = f1_valid
    results['embedding']['test']['f1'] = f1_test

    ################################################################################
    # %% get the datasets
    df_info = pd.read_csv(DATA_INFO_PATH)
    df_info = df_info.drop(df_info.columns[0], axis=1)
    # remove low contrast images (all black)
    df_info = df_info[df_info.low_contrast == 0]

    # Train Validation Test Split
    spliter = MURA.MURA_TrainValidTestSplitter(df_info, train_frac=0.5,
                                          ratio_known_normal=0.05,
                                          ratio_known_abnormal=0.05, random_state=42)
    spliter.split_data(verbose=True)
    valid_df = spliter.get_subset('valid')
    test_df = spliter.get_subset('test')
    # make datasets
    valid_dataset = MURA.MURA_Dataset(valid_df, data_path=DATA_PATH, load_mask=True, load_semilabels=True, output_size=512)
    test_dataset = MURA.MURA_Dataset(test_df, data_path=DATA_PATH, load_mask=True, load_semilabels=True, output_size=512)

    ################################################################################
    # %% Load the AE model and compute valid and test scores of reconstruction
    print('>>> initializing AE')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # define model
    ae_net = AE_ResNet18(embed_dim=256, output_size=(1, 512, 512))
    ae_net = ae_net.to(device)
    # get the trained parameters
    model_dict = torch.load(PATH_TO_OUTPUTS + f'model/deepSAD_model_{replicat_i}.pt', map_location=device)
    ae_state_dict = model_dict['ae_net_dict']
    ae_net.load_state_dict(ae_state_dict)

    print('>>> AE initialized')
    print('>>> Validating AE')
    valid_time, valid_scores = test_AE(ae_net, valid_dataset, device=device)
    print('>>> Testing AE')
    test_time, test_scores = test_AE(ae_net, test_dataset, device=device)
    print('>>> Computing scores and thresholds')
    scores_v = np.array(valid_scores)[:,2]
    labels_v = np.array(valid_scores)[:,1]
    scores_t = np.array(test_scores)[:,2]
    labels_t = np.array(test_scores)[:,1]
    thres, f1_valid = get_best_threshold(scores_v, labels_v, metric=f1_score)
    auc_valid = roc_auc_score(labels_v, scores_v)
    f1_test = f1_score(np.where(scores_t > thres, 1, 0), labels_t)
    auc_test = roc_auc_score(labels_t, scores_t)
    print(f'>>> valid auc : {auc_valid}')
    print(f'>>> valid f1 : {f1_valid}')
    print(f'>>> test auc : {auc_test}')
    print(f'>>> test f1 : {f1_test}')

    results['reconstruction']['scores_threshold'] = thres
    results['reconstruction']['valid']['time'] = valid_time
    results['reconstruction']['valid']['auc'] = auc_valid
    results['reconstruction']['valid']['f1'] = f1_valid
    results['reconstruction']['valid']['scores'] = valid_scores
    results['reconstruction']['test']['time'] = test_time
    results['reconstruction']['test']['auc'] = auc_test
    results['reconstruction']['test']['f1'] = f1_test
    results['reconstruction']['test']['scores'] = test_scores

    # save corrected results on disk
    with open(PATH_TO_OUTPUTS + f'results/results_corrected_{replicat_i}.json', 'w') as f:
        json.dump(results, f)

    print('>>> results saved at' + PATH_TO_OUTPUTS + f'results/results_corrected_{replicat_i}.json')

if __name__ == '__main__':
    for i in [1,2,3,4]:
        main(i)
