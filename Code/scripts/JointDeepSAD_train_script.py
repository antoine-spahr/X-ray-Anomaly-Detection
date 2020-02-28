import torch
import torch.cuda
import logging
import numpy as np
import pandas as pd
import random
from datetime import datetime
import os
import sys
sys.path.append('../')

from src.datasets.MURADataset import MURA_TrainValidTestSplitter, MURA_Dataset
from src.models.JointDeepSAD import JointDeepSAD
from src.models.networks.AE_ResNet18_net import AE_ResNet18
from src.utils.utils import summary_string

################################################################################
#                                Settings                                      #
################################################################################
# Import Export
Experiment_Name = 'JointDeepSAD_'
DATA_PATH = r'../../data/PROCESSED/'
DATA_INFO_PATH = r'../../data/data_info.csv'
OUTPUT_PATH = r'../../Outputs/' + Experiment_Name + datetime.today().strftime('%Y_%m_%d_%Hh%M')+'/'
# make output dir
if not os.path.isdir(OUTPUT_PATH+'models/'): os.makedirs(OUTPUT_PATH+'model/')
if not os.path.isdir(OUTPUT_PATH+'results/'): os.makedirs(OUTPUT_PATH+'results/')
if not os.path.isdir(OUTPUT_PATH+'logs/'): os.makedirs(OUTPUT_PATH+'logs/')

# General
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
n_thread = 0
n_seeds = 4
seeds = [int(''.join(['1']*(i+1))) for i in range(n_seeds)]#[42, 4242, 424242, 42424242, ]
print_batch_progress = True

# Datasets
train_frac = 0.5
ratio_known_normal = 0.05
ratio_known_abnormal = 0.05
n_jobs_dataloader = 8
batch_size = 16
img_size = 512

# Training
lr = 1e-4
lr_milestone = [59]
n_epoch = 100
weight_decay = 1e-6
criterion_weight = (0.5, 0.5)
model_path_to_load = None

# Network
eta = 1.0 # <<<<< change to zero to get DeepSVDD (only unsupervized)
embed_dim = 256
ae_pretrain = False
ae_out_size = (1, img_size, img_size)

# Additional comment to add in logger
Note = None

################################################################################
#                                Training                                      #
################################################################################

def main(seed_i):
    """
    Train jointly the AutoEncoder and the DeepSAD model following Lukas Ruff et
    al. (2019) work adapted to the MURA dataset (preprocessing inspired from the
    work of Davletshina et al. (2020)). The network structure is a ResNet18
    AutoEncoder. The network is trained with two loss functions: a masked MSE
    loss for the reconstruction and the DeepSAD loss on the embedding. The
    Encoder is not initialized with weights trained on ImageNet (but it's possible).
    The thresholds on scores (AE scores and DeepSAD scores) are selected on the
    validation set as the one maximiing the F1-scores. The ROC AUC is reported
    on the test and validation set.
    """
    # initialize logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    try:
        print(logger.handlers)
        logger.handlers[1].stream.close()
        logger.removeHandler(logger.handlers[1])
    except IndexError:
        pass
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    log_file = OUTPUT_PATH + 'logs/' + f'log_{seed_i+1}.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # print path
    logger.info('Brief summary of experiment : \n' + main.__doc__)
    if Note is not None: logger.info(Note + '\n')
    logger.info(f'Log file : {log_file}')
    logger.info(f'Data path : {DATA_PATH}')
    logger.info(f'Outputs path : {OUTPUT_PATH}' + '\n')

    ############################## Make datasets ###############################
    # load data_info
    df_info = pd.read_csv(DATA_INFO_PATH)
    df_info = df_info.drop(df_info.columns[0], axis=1)
    # remove low contrast images (all black)
    df_info = df_info[df_info.low_contrast == 0]

    # Train Validation Test Split
    spliter = MURA_TrainValidTestSplitter(df_info, train_frac=train_frac,
                                          ratio_known_normal=ratio_known_normal,
                                          ratio_known_abnormal=ratio_known_abnormal, random_state=42)
    spliter.split_data(verbose=False)
    train_df = spliter.get_subset('train')
    valid_df = spliter.get_subset('valid')
    test_df = spliter.get_subset('test')
    # make datasets
    train_dataset = MURA_Dataset(train_df, data_path=DATA_PATH, load_mask=True,
                                 load_semilabels=True, output_size=img_size)
    valid_dataset = MURA_Dataset(valid_df, data_path=DATA_PATH, load_mask=True,
                                 load_semilabels=True, output_size=img_size)
    test_dataset = MURA_Dataset(test_df, data_path=DATA_PATH, load_mask=True,
                                load_semilabels=True, output_size=img_size)
    # print info to logger
    logger.info(f'Train fraction : {train_frac:.0%}')
    logger.info(f'Fraction knonw normal : {ratio_known_normal:.0%}')
    logger.info(f'Fraction known abnormal : {ratio_known_abnormal:.0%}')
    logger.info('Split Summary \n' + str(spliter.print_stat(returnTable=True)))
    logger.info('Online preprocessing pipeline : \n' + str(train_dataset.transform) + '\n')

    ################################ Set Up ####################################
    # Set seed
    seed = seeds[seed_i]
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        logger.info(f'Set seed {seed_i+1:02}/{n_seeds:02} to {seed}')

    # set number of thread
    if n_thread > 0:
        torch.set_num_threads(n_thread)
    # print info in logger
    logger.info(f'Device : {device}')
    logger.info(f'Number of thread : {n_thread}')
    logger.info(f'Number of dataloader worker for Joint DeepSAD : {n_jobs_dataloader}' + '\n')

    ######################### Networks Initialization ##########################
    net = AE_ResNet18(embed_dim=embed_dim, pretrain_ResNetEnc=ae_pretrain,
                      output_size=ae_out_size, return_embed=True)
    net = net.to(device)

    # initialization of the Model
    jointDeepSAD = JointDeepSAD(net, eta=eta)
    # add info to logger
    logger.info(f'Network : {net.__class__.__name__}')
    logger.info(f'Embedding dimension : {embed_dim}')
    logger.info(f'Autoencoder pretrained on ImageNet : {ae_pretrain}')
    logger.info(f'DeepSAD eta : {eta}')
    logger.info('Autoencoder architecture: \n' + summary_string(net, (1, img_size, img_size), device=str(device)) + '\n')

    if model_path_to_load:
        jointDeepSAD.load_model(model_path_to_load, map_location=device)
        logger.info(f'Model Loaded from {model_path_to_load}' + '\n')

    ################################ Training ##################################
    # add parameter info
    logger.info(f'Joint DeepSAD number of epoch : {n_epoch}')
    logger.info(f'Joint DeepSAD learning rate : {lr}')
    logger.info(f'Joint DeepSAD learning rate milestone : {lr_milestone}')
    logger.info(f'Joint DeepSAD weight_decay : {weight_decay}')
    logger.info(f'Joint DeepSAD optimizer : Adam')
    logger.info(f'Joint DeepSAD batch_size {batch_size}')
    logger.info(f'Joint DeepSAD number of dataloader worker : {n_jobs_dataloader}')
    logger.info(f'Joint DeepSAD criterion weighting : {criterion_weight[0]} Masked MSE + {criterion_weight[1]} DeepSAD loss' + '\n')

    # train DeepSAD
    jointDeepSAD.train(train_dataset, lr=lr, n_epoch=n_epoch, lr_milestone=lr_milestone,
                  batch_size=batch_size, weight_decay=weight_decay, device=device,
                  n_jobs_dataloader=n_jobs_dataloader,
                  print_batch_progress=print_batch_progress, criterion_weight=criterion_weight)

    # validate DeepSAD
    jointDeepSAD.validate(valid_dataset, device=device, n_jobs_dataloader=n_jobs_dataloader,
                 print_batch_progress=print_batch_progress, criterion_weight=criterion_weight)

    # test DeepSAD
    jointDeepSAD.test(test_dataset, device=device, n_jobs_dataloader=n_jobs_dataloader,
                 print_batch_progress=print_batch_progress, criterion_weight=criterion_weight)

    # save results
    jointDeepSAD.save_results(OUTPUT_PATH + f'results/JointDeepSAD_results_{seed_i+1}.json')
    logger.info('Test results saved at ' + OUTPUT_PATH + f'results/JointDeepSAD_results_{seed_i+1}.json' + '\n')
    # save model
    jointDeepSAD.save_model(OUTPUT_PATH + f'model/JointDeepSAD_model_{seed_i+1}.pt')
    logger.info('Model saved at ' + OUTPUT_PATH + f'model/JointDeepSAD_model_{seed_i+1}.pt')

if __name__ == '__main__':
    # train for each seeds
    for i in range(n_seeds):
        main(i)