import torch
import torch.cuda
import logging
import numpy as np
import pandas as pd
import random
from datetime import datetime
import os
import sys
sys.path.append('../../')

from src.datasets.MURADataset import MURA_TrainValidTestSplitter, MURA_Dataset
from src.models.JointDeepSAD import JointDeepSAD
from src.models.networks.AE_ResNet18_dual import AE_SVDD_Hybrid
from src.utils.utils import summary_string

################################################################################
#                                Settings                                      #
################################################################################
# Import Export
Experiment_Name = 'JointDSAD_Frac'
DATA_PATH = r'../../../data/PROCESSED/'
DATA_INFO_PATH = r'../../../data/data_info.csv'
OUTPUT_PATH = r'../../../Outputs/' + Experiment_Name + '_' + datetime.today().strftime('%Y_%m_%d_%Hh%M')+'/'
# make output dir
if not os.path.isdir(OUTPUT_PATH+'models/'): os.makedirs(OUTPUT_PATH+'model/')
if not os.path.isdir(OUTPUT_PATH+'results/'): os.makedirs(OUTPUT_PATH+'results/')
if not os.path.isdir(OUTPUT_PATH+'logs/'): os.makedirs(OUTPUT_PATH+'logs/')

# General
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
n_thread = 0
seed = 1
print_batch_progress = True

# Datasets
train_frac = 0.5
ratio_known_normal = 0.05
ratio_known_abnormal = [0.010, 0.050, 0.100]
n_jobs_dataloader = 8
batch_size = 16
img_size = 512

# Training
lr = 1e-4
lr_milestone = [40,80]
n_epoch = 100
n_epoch_pretrain = 5
weight_decay = 1e-6
criterion_weight = (0.5, 0.5)
model_path_to_load = None

# Network
eta = 1.0 # <<<<< change to zero to get DeepSVDD (only unsupervized)
ae_pretrain = False
ae_out_size = (1, img_size, img_size)

################################################################################
#                                Training                                      #
################################################################################

def main(ratio_known_abnormal_i):
    """
    Joint DSAD experiment with different fraction of known abnormal sample : 0.0%
    0.25% 0.5% 1.0% 5.0% and 10.0% over a single replicate.
    """
    # initialize logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    try:
        logger.handlers[1].stream.close()
        logger.removeHandler(logger.handlers[1])
    except IndexError:
        pass
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    log_file = OUTPUT_PATH + 'logs/' + f'log_{ratio_known_abnormal_i:.2%}.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # print path and main docstring with experiment summary
    logger.info('Brief summary of experiment : \n' + main.__doc__)
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
                                          ratio_known_abnormal=ratio_known_abnormal_i, random_state=42)
    spliter.split_data(verbose=False)
    train_df = spliter.get_subset('train')
    valid_df = spliter.get_subset('valid')
    test_df = spliter.get_subset('test')
    # make datasets
    train_dataset = MURA_Dataset(train_df, data_path=DATA_PATH, load_mask=True,
                                 load_semilabels=True, output_size=img_size,
                                 data_augmentation=True)
    valid_dataset = MURA_Dataset(valid_df, data_path=DATA_PATH, load_mask=True,
                                 load_semilabels=True, output_size=img_size,
                                 data_augmentation=False)
    test_dataset = MURA_Dataset(test_df, data_path=DATA_PATH, load_mask=True,
                                 load_semilabels=True, output_size=img_size,
                                 data_augmentation=False)
    # print info to logger
    logger.info(f'Train fraction : {train_frac:.0%}')
    logger.info(f'Fraction knonw normal : {ratio_known_normal:.0%}')
    logger.info(f'Fraction known abnormal : {ratio_known_abnormal_i:.0%}')
    logger.info('Split Summary \n' + str(spliter.print_stat(returnTable=True)))
    logger.info('Online preprocessing pipeline : \n' + str(train_dataset.transform) + '\n')

    ################################ Set Up ####################################
    # Set seed
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        logger.info(f'Set seed to {seed}')

    # set number of thread
    if n_thread > 0:
        torch.set_num_threads(n_thread)

    # print info in logger
    logger.info(f'Device : {device}')
    logger.info(f'Number of thread : {n_thread}')
    logger.info(f'Number of dataloader worker for Joint DeepSAD : {n_jobs_dataloader}' + '\n')

    ######################### Networks Initialization ##########################
    net = AE_SVDD_Hybrid(pretrain_ResNetEnc=ae_pretrain, output_channels=ae_out_size[0], return_svdd_embed=True)
    net = net.to(device)

    # add info to logger
    logger.info(f'Network : {net.__class__.__name__}')
    logger.info(f'Autoencoder pretrained on ImageNet : {ae_pretrain}')
    logger.info(f'DeepSAD eta : {eta}')
    logger.info('Network architecture: \n' + summary_string(net, (1, img_size, img_size), device=str(device), batch_size=batch_size) + '\n')

    # initialization of the Model
    jointDeepSAD = JointDeepSAD(net, eta=eta)

    if model_path_to_load:
        jointDeepSAD.load_model(model_path_to_load, map_location=device)
        logger.info(f'Model Loaded from {model_path_to_load}' + '\n')

    ################################ Training ##################################
    # add parameter info
    logger.info(f'Joint DeepSAD number of epoch : {n_epoch}')
    logger.info(f'Joint DeepSAD number of pretraining epoch: {n_epoch_pretrain}')
    logger.info(f'Joint DeepSAD learning rate : {lr}')
    logger.info(f'Joint DeepSAD learning rate milestone : {lr_milestone}')
    logger.info(f'Joint DeepSAD weight_decay : {weight_decay}')
    logger.info(f'Joint DeepSAD optimizer : Adam')
    logger.info(f'Joint DeepSAD batch_size {batch_size}')
    logger.info(f'Joint DeepSAD number of dataloader worker : {n_jobs_dataloader}')
    logger.info(f'Joint DeepSAD criterion weighting : {criterion_weight[0]} Reconstruction loss + {criterion_weight[1]} SVDD embdedding loss' + '\n')

    # train DeepSAD
    jointDeepSAD.train(train_dataset, lr=lr, n_epoch=n_epoch, n_epoch_pretrain=n_epoch_pretrain,
                  lr_milestone=lr_milestone, batch_size=batch_size, weight_decay=weight_decay,
                  device=device, n_jobs_dataloader=n_jobs_dataloader,
                  print_batch_progress=print_batch_progress, criterion_weight=criterion_weight,
                  valid_dataset=None)

    # validate DeepSAD
    jointDeepSAD.validate(valid_dataset, device=device, n_jobs_dataloader=n_jobs_dataloader,
                 print_batch_progress=print_batch_progress, criterion_weight=criterion_weight)

    # test DeepSAD
    jointDeepSAD.test(test_dataset, device=device, n_jobs_dataloader=n_jobs_dataloader,
                 print_batch_progress=print_batch_progress, criterion_weight=criterion_weight)

    # save results
    jointDeepSAD.save_results(OUTPUT_PATH + f'results/JointDeepSAD_results_{ratio_known_abnormal_i:.2%}.json')
    logger.info('Test results saved at ' + OUTPUT_PATH + f'results/JointDeepSAD_results_{ratio_known_abnormal_i:.2%}.json')

    # save model
    jointDeepSAD.save_model(OUTPUT_PATH + f'model/JointDeepSAD_model_{ratio_known_abnormal_i:.2%}.pt')
    logger.info('Model saved at ' + OUTPUT_PATH + f'model/JointDeepSAD_model_{ratio_known_abnormal_i:.2%}.pt')

if __name__ == '__main__':
    # experiment for each ratio of known abnormal
    for ratio_i in ratio_known_abnormal:
        main(ratio_i)
