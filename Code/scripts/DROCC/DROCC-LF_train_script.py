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
from src.models.DROCC import DROCC
from src.models.networks.ResNet18_binary import ResNet18_binary
from src.utils.utils import summary_string

################################################################################
#                                Settings                                      #
################################################################################
use_LFOC = True

# Import Export
Experiment_Name = 'DROCC'
if use_LFOC : Experiment_Name += '-LF'
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
n_seeds = 2#4
seeds = [int(''.join(['1']*(i+1))) for i in range(n_seeds)]
print_batch_progress = True

# Datasets
train_frac = 0.5
ratio_known_normal = 0.05
ratio_known_abnormal = 0.05
n_jobs_dataloader = 8
batch_size = 16
img_size = 512

# Training
r = img_size / 2 # sqrt(img_size ** 2) / 2 as suggested in Goyal et al. 2020
gamma = 0.5
mu = 0.25 # weight of adversarial sample in loss (typically between 0 and 1)

lr = 1e-3
lr_adv = 1e-1
lr_milestone = [25, 50]
n_epoch = 65
n_epoch_init = 5
n_epoch_adv = 5
weight_decay = 1e-6
model_path_to_load = None

# Network
pretrain = False

################################################################################
#                                Training                                      #
################################################################################

def main(seed_i):
    """
    Implementation of the semi-supervised DROCC-LF model proposed by Goyal et al (2020).
    The model uses a binary classifier with a ResNet18 backbone. The output is a
    logit that serves as anomaly score. The loss is computed as Binary Cross
    Entropy loss with logit. The training consist of few epoch trained only with
    the normal samples. Then each epoch starts with the generation of adversarial
    examples. The adversarial search is performed only on normal samples as we
    want the network to learn the manifold of normal samples. It uses a slightly
    modifided projection gradient descent algorithm (the sample is projected on
    a learnable elipsoid). Then the samples and adversarial samples are passed
    through the network similarly to a standard classification task. Note that
    the input samples are masked with the mask generated in the preprocesing steps.
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
    log_file = OUTPUT_PATH + 'logs/' + f'log_{seed_i+1}.txt'
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
    logger.info(f'Number of dataloader worker for {Experiment_Name} : {n_jobs_dataloader}' + '\n')

    ######################### Networks Initialization ##########################
    net = ResNet18_binary(pretrained=pretrain)
    net = net.to(device)

    # add info to logger
    logger.info(f'Network : {net.__class__.__name__}')
    logger.info(f'ResNet18 pretrained on ImageNet : {pretrain}')
    logger.info('Network architecture: \n' + summary_string(net, (1, img_size, img_size), device=str(device), batch_size=batch_size) + '\n')

    # initialization of the Model
    drocc = DROCC(net, r)

    if model_path_to_load:
        drocc.load_model(model_path_to_load, map_location=device)
        logger.info(f'Model Loaded from {model_path_to_load}' + '\n')

    ################################ Training ##################################
    # add parameter info
    logger.info(f'{Experiment_Name} radius r : {r}')
    logger.info(f'{Experiment_Name} gamma : {gamma}')
    logger.info(f'{Experiment_Name} adversarial importance mu : {mu}')
    logger.info(f'{Experiment_Name} number of initial epoch : {n_epoch_init}')
    logger.info(f'{Experiment_Name} number of epoch : {n_epoch}')
    logger.info(f'{Experiment_Name} number of adversarial search epoch: {n_epoch_adv}')
    logger.info(f'{Experiment_Name} learning rate : {lr}')
    logger.info(f'{Experiment_Name} adversarial search learning rate : {lr_adv}')
    logger.info(f'{Experiment_Name} learning rate milestone : {lr_milestone}')
    logger.info(f'{Experiment_Name} weight_decay : {weight_decay}')
    logger.info(f'{Experiment_Name} optimizer : Adam')
    logger.info(f'{Experiment_Name} batch_size {batch_size}')
    logger.info(f'{Experiment_Name} number of dataloader worker : {n_jobs_dataloader}')

    # train DROCC
    drocc.train(train_dataset, gamma=gamma, mu=mu, lr=lr, lr_adv=lr_adv,
                lr_milestone=lr_milestone, weight_decay=weight_decay,
                n_epoch=n_epoch, n_epoch_init=n_epoch_init, n_epoch_adv=n_epoch_adv,
                batch_size=batch_size, device=device, n_jobs_dataloader=n_jobs_dataloader,
                LFOC=use_LFOC, print_batch_progress=print_batch_progress)

    # validate DROCC
    drocc.validate(valid_dataset, device=device,
                   n_jobs_dataloader=n_jobs_dataloader,
                   print_batch_progress=print_batch_progress)

    # test DROCC
    drocc.test(test_dataset, device=device,
               n_jobs_dataloader=n_jobs_dataloader,
               print_batch_progress=print_batch_progress)

    # save results
    drocc.save_results(OUTPUT_PATH + f'results/{Experiment_Name}_results_{seed_i+1}.json')
    logger.info('Test results saved at ' + OUTPUT_PATH + f'results/{Experiment_Name}_results_{seed_i+1}.json' + '\n')

    # save model
    drocc.save_model(OUTPUT_PATH + f'model/{Experiment_Name}_model_{seed_i+1}.pt')
    logger.info('Model saved at ' + OUTPUT_PATH + f'model/{Experiment_Name}_model_{seed_i+1}.pt')

if __name__ == '__main__':
    # experiment for each seeds
    for i in range(n_seeds):
        main(i)
