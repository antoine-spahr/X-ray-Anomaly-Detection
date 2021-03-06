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
from src.models.DeepSAD import DeepSAD
from src.models.networks.AE_ResNet18_net import ResNet18_Encoder, AE_ResNet18
from src.utils.utils import summary_string

################################################################################
#                                Settings                                      #
################################################################################
# Import Export
Experiment_Name = 'DeepSVDD_'
DATA_PATH = r'../../../data/PROCESSED/'
DATA_INFO_PATH = r'../../../data/data_info.csv'
OUTPUT_PATH = r'../../../Outputs/' + Experiment_Name + datetime.today().strftime('%Y_%m_%d_%Hh%M')+'/'
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
ratio_known_normal = 0.00 # no known samples in training and only normal samples
ratio_known_abnormal = 0.00
n_jobs_dataloader = 8
batch_size = 16
img_size = 512

ae_n_jobs_dataloader = 8
ae_batch_size = 16

# Training
lr = 1e-4
lr_milestone = [59]
n_epoch = 100
weight_decay = 1e-6
pretrain = True
model_path_to_load = None

ae_lr = 1e-4
ae_lr_milestone = [59]
ae_n_epoch = 100
ae_weight_decay = 1e-6

# Network
eta = 0.0 # only unsupervized
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
    Train a DeepSVDD model following Lukas Ruff et al. (2018) work and code structure
    of their work on DeepSAD (2019) adapted to the MURA dataset (preprocessing
    inspired from the work of Davletshina et al. (2020)). The DeepSAD network
    structure is a ResNet18 Encoder. The Encoder is pretrained via Autoencoder
    training. The Autoencoder itself is not initialized with weights trained on
    ImageNet. The best threshold on the scores is defined using the validation
    set as the one maximizing the F1-score. The ROC AUC is reported on the test
    and validation set. This experiment is an unsupervized version of the DeepSAD
    (i.e. without known samples).
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
    train_dataset = MURA_Dataset(train_df, data_path=DATA_PATH, load_mask=True, load_semilabels=True, output_size=img_size)
    valid_dataset = MURA_Dataset(valid_df, data_path=DATA_PATH, load_mask=True, load_semilabels=True, output_size=img_size)
    test_dataset = MURA_Dataset(test_df, data_path=DATA_PATH, load_mask=True, load_semilabels=True, output_size=img_size)
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
    logger.info(f'Number of dataloader worker for DeepSVDD : {n_jobs_dataloader}')
    logger.info(f'Autoencoder number of dataloader worker : {ae_n_jobs_dataloader}' + '\n')

    ######################### Networks Initialization ##########################
    ae_net = AE_ResNet18(embed_dim=embed_dim, pretrain_ResNetEnc=ae_pretrain,
                         output_size=ae_out_size)
    ae_net = ae_net.to(device)
    net = ResNet18_Encoder(embed_dim=embed_dim, pretrained=False)
    net = net.to(device)

    # initialization of the Model
    deepSAD = DeepSAD(net, ae_net=ae_net, eta=eta)
    # add info to logger
    logger.info(f'Autoencoder : {ae_net.__class__.__name__}')
    logger.info(f'Encoder : {net.__class__.__name__}')
    logger.info(f'Embedding dimension : {embed_dim}')
    logger.info(f'Autoencoder pretrained on ImageNet : {ae_pretrain}')
    logger.info(f'DeepSVDD eta : {eta}')
    logger.info('Autoencoder architecture: \n' + summary_string(ae_net, (1, img_size, img_size), device=str(device)) + '\n')

    if model_path_to_load:
        deepSAD.load_model(model_path_to_load, load_ae=True, map_location=device)
        logger.info(f'Model Loaded from {model_path_to_load}' + '\n')

    ############################## Pretraining #################################
    logger.info(f'Pretraining DeepSVDD via Autoencoder : {pretrain}')
    if pretrain:
        # add parameter info
        logger.info(f'Autoencoder number of epoch : {ae_n_epoch}')
        logger.info(f'Autoencoder learning rate : {ae_lr}')
        logger.info(f'Autoencoder learning rate milestone : {ae_lr_milestone}')
        logger.info(f'Autoencoder weight_decay : {ae_weight_decay}')
        logger.info(f'Autoencoder optimizer : Adam')
        logger.info(f'Autoencoder batch_size {ae_batch_size}' + '\n')
        # train AE
        deepSAD.pretrain(train_dataset, valid_dataset, test_dataset, lr=ae_lr,
                         n_epoch=ae_n_epoch, lr_milestone=ae_lr_milestone,
                         batch_size=ae_batch_size, weight_decay=ae_weight_decay,
                         device=device, n_jobs_dataloader=ae_n_jobs_dataloader,
                         print_batch_progress=print_batch_progress)

    ################################ Training ##################################
    # add parameter info
    logger.info(f'DeepSVDD number of epoch : {n_epoch}')
    logger.info(f'DeepSVDD learning rate : {lr}')
    logger.info(f'DeepSVDD learning rate milestone : {lr_milestone}')
    logger.info(f'DeepSVDD weight_decay : {weight_decay}')
    logger.info(f'DeepSVDD optimizer : Adam')
    logger.info(f'DeepSVDD batch_size {batch_size}')
    logger.info(f'DeepSVDD number of dataloader worker : {n_jobs_dataloader}' + '\n')

    # train DeepSAD
    deepSAD.train(train_dataset, lr=lr, n_epoch=n_epoch, lr_milestone=lr_milestone,
                  batch_size=batch_size, weight_decay=weight_decay, device=device,
                  n_jobs_dataloader=n_jobs_dataloader,
                  print_batch_progress=print_batch_progress)

    # validate DeepSAD
    deepSAD.validate(valid_dataset, device=device, n_jobs_dataloader=n_jobs_dataloader,
                 print_batch_progress=print_batch_progress)

    # test DeepSAD
    deepSAD.test(test_dataset, device=device, n_jobs_dataloader=n_jobs_dataloader,
                 print_batch_progress=print_batch_progress)

    # save results
    deepSAD.save_results(OUTPUT_PATH + f'results/DeepSVDD_results_{seed_i+1}.json')
    logger.info('Test results saved at ' + OUTPUT_PATH + f'results/DeepSVDD_results_{seed_i+1}.json' + '\n')
    # save model
    deepSAD.save_model(OUTPUT_PATH + f'model/DeepSVDD_model_{seed_i+1}.pt')
    logger.info('Model saved at ' + OUTPUT_PATH + f'model/DeepSVDD_model_{seed_i+1}.pt')

if __name__ == '__main__':
    # train for each seeds
    for i in range(n_seeds):
        main(i)
