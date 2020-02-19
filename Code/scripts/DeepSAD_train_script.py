import torch
import torch.cuda
import torchsummary
import logging
import numpy as np
import pandas as pd
import random
from datetime import datetime
import os
import sys
sys.path.append('../')

from src.datasets.MURADataset import MURA_TrainValidTestSplitter, MURA_Dataset
from src.models.DeepSAD import DeepSAD
from src.models.networks.AE_ResNet18_net import ResNet18_Encoder, AE_ResNet18

# TO DO :
# 1) save epoch_loss (and ROC_AUC ?) at each epoch

################################################################################
#                                Settings                                      #
################################################################################
# Import Export
Experiment_Name = 'DeepSAD_'
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
seeds = [42, 4242, 424242, 42424242]
print_batch_progress = True

# Datasets
train_frac = 0.5
ratio_known_normal = 0.05
ratio_known_abnormal = 0.05
n_jobs_dataloader = 4
batch_size = 16#32

ae_n_jobs_dataloader = 4
ae_batch_size = 16#32

# Training
lr = 1e-4
lr_milestone = [0]#[399]
n_epoch = 2#600
weight_decay = 1e-6
pretrain = False
model_path_to_load = None

ae_lr = 1e-4
ae_lr_milestone = [0]#[399]
ae_n_epoch = 2#1000
ae_weight_decay = 1e-6

# Network
eta = 1.0 # <<<<< change to zero to get DeepSVDD (only unsupervized)
embed_dim = 256
ae_pretrain = True
ae_out_channel = 1

# Additional comment to add in logger
Note = None

################################################################################
#                                Training                                      #
################################################################################

def main():
    """

    """
    # initialize logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    log_file = OUTPUT_PATH + 'logs/' + 'log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # print path
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
    spliter = MURA_TrainValidTestSplitter(df_info.sample(frac=0.003), train_frac=train_frac,
                                          ratio_known_normal=ratio_known_normal,
                                          ratio_known_abnormal=ratio_known_abnormal, random_state=42)
    spliter.split_data(verbose=False)
    train_df = spliter.get_subset('train')
    valid_df = spliter.get_subset('valid')
    test_df = spliter.get_subset('test')
    # make datasets
    train_dataset = MURA_Dataset(train_df, data_path=DATA_PATH, load_mask=True, load_semilabels=True)
    valid_dataset = MURA_Dataset(valid_df, data_path=DATA_PATH, load_mask=True, load_semilabels=True)
    test_dataset = MURA_Dataset(test_df, data_path=DATA_PATH, load_mask=True, load_semilabels=True)
    # print info to logger
    logger.info(f'Train fraction : {train_frac:.0%}')
    logger.info(f'Fraction knonw normal : {ratio_known_normal:.0%}')
    logger.info(f'Fraction known abnormal : {ratio_known_abnormal:.0%}')
    logger.info('Split Summary \n' + str(spliter.print_stat(returnTable=True)))
    logger.info('Online preprocessing pipeline : \n' + str(train_dataset.transform) + '\n')

    ################################ Set Up ####################################
    # Set seed
    seed = seeds[0]
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
    logger.info(f'Number of dataloader worker for DeepSAD : {n_jobs_dataloader}')
    logger.info(f'Autoencoder number of dataloader worker : {ae_n_jobs_dataloader}' + '\n')

    ######################### Networks Initialization ##########################
    ae_net = AE_ResNet18(embed_dim=embed_dim, pretrain_ResNetEnc=ae_pretrain,
                         output_channel=ae_out_channel)
    net = ResNet18_Encoder(embed_dim=embed_dim, pretrained=False)
    # initialization of the Model
    deepSAD = DeepSAD(net, ae_net=ae_net, eta=eta)
    # add info to logger
    logger.info(f'Autoencoder : {ae_net.__class__.__name__}')
    logger.info(f'Encoder : {net.__class__.__name__}')
    logger.info(f'Embedding dimension : {embed_dim}')
    logger.info(f'Autoencoder pretrained on ImageNet : {ae_pretrain}')
    logger.info(f'DeepSAD eta : {eta}')
    #logger.info('Autoencoder architecture: \n' + torchsummary.summary(ae_net, (1, 512, 512), device=device) + '\n')

    if model_path_to_load:
        deepSAD.load_model(model_path_to_load, load_ae=True, map_location=device)
        logger.info(f'Model Loaded from {model_path_to_load}' + '\n')

    ############################## Pretraining #################################
    logger.info(f'Pretraining DeepSAD via Autoencoder : {pretrain}')
    if pretrain:
        # add parameter info
        logger.info(f'Autoencoder number of epoch : {ae_n_epoch}')
        logger.info(f'Autoencoder learning rate : {ae_lr}')
        logger.info(f'Autoencoder learning rate milestone : {ae_lr_milestone}')
        logger.info(f'Autoencoder weight_decay : {ae_weight_decay}')
        logger.info(f'Autoencoder optimizer : Adam')
        logger.info(f'Autoencoder batch_size {ae_batch_size}' + '\n')
        # train AE
        deepSAD.pretrain(train_dataset, valid_dataset, lr=ae_lr, n_epoch=ae_n_epoch,
                         lr_milestone=ae_lr_milestone, batch_size=ae_batch_size,
                         weight_decay=ae_weight_decay, device=device,
                         n_jobs_dataloader=ae_n_jobs_dataloader,
                         print_batch_progress=print_batch_progress)
        # save results
        deepSAD.save_ae_results(OUTPUT_PATH + 'results/AE_results.json')

    ################################ Training ##################################
    # add parameter info
    logger.info(f'DeepSAD number of epoch : {n_epoch}')
    logger.info(f'DeepSAD learning rate : {lr}')
    logger.info(f'DeepSAD learning rate milestone : {lr_milestone}')
    logger.info(f'DeepSAD weight_decay : {weight_decay}')
    logger.info(f'DeepSAD optimizer : Adam')
    logger.info(f'DeepSAD batch_size {batch_size}')
    logger.info(f'DeepSAD number of dataloader worker : {n_jobs_dataloader}' + '\n')
    # train DeepSAD
    deepSAD.train(train_dataset, lr=lr, n_epoch=n_epoch, lr_milestone=lr_milestone,
                  batch_size=batch_size, weight_decay=weight_decay, device=device,
                  n_jobs_dataloader=n_jobs_dataloader,
                  print_batch_progress=print_batch_progress)
    # validate DeepSAD
    deepSAD.test(valid_dataset, device=device, n_jobs_dataloader=n_jobs_dataloader,
                 print_batch_progress=print_batch_progress)
    deepSAD.save_results(OUTPUT_PATH + 'results/DeepSAD_valid_results.json')
    logger.info('Validation results saved at ' + OUTPUT_PATH + 'results/DeepSAD_valid_results.json')
    # test DeepSAD
    deepSAD.test(test_dataset, device=device, n_jobs_dataloader=n_jobs_dataloader,
                 print_batch_progress=print_batch_progress)
    deepSAD.save_results(OUTPUT_PATH + 'results/DeepSAD_test_results.json')
    logger.info('Test results saved at ' + OUTPUT_PATH + 'results/DeepSAD_test_results.json')
    # save model
    deepSAD.save_model(OUTPUT_PATH + 'model/deepSAD_model.pt')
    logger.info('Model saved at ' + OUTPUT_PATH + 'model/deepSAD_model.pt')

if __name__ == '__main__':
    main()

#%%
# import matplotlib.pyplot as plt
# from src.datasets.MURADataset import MURA_TrainValidTestSplitter, MURA_Dataset
# from src.models.networks.AE_ResNet18_net import ResNet18_Encoder
#
# DATA_PATH = '../../data/'
# df = pd.read_csv(DATA_PATH+'data_info.csv')
# df = df.drop(df.columns[0], axis=1)
# df = df[df.low_contrast == 0]
#
# spliter = MURA_TrainValidTestSplitter(df, train_frac=0.5, ratio_known_normal=0.05, ratio_known_abnormal=0.05)
# spliter.split_data(verbose=True)
#
# train_df = spliter.get_subset('train')
# valid_df = spliter.get_subset('valid')
# test_df = spliter.get_subset('test')
#
# datasetMURA = MURA_Dataset(train_df, data_path=DATA_PATH+'PROCESSED/', load_mask=True, load_semilabels=True)
# image_test, label, mask, semi_label, idx = datasetMURA.__getitem__(6543)
#
# print(datasetMURA.transform)
#
# net = ResNet18_Encoder(embed_dim=256, pretrained=True)
#
# torch.unsqueeze(image_test, dim=0).dtype
# net(torch.unsqueeze(image_test, dim=0))
#
# fig, ax = plt.subplots(1,1,figsize=(8,8))
# ax.set_title('Transformed sample from the MURA dataset')
# ax.imshow(image_test[0,:,:], cmap='Greys_r')
# plt.show()
