import numpy as np
import pandas as pd
import glob
import json
import os
import sys
sys.path.append('../../')

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.results_processing import load_experiment_results, scores_as_df
from src.utils.results_processing import metric_barplot, plot_scores_dist1D, plot_scores_dist2D, plot_loss, metric_curves

################################################################################
#                                Settings                                      #
################################################################################
# Path to the experiments folder (outputs of train scripts)
EXPERIMENT_PATH = r'../../../Outputs/'
# names of the experiment(s) to process
exp_folders = ['DROCC_2020_04_02_11h56']
exp_names = ['DROCC']
SAVE_PATHES = [EXPERIMENT_PATH + folder + '/analysis/' for folder in exp_folders]
FIG_RES = 200
fontsize=12

################################################################################
#                              Load results                                    #
################################################################################

results_all = load_experiment_results(EXPERIMENT_PATH, exp_folders, exp_names)

################################################################################
#                         Analyse all experiments                              #
################################################################################

for SAVE_PATH, results_list, exp_name in zip(SAVE_PATHES, results_all.values(), exp_names):
    # make output folders if not present
    if not os.path.isdir(SAVE_PATH): os.makedirs(SAVE_PATH)
    if not os.path.isdir(SAVE_PATH+'combiner_params/'): os.makedirs(SAVE_PATH+'combiner_params/')
    if not os.path.isdir(SAVE_PATH+'scores_dist/'): os.makedirs(SAVE_PATH+'scores_dist/')
    if not os.path.isdir(SAVE_PATH+'AUC_tables/'): os.makedirs(SAVE_PATH+'AUC_tables/')

    ############################################################################
    #                              Plot loss                                   #
    ############################################################################

    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    cmap_name = 'Oranges'

    plot_loss(results_list, 'train/loss', cmap_name, ax=ax)
    ax.set_title(f'{exp_name} loss', fontsize=fontsize)

    ax.set_xlabel('Epochs', fontsize=fontsize)
    ax.set_ylabel('Epoch Loss [-]', fontsize=fontsize)
    ax.autoscale(axis='x', tight=True)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(SAVE_PATH + 'Loss_evolution.pdf', dpi=FIG_RES, bbox_inches='tight')

    ############################################################################
    #                           Plot AUC barplot                               #
    ############################################################################
    v_auc = np.zeros([len(results_list), 1])
    t_auc = np.zeros([len(results_list), 1])
    v_auprc = np.zeros([len(results_list), 1])
    t_auprc = np.zeros([len(results_list), 1])

    # get auc and auprc
    scores_names = ['embedding', 'reconstruction']
    for i, results in enumerate(results_list):
        v_auc[i]  = results['valid']['auc']
        t_auc[i]  = results['test']['auc']

        df_v = pd.DataFrame(data=np.array(results['valid']['scores']),
                            columns=['index', 'label', 'scores'])
        df_t = pd.DataFrame(data=np.array(results['test']['scores']),
                            columns=['index', 'label', 'scores'])

        v_auprc[i] = average_precision_score(df_v.label, df_v.scores)
        t_auprc[i] = average_precision_score(df_t.label, df_t.scores)


    fig, ax = plt.subplots(1, 1, figsize=(5,7))
    metric_barplot([v_auc, t_auc, v_auprc, t_auprc],
                   ['Validation AURC', 'Test AURC', 'Validation AUPRC', 'Test AUPRC'],
                   [f'{exp_name} anomaly score'],
                   colors=['tomato', 'coral', 'dodgerblue', 'cornflowerblue'], w=None, ax=ax, fontsize=fontsize,
                   jitter=False, jitter_color='lightcoral')

    ax.set_ylabel('AURC ; AUPRC [-]', fontsize=fontsize)
    ax.set_title('Validation and Test AUCs for the anomaly scores', fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(SAVE_PATH + 'AUCs_barplot.pdf', dpi=FIG_RES, bbox_inches='tight')

    # save AUC in csv
    for auc_data, name in zip([v_auc, t_auc], ['Validation', 'Test']):
        auc_df = pd.DataFrame(data=auc_data.transpose(), index=[f'{exp_name} anomaly score'])
        auc_df['mean'] = auc_df.mean(axis=1)
        auc_df['1.96std'] = 1.96 * auc_df.std(axis=1)
        auc_df.to_csv(SAVE_PATH + 'AUC_tables/' + name + '_AURC.csv')

    for auprc_data, name in zip([v_auprc, t_auprc], ['Validation', 'Test']):
        auprc_df = pd.DataFrame(data=auprc_data.transpose(), index=[f'{exp_name} anomaly score'])
        auprc_df['mean'] = auprc_df.mean(axis=1)
        auprc_df['1.96std'] = 1.96 * auprc_df.std(axis=1)
        auprc_df.to_csv(SAVE_PATH + 'AUC_tables/' + name + '_AUPRC.csv')

    ############################################################################
    #                          Plot ROC and PR Curves                          #
    ############################################################################

    fig, axs = plt.subplots(1, 2, figsize=(10,5))

    metric_curves(results_list, set='valid', curves=['roc', 'prc'], areas=False, ax=axs[0])
    axs[0].set_title('Anomaly scores validation curves', fontsize=fontsize)
    metric_curves(results_list, set='test', curves=['roc', 'prc'], areas=False, ax=axs[1])
    axs[1].set_title('Anomaly scores test curves', fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(SAVE_PATH + f'ROC_PRC_curves.pdf', dpi=FIG_RES, bbox_inches='tight')

    ############################################################################
    #                         Plot Scores Distributions                        #
    ############################################################################
    for i, results in enumerate(results_list):
        df = {}
        df['Validation'] = pd.DataFrame(data=np.array(results['valid']['scores']),
                            columns=['index', 'label', 'scores'])
        df['Test'] = pd.DataFrame(data=np.array(results['test']['scores']),
                            columns=['index', 'label', 'scores'])

        fig, axs = plt.subplots(1, 2, figsize=(10,5))
        for j, name in enumerate(['Validation', 'Test']):
            legend = False if j == 0 else True
            plot_scores_dist1D(df[name].scores, df[name].label, nbin=100, ax=axs[j], colors=['limegreen', 'Orangered'], density=True, alpha=0.2, legend=legend)
            axs[j].set_xlabel('Anomaly Scores', fontsize=fontsize)
            axs[j].set_title(name + ' Anomaly Scores', fontsize=fontsize)
            axs[j].set_ylabel('freq [-]', fontsize=fontsize)

        fig.tight_layout()
        fig.savefig(SAVE_PATH + f'scores_dist/scores_distribution_{i+1}.pdf', dpi=FIG_RES, bbox_inches='tight')
