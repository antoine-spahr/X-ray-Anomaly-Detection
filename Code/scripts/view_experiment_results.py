import numpy as np
import pandas as pd
import glob
import json
import os
import sys
sys.path.append('../')

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from src.postprocessing.scoresCombiner import ScoresCombiner
from src.utils.results_processing import load_experiment_results, scores_as_df
from src.utils.results_processing import metric_barplot, plot_scores_dist1D, plot_scores_dist2D, plot_loss, metric_curves

################################################################################
#                                Settings                                      #
################################################################################
# Path to the experiments folder (outputs of train scripts)
EXPERIMENT_PATH = r'../../Outputs/'
# names of the experiment(s) to process
exp_folders = ['DeepSVDD_2020_03_02_16h35']#['DeepSAD_2020_02_25_11h12']
exp_names = ['DeepSVDD']#['DeepSAD']
SAVE_PATHES = [EXPERIMENT_PATH + folder + '/analysis/' for folder in exp_folders]
FIG_RES = 200
fontsize=12

################################################################################
#                              Load results                                    #
################################################################################

results_all = load_experiment_results(EXPERIMENT_PATH, exp_folders, exp_names)
#results_list = results_all[exp_names[0]]

################################################################################
#                         Analyse all experiments                              #
################################################################################

for SAVE_PATH, results_list in zip(SAVE_PATHES, results_all.values()):
    # make output folders if not present
    if not os.path.isdir(SAVE_PATH): os.makedirs(SAVE_PATH)
    if not os.path.isdir(SAVE_PATH+'combiner_params/'): os.makedirs(SAVE_PATH+'combiner_params/')
    if not os.path.isdir(SAVE_PATH+'scores_dist/'): os.makedirs(SAVE_PATH+'scores_dist/')
    if not os.path.isdir(SAVE_PATH+'AUC_tables/'): os.makedirs(SAVE_PATH+'AUC_tables/')

    ############################################################################
    #                      Compute Composite Score                             #
    ############################################################################
    combined_scores_valid = []
    combined_auc_valid = []
    combined_scores_test = []
    combined_auc_test = []

    for i, results in enumerate(results_list):
        # compute composite score
        df_v = scores_as_df(results, 'valid')
        df_t = scores_as_df(results, 'test')
        combiner = ScoresCombiner()
        # fit on validation scores
        v_scores, v_auc = combiner.fit(np.array(df_v.scores_em),
                                       np.array(df_v.scores_rec),
                                       np.array(df_v.label),
                                       search_res=0.01,
                                       verbose=1, return_scores=3)
        # apply on test scores
        t_scores = combiner.compute_scores(np.array(df_t.scores_em),
                                        np.array(df_t.scores_rec))
        t_auc = roc_auc_score(np.array(df_t.label), t_scores)
        # Store
        combined_scores_valid.append(v_scores)
        combined_auc_valid.append(v_auc)
        combined_scores_test.append(t_scores)
        combined_auc_test.append(t_auc)
        # Save combiner params
        combiner.save_param(SAVE_PATH + 'combiner_params/' + f'scores_params_{i+1}.json')

    ############################################################################
    #                              Plot loss                                   #
    ############################################################################

    fig, axs = plt.subplots(2, 1, figsize=(10,6))
    cmap_names = ['Oranges', 'Blues']
    #handles, labels = [], exp_names
    plot_loss(results_list, 'embedding/train/loss', cmap_names[0], ax=axs[0])
    axs[0].set_title(f'{exp_names[i]} loss', fontsize=fontsize)
    plot_loss(results_list, 'reconstruction/train/loss', cmap_names[0], ax=axs[1])
    axs[1].set_title('Autoencoder loss', fontsize=fontsize)

    for ax in axs:
        ax.set_xlabel('Epochs', fontsize=fontsize)
        ax.set_ylabel('Epoch Loss [-]', fontsize=fontsize)
        ax.autoscale(axis='x', tight=True)
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(SAVE_PATH + 'Loss_evolution.pdf', dpi=FIG_RES, bbox_inches='tight')

    ############################################################################
    #                           Plot AUC barplot                               #
    ############################################################################
    v_auc = np.zeros([len(results_list), 3])
    t_auc = np.zeros([len(results_list), 3])
    v_auprc = np.zeros([len(results_list), 3])
    t_auprc = np.zeros([len(results_list), 3])

    # get auc and auprc
    scores_names = ['embedding', 'reconstruction']
    for i, results in enumerate(results_list):
        for j, scores_type in enumerate(scores_names):
            v_auc[i,j]  = results[scores_type]['valid']['auc']
            t_auc[i,j]  = results[scores_type]['test']['auc']
        v_auc[i, 2] = combined_auc_valid[i]
        t_auc[i, 2] = combined_auc_test[i]

        df_v = scores_as_df(results, 'valid')
        df_t = scores_as_df(results, 'test')

        v_auprc[i, 0] = average_precision_score(df_v.label, df_v.scores_em)
        t_auprc[i, 0] = average_precision_score(df_t.label, df_t.scores_em)
        v_auprc[i, 1] = average_precision_score(df_v.label, df_v.scores_rec)
        t_auprc[i, 1] = average_precision_score(df_t.label, df_t.scores_rec)
        v_auprc[i, 2] = average_precision_score(df_v.label, combined_scores_valid[i])
        t_auprc[i, 2] = average_precision_score(df_t.label, combined_scores_test[i])


    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    metric_barplot([v_auc, t_auc, v_auprc, t_auprc],
                   ['Validation AURC', 'Test AURC', 'Validation AUPRC', 'Test AUPRC'],
                   [name.title() for name in scores_names+['composite']],
                   colors=['tomato', 'coral', 'dodgerblue', 'cornflowerblue'], w=None, ax=ax, fontsize=fontsize,
                   jitter=False, jitter_color='lightcoral')

    ax.set_ylabel('AURC ; AUPRC [-]', fontsize=fontsize)
    ax.set_title('Validation and Test AUCs for various scores', fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(SAVE_PATH + 'AUCs_barplot.pdf', dpi=FIG_RES, bbox_inches='tight')

    # save AUC in csv
    for auc_data, name in zip([v_auc, t_auc], ['Validation', 'Test']):
        auc_df = pd.DataFrame(data=auc_data.transpose(), index=scores_names+['composite'])
        auc_df['mean'] = auc_df.mean(axis=1)
        auc_df['1.96std'] = 1.96 * auc_df.std(axis=1)
        auc_df.to_csv(SAVE_PATH + 'AUC_tables/' + name + '_AURC.csv')

    for auprc_data, name in zip([v_auprc, t_auprc], ['Validation', 'Test']):
        auprc_df = pd.DataFrame(data=auprc_data.transpose(), index=scores_names+['composite'])
        auprc_df['mean'] = auprc_df.mean(axis=1)
        auprc_df['1.96std'] = 1.96 * auprc_df.std(axis=1)
        auprc_df.to_csv(SAVE_PATH + 'AUC_tables/' + name + '_AUPRC.csv')

    ############################################################################
    #                          Plot ROC and PR Curves                          #
    ############################################################################

    fig, axs = plt.subplots(2, 2, figsize=(10,10))

    metric_curves(results_list, 'scores_em', set='valid', curves=['roc', 'prc'], areas=False, ax=axs[0,0])
    axs[0,0].set_title('Embedding scores validation curves', fontsize=fontsize)
    metric_curves(results_list, 'scores_em', set='test', curves=['roc', 'prc'], areas=False, ax=axs[1,0])
    axs[1,0].set_title('Embedding scores test curves', fontsize=fontsize)
    metric_curves(results_list, 'scores_rec', set='valid', curves=['roc', 'prc'], areas=False, ax=axs[0,1])
    axs[0,1].set_title('Reconstruction scores validation curves', fontsize=fontsize)
    metric_curves(results_list, 'scores_rec', set='test', curves=['roc', 'prc'], areas=False, ax=axs[1,1])
    axs[1,1].set_title('Reconstruction scores test curves', fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(SAVE_PATH + f'ROC_PRC_curves.pdf', dpi=FIG_RES, bbox_inches='tight')

    ############################################################################
    #                         Plot Scores Distributions                        #
    ############################################################################
    for i, results in enumerate(results_list):
        df = {}
        df['Validation'] = scores_as_df(results, 'valid')
        df['Validation']['scores_comb'] = combined_scores_valid[i]
        df['Test'] = scores_as_df(results, 'test')
        df['Test']['scores_comb'] = combined_scores_test[i]

        fig, axs = plt.subplots(2, 4, figsize=(16,8))
        for j, name in enumerate(['Validation', 'Test']):
            legend = False if j == 0 else True
            plot_scores_dist1D(df[name].scores_em, df[name].label, nbin=100, ax=axs[j,0], colors=['limegreen', 'Orangered'], density=True, alpha=0.2)
            plot_scores_dist1D(df[name].scores_rec, df[name].label, nbin=100, ax=axs[j,1], colors=['limegreen', 'Orangered'], density=True, legend=legend, alpha=0.2)
            plot_scores_dist1D(df[name].scores_comb, df[name].label, nbin=100, ax=axs[j,2], colors=['limegreen', 'Orangered'], density=True, alpha=0.2)
            axs[j,0].set_xlabel('Embedding Scores', fontsize=fontsize)
            axs[j,0].set_title(name + ' Embedding Scores', fontsize=fontsize)
            axs[j,1].set_xlabel('Reconstruction Scores', fontsize=fontsize)
            axs[j,1].set_title(name + ' Reconstruction Scores', fontsize=fontsize)
            axs[j,2].set_xlabel('Combined Scores', fontsize=fontsize)
            axs[j,2].set_title(name + ' Combined Scores', fontsize=fontsize)
            axs[j,0].set_ylabel('freq [-]', fontsize=fontsize)

            plot_scores_dist2D(np.log(np.array(df[name].scores_em)+ 1e-9), np.log(np.array(df[name].scores_rec)+ 1e-9), df[name].label, ax=axs[j,3], kde=True, scatter=False, alphas=[0.05, 0.4])
            axs[j,3].set_xlabel('Log Embedding scores', fontsize=fontsize)
            axs[j,3].set_ylabel('Log Reconstruction scores', fontsize=fontsize)
            axs[j,3].set_title('Log Embedding vs Log Reconstruction scores', fontsize=fontsize)
            axs[j,3].set_ylim([-10,-2.5])

        fig.tight_layout()
        fig.savefig(SAVE_PATH + f'scores_dist/scores_distribution_{i+1}.pdf', dpi=FIG_RES, bbox_inches='tight')

# %%
# results_list = results_all[exp_names[0]]
#
# fig, ax = plt.subplots(1, 1, figsize=(6,6))
# metric_curves(results_list, 'scores_em', set='test', curves=['roc', 'prc'], areas=False, ax=ax)
# plt.show()
