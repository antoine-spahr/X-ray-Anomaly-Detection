import numpy as np
import pandas as pd
import glob
import json
import os
import sys
sys.path.append('../')

from sklearn.metrics import roc_auc_score

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from src.postprocessing.scoresCombiner import ScoresCombiner
from src.utils.results_processing import load_experiment_results, scores_as_df
from src.utils.results_processing import metric_barplot, plot_scores_dist1D, plot_scores_dist2D

################################################################################
#                                Settings                                      #
################################################################################
# Path to the experiments folder (outputs of train scripts)
EXPERIMENT_PATH = r'../../Outputs/tests/'
# names of the experiment(s) to process
exp_folders = ['DeepSAD_2020_02_27_16h55']
exp_names = ['DeepSAD']
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
        combiner.save_param(SAVE_PATH + 'combiner_params/' + f'scores_params_{i}.json')

    ############################################################################
    #                              Plot loss                                   #
    ############################################################################

    fig, axs = plt.subplots(2, 1, figsize=(10,6))
    cmap_names = ['Oranges', 'Blues']
    #handles, labels = [], exp_names
    plot_loss(results_list, 'embedding/train/loss', cmap_names[0], ax=axs[0])
    axs[0].set_title('DeepSAD loss', fontsize=fontsize)
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

    # get auc
    scores_names = ['embedding', 'reconstruction']
    for i, results in enumerate(results_list):
        for j, scores_type in enumerate(scores_names):
            v_auc[i,j]  = results[scores_type]['valid']['auc']
            t_auc[i,j]  = results[scores_type]['test']['auc']
        v_auc[i, 2] = combined_auc_valid[i]
        t_auc[i, 2] = combined_auc_test[i]


    fig, ax = plt.subplots(1, 1, figsize=(8,6))
    metric_barplot([v_auc, t_auc],
                   ['Validation', 'Test'],
                   [name.title() for name in scores_names+['composite']],
                   colors=['lightgray', 'dimgray'], w=None, ax=ax, fontsize=fontsize,
                   jitter=True, jitter_color='lightcoral')

    ax.set_ylabel('AUC [-]', fontsize=fontsize)
    ax.set_title('Validation and Test AUC for the various scores', fontsize=fontsize)
    fig.tight_layout()
    fig.savefig(SAVE_PATH + 'AUC_barplot.pdf', dpi=FIG_RES, bbox_inches='tight')

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
            plot_scores_dist1D(df[name].scores_em, df[name].label, ax=axs[j,0], density=False)
            plot_scores_dist1D(df[name].scores_rec, df[name].label, ax=axs[j,1], density=False, legend=legend)
            plot_scores_dist1D(df[name].scores_comb, df[name].label, ax=axs[j,2], density=False)
            axs[j,0].set_xlabel('Embedding Scores', fontsize=fontsize)
            axs[j,0].set_title(name + ' Embedding Scores', fontsize=fontsize)
            axs[j,1].set_xlabel('Reconstruction Scores', fontsize=fontsize)
            axs[j,1].set_title(name + ' Reconstruction Scores', fontsize=fontsize)
            axs[j,2].set_xlabel('Combined Scores', fontsize=fontsize)
            axs[j,2].set_title(name + ' Combined Scores', fontsize=fontsize)
            axs[j,0].set_ylabel('Counts [-]', fontsize=fontsize)

            plot_scores_dist2D(df[name].scores_em, df[name].scores_rec, df[name].label, ax=axs[j,3])
            axs[j,3].set_xlabel('Embedding scores', fontsize=fontsize)
            axs[j,3].set_ylabel('Reconstruction scores', fontsize=fontsize)
            axs[j,3].set_title('Embedding vs Reconstruction scores', fontsize=fontsize)

        fig.tight_layout()
        fig.savefig(SAVE_PATH + f'scores_dist/scores_distribution_{i}.pdf', dpi=FIG_RES, bbox_inches='tight')
