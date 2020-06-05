import numpy as np
import pandas as pd
import glob
import json
import os
import sys
sys.path.append('../')

from sklearn.metrics import roc_auc_score, roc_curve

from src.utils.results_processing import metric_barplot
from src.datasets.MURADataset import MURA_TrainValidTestSplitter

import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
FIGURE_PATH = '../../Figures/'
OUTPUT_PATH = '../../Outputs/'
expe_folders = ['AE_DSAD_2020_05_24_08h10',
                'AE_DMSAD_2020_05_25_08h33',
                'SimCLR_DSAD_2020_06_01_10h52',
                'SimCLR_DMSAD_2020_05_19_09h11']
pretrain = ['AE','AE','SimCLR','SimCLR']
model = ['DSAD','DMSAD','DSAD','DMSAD']

data_info_path = '../../data/data_info.csv'

def plot_tSNE_bodypart(tsne2D, body_part, ax, title='', legend=False):
    """
    plot a 2D t-SNE by body part.
    """
    cmap = matplotlib.cm.get_cmap('Set2')
    color_list = cmap(np.linspace(0.1,0.9,7))

    for bp, color in zip(np.unique(body_part), color_list):
        ax.scatter(tsne2D[body_part == bp, 0],
                   tsne2D[body_part == bp, 1],
                   s=2, color=color, marker='.', alpha=0.8)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, fontweight='bold')

    if legend:
        handles = [matplotlib.patches.Patch(facecolor=color) for color in color_list]
        leg_name = [bp.title() for bp in np.unique(body_part)]
        ax.legend(handles, leg_name, ncol=4, loc='upper center', frameon=False,
                  fontsize=12, bbox_to_anchor=(1, 0), bbox_transform=ax.transAxes)

def plot_tSNE_label(tsne2D, labels, ax, title='', legend=False):
    """
    plot a 2D t-SNE by labels.
    """
    color_dict = {1: 'coral', 0: 'limegreen'}

    for lab, color in color_dict.items():
        ax.scatter(tsne2D[labels == lab, 0],
                   tsne2D[labels == lab, 1],
                   s=2, color=color, marker='.', alpha=0.5)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, fontweight='bold')

    if legend:
        handles = [matplotlib.patches.Patch(facecolor=color, alpha=1) for color in color_dict.values()]
        leg_names = ['Normal' if lab == 0 else 'Abnormal' for lab in color_dict.keys()]
        ax.legend(handles, leg_names, ncol=2, loc='upper center', frameon=False,
                  fontsize=12, bbox_to_anchor=(1, 0), bbox_transform=ax.transAxes)

# def get_AUC_list(scores, labels, body_part):
#     """
#
#     """
#     auc_list = [roc_auc_score(labels, scores)]
#     name_list = ['All']
#     for bp in np.unique(body_part):
#         auc_list.append(roc_auc_score(labels[body_part == bp], scores[body_part == bp]))
#         name_list.append(bp.title())
#
#     return np.array(auc_list).reshape(1,-1), name_list

#%%#############################################################################
#                                   get data                                   #
################################################################################
df_info = pd.read_csv(data_info_path)
df_info = df_info.drop(df_info.columns[0], axis=1)
df_info = df_info[df_info.low_contrast == 0]
# Get valid and test set
spliter = MURA_TrainValidTestSplitter(df_info, train_frac=0.5,
                                      ratio_known_normal=0.05,
                                      ratio_known_abnormal=0.05, random_state=42)
spliter.split_data(verbose=False)
valid_df = spliter.get_subset('valid')
test_df = spliter.get_subset('test')

# %% Get representstion of first replicate
# load t-SNE representations of valid set
rep = 1
set = 'valid'
df_set = valid_df if set == 'valid' else test_df
df_sim = {'AE':[], 'SimCLR':[]}
df_ad = {'AE':{'DSAD':[], 'DMSAD':[]}, 'SimCLR':{'DSAD':[], 'DMSAD':[]}}

for expe, pre, mod in zip(expe_folders, pretrain, model):
    with open(OUTPUT_PATH + expe + f'/results/results_{rep}.json', 'r') as f:
        results = json.load(f)

    # representation of SimCLR or AE
    df_tmp = df_set.copy() \
                   .drop(columns=['patient_any_abnormal', 'body_part_abnormal', 'low_contrast', 'semi_label'])

    cols = ['idx', '512_embed', '128_embed'] if pre == 'SimCLR' else ['idx', 'label', 'AE_score', '512_embed', '128_embed']

    df_scores = pd.DataFrame(data=results[pre][set]['embedding'], columns=cols) \
                  .set_index('idx')

    df_sim[pre].append(pd.merge(df_tmp, df_scores, how='inner', left_index=True, right_index=True))

    # scores and embedding of D(M)SAD
    df_tmp = df_set.copy() \
                   .drop(columns=['patient_any_abnormal', 'body_part_abnormal', 'low_contrast', 'semi_label'])

    cols = ['idx', 'label', 'ad_score', 'sphere_idx','128_embed'] if mod == 'DMSAD' else  ['idx', 'label', 'ad_score', '128_embed']
    # if pre == 'AE' and mod == 'DMSAD':
    #     cols = ['idx', 'label', 'ad_score', '128_embed']

    df_scores = pd.DataFrame(data=results['AD'][set]['scores'], columns=cols) \
                  .set_index('idx') \
                  .drop(columns=['label'])
    df_ad[pre][mod].append(pd.merge(df_tmp, df_scores, how='inner', left_index=True, right_index=True))

# get all AUC
AUC_valid = {'AE':{}, 'SimCLR':{}}
AUC_test = {'AE':{}, 'SimCLR':{}}

for expe, pre, mod in zip(expe_folders, pretrain, model):
    AUC_rep_valid, AUC_rep_test = [], []
    for results_path in glob.glob(OUTPUT_PATH + expe + f'/results/*.json'):
        with open(results_path, 'r') as f:
            results = json.load(f)

        AUC_rep_valid.append(results['AD']['valid']['auc'])
        AUC_rep_test.append(results['AD']['test']['auc'])

    AUC_valid[pre][mod] = AUC_rep_valid
    AUC_test[pre][mod] = AUC_rep_test

#%%#############################################################################
#                          T-SNE initial Representation                        #
################################################################################
fig, axs = plt.subplots(2,2,figsize=(8,9), gridspec_kw=dict(hspace=0.3, wspace=0.05))

for col, pre in enumerate(['AE', 'SimCLR']):
    df = df_sim[pre][0]
    pretrain_name = 'Contrastive' if pre == 'SimCLR' else pre
    embed2D = np.stack(df['512_embed'].values, axis=0)
    labels = df.abnormal_XR.values
    body_part = df.body_part.values
    leg = True if col == 0 else False

    plot_tSNE_bodypart(embed2D, body_part, axs[0,col], pretrain_name, legend=leg)
    plot_tSNE_label(embed2D, labels, axs[1,col], legend=leg)

axs[0,0].text(-0.05, 0.5, 'By Body Part', fontsize=12,
              rotation=90, ha='center', va='center', transform=axs[0,0].transAxes)
axs[1,0].text(-0.05, 0.5, 'By Label', fontsize=12,
              rotation=90, ha='center', va='center', transform=axs[1,0].transAxes)

fig.savefig(FIGURE_PATH + 'Init_TSNE.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%#############################################################################
#                              T-SNE SAD Representation                        #
################################################################################


fig, axs = plt.subplots(2,2,figsize=(8,8), gridspec_kw=dict(hspace=0.05, wspace=0.05))

for col, pre in enumerate(['AE', 'SimCLR']):
    for row, mod in enumerate(['DSAD', 'DMSAD']):
        df = df_ad[pre][mod][0]
        pretrain_name = 'Contrastive' if pre == 'SimCLR' else pre
        embed2D = np.stack(df['128_embed'].values, axis=0)
        labels = df.abnormal_XR.values
        leg = True if (col == 0) and (row == 1) else False
        title = pretrain_name if row == 0 else ''
        plot_tSNE_label(embed2D, labels, axs[row,col], title=title, legend=leg)

axs[0,0].text(-0.05, 0.5, 'DSAD', fontsize=12,
              rotation=90, ha='center', va='center', transform=axs[0,0].transAxes)
axs[1,0].text(-0.05, 0.5, 'DMSAD', fontsize=12,
              rotation=90, ha='center', va='center', transform=axs[1,0].transAxes)

fig.savefig(FIGURE_PATH + 'AD_TSNE.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%#############################################################################
#                                     AUC Barplot                              #
################################################################################
group_name = []
auc_test, auc_valid = [], []
for pre in ['AE', 'SimCLR']:
    for mod in ['DSAD', 'DMSAD']:
        pretrain_name = 'Contrastive' if pre == 'SimCLR' else pre
        group_name.append(pretrain_name + '\n' + mod)
        auc_valid.append(np.array(AUC_valid[pre][mod]))
        auc_test.append(np.array(AUC_test[pre][mod]))

auc = [pd.DataFrame(auc_valid).T.values, pd.DataFrame(auc_test).T.values]

fig, ax = plt.subplots(1, 1, figsize=(7,6))

metric_barplot(auc, ['Validation', 'Test'], group_name, colors=['lightcoral', 'lightsalmon'],
              legend_kwargs=dict(loc='upper right', ncol=1, frameon=False, framealpha=0.0,
                        fontsize=12, bbox_to_anchor=(1, 1.05), bbox_transform=ax.transAxes))
ax.set_ylabel('AUC [-]')

fig.savefig(FIGURE_PATH + 'Contrastive_BarPlot.pdf', dpi=300, bbox_inches='tight')
plt.show()

np.nanmean(pd.DataFrame(auc_valid).T.values, axis=0)
np.nanstd(pd.DataFrame(auc_valid).T.values, axis=0)*1.96

np.nanmean(pd.DataFrame(auc_test).T.values, axis=0)
np.nanstd(pd.DataFrame(auc_test).T.values, axis=0)*1.96
