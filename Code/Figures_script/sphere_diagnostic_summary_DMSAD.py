import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import PIL.Image
from sklearn.metrics import roc_auc_score, average_precision_score
import glob
import os
import sys
sys.path.append('../')

import src.datasets.transforms as tf
import src.datasets.MURADataset as MURA
from src.utils.results_processing import metric_barplot

DATA_PATH = r'../../data/PROCESSED/'
DATA_INFO_PATH = r'../../data/data_info.csv'
OUTPUT_PATH = r'../../Outputs/JointDMSAD_2020_05_08_12h06_lin/'
FIGURE_PATH = OUTPUT_PATH + 'analysis/diagnostic/'
if not os.path.isdir(FIGURE_PATH): os.makedirs(FIGURE_PATH)

transparent = True
dpi = 200

#%%#############################################################################
#                              Load scores & dataset                           #
################################################################################

# load data_info
df_info = pd.read_csv(DATA_INFO_PATH)
df_info = df_info.drop(df_info.columns[0], axis=1)
# remove low contrast images (all black)
df_info = df_info[df_info.low_contrast == 0]

# Perform the datasplit as in the training
spliter = MURA.MURA_TrainValidTestSplitter(df_info, train_frac=0.5,
                                           ratio_known_normal=0.05,
                                           ratio_known_abnormal=0.05, random_state=42)
spliter.split_data(verbose=True)
train_df = spliter.get_subset('train')
valid_df = spliter.get_subset('valid')
test_df = spliter.get_subset('test')
# load scores and merge them with the data information
df = []
for set, df_set in zip(['valid', 'test'], [valid_df, test_df]):
    df_tmp = df_set.copy() \
                   .drop(columns=['patient_any_abnormal', 'body_part_abnormal', 'low_contrast', 'semi_label'])

    N_score = 0
    for i, fname in enumerate(glob.glob(OUTPUT_PATH + 'results/with_embed/*.json')):
        with open(fname) as f:
            results = json.load(f)
            df_scores = pd.DataFrame(data=results['embedding'][set]['scores'], columns=['idx', 'label', f'AD_scores_{i+1}', 'Nsphere', 'em']) \
                          .set_index('idx') \
                          .drop(columns=['label'])
            df.append(pd.merge(df_tmp, df_scores, how='inner', left_index=True, right_index=True))
            N_score += 1

# concat valid and test
df = pd.concat(df, axis=0)

# keep only normal samples
df_normal = df[df.abnormal_XR == 0]

#%%#############################################################################
#                                    T-spines                                  #
################################################################################
from sklearn.manifold import TSNE
import torch
# load centers and radii
model = torch.load(OUTPUT_PATH+'model/JointDMSAD_model_1.pt', map_location='cpu')
R = model['R']
c = model['c']

embed = np.stack(df.em.values, axis=0)
embed = np.concatenate((c, embed), axis=0)

#%% T-SNE
tsne_em = TSNE(n_components=2).fit_transform(embed)

#%% separate center and points
tsne_c = tsne_em[:c.shape[0],:]
tsne_em = tsne_em[c.shape[0]:,:]
#%% Mask
normal_mask = (df.abnormal_XR == 0).values
abnormal_mask = (df.abnormal_XR == 1).values

border_mask = (np.abs(df.AD_scores_1) < 1e-1).values


# %%
fig, ax = plt.subplots(1, 1, figsize=(8,8))
ax.scatter(tsne_em[abnormal_mask, 0], tsne_em[abnormal_mask, 1], s=3, c='coral', marker='.', alpha=0.5)
ax.scatter(tsne_em[normal_mask, 0], tsne_em[normal_mask, 1], s=3, c='limegreen', marker='.', alpha=0.5)
#ax.scatter(tsne_em[border_mask, 0], tsne_em[border_mask, 1], s=5, c='k', marker='o', alpha=0.6)
#ax.scatter(tsne_c[:,0], tsne_c[:,1], s=20, c='black', marker='x', alpha=0.9)
ax.set_axis_off()
plt.show()

#%%#############################################################################
#                         Score Distribution by Sphere                         #
################################################################################
i = 1
sphere_list = np.sort(df.Nsphere.unique())

fig = plt.figure(figsize=(16,12))
if transparent: fig.set_alpha(0)
gs = fig.add_gridspec(3, 4, hspace=0.5, wspace=0.3)

axs = [fig.add_subplot(gs[k,0]) for k in range(3)] # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

for ax, sphere_i in zip(axs, sphere_list):
    # plot abnormal distribution
    ax.hist(df[(df.Nsphere == sphere_i) & (df.abnormal_XR == 1)].loc[:,f'AD_scores_{i}'],
            bins=40, density=False, log=True,
            range=(df[df.Nsphere == sphere_i].loc[:,f'AD_scores_{i}'].min(), df[df.Nsphere == sphere_i].loc[:,f'AD_scores_{i}'].max()),#range=(df[f'AD_scores_{i}'].min(), df[f'AD_scores_{i}'].max()),
            color='coral', alpha=0.4)

    # plot normal distribution
    ax.hist(df[(df.Nsphere == sphere_i) & (df.abnormal_XR == 0)].loc[:,f'AD_scores_{i}'],
            bins=40, density=False, log=True,
            range=(df[df.Nsphere == sphere_i].loc[:,f'AD_scores_{i}'].min(), df[df.Nsphere == sphere_i].loc[:,f'AD_scores_{i}'].max()),#range=(df[f'AD_scores_{i}'].min(), df[f'AD_scores_{i}'].max()),
            color='limegreen', alpha=0.4)
    ax.set_title(f'Sphere {sphere_i+1}', fontsize=12, fontweight='bold')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.set_xlabel('anomaly score')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

for ax, letter in zip(axs, ['A', 'B', 'C']):
    ax.text(-0.1, 1.02, letter, fontsize=15, fontweight='bold', transform=ax.transAxes)
axs[0].set_ylabel('log(count) [-]')

################################################################################
#                                   T-SNE                                      #
################################################################################

ax = fig.add_subplot(gs[0:2, 1:3])

ax.scatter(tsne_em[abnormal_mask, 0], tsne_em[abnormal_mask, 1], s=3, c='coral', marker='.', alpha=0.5)
ax.scatter(tsne_em[normal_mask, 0], tsne_em[normal_mask, 1], s=3, c='limegreen', marker='.', alpha=0.5)
#ax.scatter(tsne_em[border_mask, 0], tsne_em[border_mask, 1], s=5, c='k', marker='o', alpha=0.6)
#ax.scatter(tsne_c[:,0], tsne_c[:,1], s=20, c='black', marker='x', alpha=0.9)
ax.set_axis_off()
ax.set_title('t-SNE visualisation of test and validation samples', fontsize=12, fontweight='bold')

ax.text(0.0, 1.02, 'D', fontsize=15, fontweight='bold', transform=ax.transAxes)

################################################################################
#                               Body part by Sphere                            #
################################################################################
BodyPart_list = np.sort(df_normal.body_part.unique())
sphere_list = np.sort(df_normal.Nsphere.unique())
idx_df = pd.MultiIndex.from_product([BodyPart_list, sphere_list])

df_count = df_normal.groupby(by=['body_part', 'Nsphere']).count() \
                    .reindex(idx_df, fill_value=0) \
                    .iloc[:,0] \
                    .reset_index(level=1) \
                    .rename(columns={'patientID':'Count', 'level_1':'Nsphere'})

# horizontal
ax = fig.add_subplot(gs[2, 1:]) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

prev = [0]*len(df_count.Nsphere.unique())

cmap = matplotlib.cm.get_cmap('Spectral')
color_list = cmap(np.linspace(0, 1, 7))

for part, color in zip(BodyPart_list, color_list):
    ax.barh([f'Sphere {i+1}' for i in df_count.Nsphere.unique()], df_count.loc[part, 'Count'],
           height=0.8, left=prev, color=color, ec='k', lw=0, label=part.title())
    prev += df_count.loc[part, 'Count'].values

for i, count in enumerate(prev):
    ax.text(count+50, i, str(count), va='center', fontsize=10, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, ncol=7, loc='lower center', bbox_to_anchor=(0.5, -0.4), frameon=False)
ax.set_xlabel('Counts [-]', fontsize=12)
ax.set_title('Valid and Test Normal Body part counts by hypersphere', fontsize=12, fontweight='bold')

ax.text(-0.02, 1.02, 'E', fontsize=15, fontweight='bold', transform=ax.transAxes)


################################################################################
#                                  AUC by Sphere                               #
################################################################################
sphere_list = np.sort(df.Nsphere.unique())

ax = fig.add_subplot(gs[:2,3])

auc = [roc_auc_score(df[df.Nsphere == i].abnormal_XR, df[df.Nsphere == i].AD_scores_1) for i in sphere_list]
auc.append(roc_auc_score(df.abnormal_XR, df.AD_scores_1))

ax.bar(np.arange(len(sphere_list)+1), auc, color='darkgray')
for x, h in zip(np.arange(len(sphere_list)+1), auc):
    ax.text(x, h+0.02, f'{h:.2%}', va='bottom', ha='center', fontsize=11)
ax.set_xticks(np.arange(4))
ax.set_xticklabels([f'Sphere {i+1}' for i in sphere_list] + ['All'], rotation=45, ha='right')
ax.set_ylabel('AUC [-]')
ax.set_ylim([0,1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('AUC by sphere', fontsize=12, fontweight='bold')

ax.text(-0.1, 1.02, 'F', fontsize=15, fontweight='bold', transform=ax.transAxes)

fig.savefig(FIGURE_PATH + 'summary_by_sphere.pdf', dpi=dpi, bbox_inches='tight')
plt.show()
