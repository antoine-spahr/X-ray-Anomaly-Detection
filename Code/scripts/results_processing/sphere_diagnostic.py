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
sys.path.append('../../')

import src.datasets.transforms as tf
import src.datasets.MURADataset as MURA
from src.utils.results_processing import metric_barplot

DATA_PATH = r'../../../data/PROCESSED/'
DATA_INFO_PATH = r'../../../data/data_info.csv'
OUTPUT_PATH = r'../../../Outputs/JointDMSAD_2020_04_25_09h35_150ep/'
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
    for i, fname in enumerate(glob.glob(OUTPUT_PATH + 'results/*.json')):
        with open(fname) as f:
            results = json.load(f)
            df_scores = pd.DataFrame(data=results['embedding'][set]['scores'], columns=['idx', 'label', f'AD_scores_{i+1}', 'Nsphere']) \
                          .set_index('idx') \
                          .drop(columns=['label'])
            df.append(pd.merge(df_tmp, df_scores, how='inner', left_index=True, right_index=True))
            N_score += 1

# concat valid and test
df = pd.concat(df, axis=0)

# keep only normal samples
df_normal = df[df.abnormal_XR == 0]

#%%#############################################################################
#                         Score Distribution by Sphere                         #
################################################################################
i = 1
sphere_list = np.sort(df.Nsphere.unique())
sphere_list

fig, axs = plt.subplots(1, len(sphere_list), figsize=(len(sphere_list)*3, 3), sharex=False, sharey=True)
if transparent: fig.set_alpha(0)

for ax, sphere_i in zip(axs.reshape(-1), sphere_list):
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
    ax.set_title(f'Sphere n°{sphere_i+1}', fontsize=12)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

fig.savefig(FIGURE_PATH + 'score_distribution_by_sphere.pdf', dpi=dpi, bbox_inches='tight')
plt.show()

#%%#############################################################################
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

fig, ax = plt.subplots(1, 1, figsize=(7,7))
if transparent: fig.set_alpha(0)

prev = [0]*len(df_count.Nsphere.unique())

cmap = matplotlib.cm.get_cmap('Spectral')
color_list = cmap(np.linspace(0, 1, 7))

for part, color in zip(BodyPart_list, color_list):
    ax.bar([f'Sphere n°{i+1}' for i in df_count.Nsphere.unique()], df_count.loc[part, 'Count'],
           width=0.75, bottom=prev, color=color, ec='k', lw=0, label=part.title())
    prev += df_count.loc[part, 'Count'].values

for i, count in enumerate(prev):
    ax.text(i, count+50, str(count), ha='center', fontsize=10, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], ncol=1, loc='upper left', bbox_to_anchor=(1.1, 1), frameon=False)
ax.set_ylabel('Counts [-]', fontsize=12)
ax.set_title('Valid and Test Normal Body part counts by hypersphere', fontsize=12, fontweight='bold')
fig.savefig(FIGURE_PATH + 'bodypart_by_sphere.pdf', dpi=dpi, bbox_inches='tight')
plt.show()

# %% horizontal
fig, ax = plt.subplots(1, 1, figsize=(19,5))
if transparent: fig.set_alpha(0)

prev = [0]*len(df_count.Nsphere.unique())

cmap = matplotlib.cm.get_cmap('Spectral')
color_list = cmap(np.linspace(0, 1, 7))

for part, color in zip(BodyPart_list, color_list):
    ax.barh([f'Sphere n°{i+1}' for i in df_count.Nsphere.unique()], df_count.loc[part, 'Count'],
           height=0.8, left=prev, color=color, ec='k', lw=0, label=part.title())
    prev += df_count.loc[part, 'Count'].values

for i, count in enumerate(prev):
    ax.text(count+50, i, str(count), va='center', fontsize=10, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, ncol=7, loc='lower center', bbox_to_anchor=(0.5, -0.25), frameon=False)
ax.set_xlabel('Counts [-]', fontsize=12)
ax.set_title('Valid and Test Normal Body part counts by hypersphere', fontsize=12, fontweight='bold')
#fig.savefig(FIGURE_PATH + 'bodypart_by_sphere.pdf', dpi=dpi, bbox_inches='tight')
plt.show()

#%%#############################################################################
#                                Sample by Sphere                              #
################################################################################
N_sph = df_normal.Nsphere.unique().shape[0]
n_h = 3
N_samp = 30#20
seed = 1
sphere_count = df_normal.Nsphere.value_counts()

img_transform = tf.Compose(tf.Grayscale(),
                           tf.ResizeMax(max_len=512),
                           tf.PadToSquare(),
                           tf.ToTorchTensor())

fig, axs = plt.subplots(n_h*N_sph, N_samp//n_h, figsize=(N_samp//n_h*2, n_h*N_sph*2), gridspec_kw={'hspace': 0.25, 'wspace': 0.0})
if transparent: fig.set_alpha(0)

for i, sphere in enumerate(np.sort(df.Nsphere.unique())):
    axs_sph = axs[n_h*i:n_h*(i+1),:]

    df_samp = df_normal[df_normal.Nsphere == sphere].sample(n=min(N_samp, sphere_count[sphere]), random_state=seed)

    for k, ax in enumerate(axs_sph.reshape(-1)):
         ax.set_axis_off()
         if k < df_samp.shape[0]:
             img = PIL.Image.open(DATA_PATH + df_samp.iloc[k]['filename'])
             mask = PIL.Image.open(DATA_PATH + df_samp.iloc[k]['mask_filename'])
             img, mask = img_transform(img, mask)
             ax.imshow(img.numpy()[0,:,:] * mask.numpy()[0,:,:], cmap='gray')

    # draw group rect
    pos = axs_sph[0,0].get_position()
    pos2 = axs_sph[0,1].get_position()
    pos3 = axs_sph[1,0].get_position()
    pos4 = axs_sph[-1,0].get_position()
    fig.patches.extend([plt.Rectangle((pos.x0-0.3*pos.width, pos4.y0-0.05*pos.height),
                            (pos2.x0-pos.x0)*(N_samp//n_h)+0.15*pos.width,
                            (pos.y0-pos3.y0)*(n_h) - (pos.y0-pos3.y1) + 0.1*pos.height,
                            fc='black', ec='black', alpha=1, zorder=-1,
                            transform=fig.transFigure, figure=fig)])
    fig.text(pos.x0-0.15*pos.width, (pos.y1+pos4.y0)/2, f'Sphere n°{sphere+1}', rotation=90, rotation_mode='anchor',
             fontsize=14, fontweight='bold', ha='center', va='center', color='white')

fig.savefig(FIGURE_PATH + 'sample_by_sphere.pdf', dpi=dpi, bbox_inches='tight')
plt.show()
