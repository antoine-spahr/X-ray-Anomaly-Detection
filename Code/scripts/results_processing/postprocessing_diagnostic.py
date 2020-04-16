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
OUTPUT_PATH = r'../../../Outputs/JointDeepSVDD_2020_03_23_09h14_milestone4080/'
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
spliter.split_data(verbose=False)
train_df = spliter.get_subset('train')
valid_df = spliter.get_subset('valid')
test_df = spliter.get_subset('test')

df = test_df.copy() \
            .drop(columns=['patient_any_abnormal', 'body_part_abnormal', 'low_contrast', 'semi_label'])

# Load the scores
N_score = 0
for i, fname in enumerate(glob.glob(OUTPUT_PATH + 'results/*.json')):
    with open(fname) as f:
        results = json.load(f)
        df_scores = pd.DataFrame(data=results['embedding']['test']['scores'], columns=['idx', 'label', f'AD_scores_{i+1}']) \
                      .set_index('idx') \
                      .drop(columns=['label'])
        df = pd.merge(df, df_scores, how='inner', left_index=True, right_index=True)
        N_score += 1

# merge the data
#test_df.drop(columns=['patient_any_abnormal', 'body_part_abnormal', 'low_contrast', 'semi_label'], inplace=True)
#df_scores = pd.DataFrame(data=results['embedding']['test']['scores'], columns=['idx', 'label', 'AD_scores']).set_index('idx')
#df = pd.merge(test_df.reset_index(), df_scores, how='inner', left_index=True, right_index=True)
df.head()
#%%#############################################################################
#                         Score Distribution by body part                      #
################################################################################
i = 1
BodyPart_list = df.body_part.unique()

fig, axs = plt.subplots(1, 7, figsize=(24,3), sharex=True, sharey=True)
if transparent: fig.set_alpha(0)
for ax, part in zip(axs.reshape(-1), BodyPart_list):
    # plot abnormal distribution
    ax.hist(df[(df.body_part == part) & (df.abnormal_XR == 1)].loc[:,f'AD_scores_{i}'],
            bins=40, density=True, log=True, range=(df[f'AD_scores_{i}'].min(), df[f'AD_scores_{i}'].max()),
            color='coral', alpha=0.4)
    # plot normal distribution
    ax.hist(df[(df.body_part == part) & (df.abnormal_XR == 0)].loc[:,f'AD_scores_{i}'],
            bins=55, density=True, log=True, range=(df[f'AD_scores_{i}'].min(), df[f'AD_scores_{i}'].max()),
            color='limegreen', alpha=0.4)
    ax.set_title(part.title(), fontsize=12)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

fig.savefig(FIGURE_PATH + 'score_distribution.pdf', dpi=dpi, bbox_inches='tight')
plt.show()

#%%#############################################################################
#                         10 highest scores by body part                       #
################################################################################

img_transform = tf.Compose(tf.Grayscale(),
                           tf.ResizeMax(max_len=512),
                           tf.PadToSquare(),
                           tf.ToTorchTensor())
j = 1

for label, low_ad in zip([1,1,0,0], [False, True, False, True]):
    fig, axs = plt.subplots(7, 10, figsize=(20,14), gridspec_kw={'hspace':0.25, 'wspace':0.0})
    if transparent: fig.set_alpha(0.0)

    for ax_row, part in zip(axs, BodyPart_list):
        df_high = df[(df.body_part == part) & (df.abnormal_XR == label)].sort_values(by=f'AD_scores_{j}', ascending=low_ad)
        ax_row[0].text(-0.1, 0.5, part.title(), fontsize=12, fontweight='bold', ha='center', va='center', rotation=90, transform=ax_row[0].transAxes)

        for i, ax in enumerate(ax_row):
            img = PIL.Image.open(DATA_PATH + df_high.iloc[i]['filename'])
            ax.imshow(img_transform(img, None)[0].numpy()[0,:,:], cmap='gray')
            ax.set_axis_off()
            ax.set_title(f'scores {df_high.iloc[i][f"AD_scores_{j}"]:.0f}', fontsize=10)

    figname = f'{"least" if low_ad else "most"}_anomalous_{"ab" if label == 1 else ""}normal'
    fig.savefig(FIGURE_PATH + figname, dpi=dpi, bbox_inches='tight')
    plt.show()

#%%#############################################################################
#                              AUC by body part                                #
################################################################################
auc = []
auprc = []
for part in df.body_part.unique():
    df_part = df[df.body_part == part]
    auc.append([roc_auc_score(df_part['abnormal_XR'], df_part[f'AD_scores_{i+1}']) for i in range(N_score)])
    auprc.append([average_precision_score(df_part['abnormal_XR'], df_part[f'AD_scores_{i+1}']) for i in range(N_score)])

auc = np.array(auc).T
auprc = np.array(auprc).T

fig, axs = plt.subplots(1,2,figsize=(20,8))
if transparent: fig.set_alpha(0.0)

metric_barplot([auc], serie_names=['AUC'], group_names=df.body_part.unique(), colors=['coral'], ax=axs[0])
axs[0].set_ylabel('AUC [-]')
metric_barplot([auprc], serie_names=['AUPRC'], group_names=df.body_part.unique(), colors=['lightskyblue'], ax=axs[1])
axs[1].set_ylabel('AUPRC [-]')

fig.savefig(FIGURE_PATH + 'body_part_auc.pdf', dpi=dpi, bbox_inches='tight')
plt.show()
