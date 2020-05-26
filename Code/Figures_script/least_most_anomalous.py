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
OUTPUT_PATH = r'../../Outputs/JointDeepSVDD_2020_03_23_09h14_milestone4080/'
FIGURE_PATH = r'../../Figures/'

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
                                           ratio_known_normal=0.00,
                                           ratio_known_abnormal=0.00, random_state=42)
spliter.split_data(verbose=True)
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
df.head()

#%%#############################################################################
#                         3 highest scores by body part                       #
################################################################################
j = 1
N = 3
BodyPart_list = df.body_part.unique()
img_transform = tf.Compose(tf.Grayscale(),
                           tf.ResizeMax(max_len=512),
                           tf.PadToSquare(),
                           tf.ToTorchTensor())

fig, axs = plt.subplots(7, 4*N, figsize=(6*N,10.5), gridspec_kw={'hspace':0.25, 'wspace':0.1})
if transparent: fig.set_alpha(0.0)

# least anomalous abnormal ; most anomalous abnormal ; least anomalous normal ; most anomalous normal
for k, (label, low_ad) in enumerate(zip([1,1,0,0], [False, True, False, True])):
    axs_sub = axs[:, N*k:N*(k+1)]

    for ax_row, part in zip(axs_sub, BodyPart_list):
        df_high = df[(df.body_part == part) & (df.abnormal_XR == label)].sort_values(by=f'AD_scores_{j}', ascending=low_ad)
        if k == 0:
            ax_row[0].text(-0.25, 0.5, part.title(), fontsize=12, fontweight='bold', ha='center', va='center', rotation=90, transform=ax_row[0].transAxes)

        for i, ax in enumerate(ax_row):
            img = PIL.Image.open(DATA_PATH + df_high.iloc[i]['filename'])
            mask = PIL.Image.open(DATA_PATH + df_high.iloc[i]['mask_filename'])
            img, mask = img_transform(img, mask)
            ax.imshow(img.numpy()[0,:,:] * mask.numpy()[0,:,:], cmap='gray')
            #ax.imshow(img_transform(img, None)[0].numpy()[0,:,:], cmap='gray')
            ax.set_axis_off()
            ax.set_title(f'scores {df_high.iloc[i][f"AD_scores_{j}"]:.0f}', fontsize=9, color='gray')

    # draw group rect
    pos = axs_sub[0,0].get_position()
    pos2 = axs_sub[0,1].get_position()
    pos3 = axs_sub[1,0].get_position()
    pos4 = axs_sub[-1,0].get_position()
    fig.patches.extend([plt.Rectangle((pos.x0-0.1*pos.width, pos4.y0-0.05*pos.height),
                                      (pos2.x0-pos.x0)*N - 0.1*pos.width,
                                      (pos.y0-pos3.y0)*7 - (pos.y0-pos3.y1) + 0.80*pos.height,
                                      fc='black', ec='black', alpha=1, zorder=-1,
                                      transform=fig.transFigure, figure=fig)])
    axs_sub[0,1].text(0.5, 1.5, f'{"least" if low_ad else "most"} anomalous {"ab" if label == 1 else ""}normal',
             fontsize=14, fontweight='bold', ha='center', va='center', color='lightgray', transform=axs_sub[0,1].transAxes)

fig.savefig(FIGURE_PATH + 'least_most_anomalous_DSVDD.pdf', dpi=dpi, bbox_inches='tight')
plt.show()
