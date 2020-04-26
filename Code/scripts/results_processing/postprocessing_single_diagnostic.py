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
OUTPUT_PATH = r'../../../Outputs/Single_bodypart/ARAE_hand_2020_04_21_19h39/'
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
# keep the body part
df_info = df_info[df_info.body_part == 'HAND']

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

# merge the data
#test_df.drop(columns=['patient_any_abnormal', 'body_part_abnormal', 'low_contrast', 'semi_label'], inplace=True)
#df_scores = pd.DataFrame(data=results['embedding']['test']['scores'], columns=['idx', 'label', 'AD_scores']).set_index('idx')
#df = pd.merge(test_df.reset_index(), df_scores, how='inner', left_index=True, right_index=True)
df.head()

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

    df_high = df[df.abnormal_XR == label].sort_values(by=f'AD_scores_{j}', ascending=low_ad)

    for i, ax in enumerate(axs.reshape(-1)):

        img = PIL.Image.open(DATA_PATH + df_high.iloc[i]['filename'])
        mask = PIL.Image.open(DATA_PATH + df_high.iloc[i]['mask_filename'])
        img, mask = img_transform(img, mask)
        ax.imshow(img.numpy()[0,:,:] * mask.numpy()[0,:,:], cmap='gray')
        ax.set_axis_off()
        ax.set_title(f'scores {df_high.iloc[i][f"AD_scores_{j}"]:.4f}', fontsize=10)

    figname = f'{"least" if low_ad else "most"}_anomalous_{"ab" if label == 1 else ""}normal'
    fig.savefig(FIGURE_PATH + figname, dpi=dpi, bbox_inches='tight')
    plt.show()
