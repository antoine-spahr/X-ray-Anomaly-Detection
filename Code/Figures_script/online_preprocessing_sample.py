import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

import src.datasets.MURADataset as MURA

DATA_PATH = r'../../data/'
OUTPUT_PATH = r'../../Outputs/'
FIGURE_PATH = r'../../Figures/'
FIG_RES = 200 # dpi
transparent = False
save_fig = True

# %% Get data info
df = pd.read_csv(DATA_PATH+'data_info.csv')
df = df.drop(df.columns[0], axis=1)
df = df[df.low_contrast == 0]

# %% split data
spliter = MURA.MURA_TrainValidTestSplitter(df, train_frac=0.5, ratio_known_normal=0.05, ratio_known_abnormal=0.05, random_state=42)
spliter.split_data(verbose=True)

train_df = spliter.get_subset('train')
test_df = spliter.get_subset('test')

# %% make dataset
dataset = MURA.MURA_Dataset(test_df, DATA_PATH+'PROCESSED/', output_size=512)

# %% figures
################################################################################
img_idx = 13

fig, axs = plt.subplots(2, 5, figsize=(12,5), gridspec_kw={'hspace':0.01, 'wspace':0.01})
if transparent: fig.set_alpha(0)
for ax in axs.reshape(-1):
    im, _, _, _, _ = dataset.__getitem__(img_idx)
    ax.imshow(im[0,:,:], cmap='Greys_r', vmin=0, vmax=1)
    ax.set_axis_off()
if save_fig : fig.savefig(FIGURE_PATH+'online_preprocessing_sample.png', dpi=FIG_RES, bbox_inches='tight')
plt.show()
