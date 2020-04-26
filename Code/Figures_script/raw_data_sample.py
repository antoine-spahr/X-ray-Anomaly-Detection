import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.io import imread
import matplotlib
import matplotlib.pyplot as plt

#import src.preprocessing.segmentation as seg

DATA_PATH = r'../../data/'
OUTPUT_PATH = r'../../Outputs/'
FIGURE_PATH = r'../../Figures/'
FIG_RES = 200 # dpi
transparent = True
save_fig = True

################################################################################
# %% Load info and some example images
################################################################################

df = pd.read_csv(DATA_PATH+'data_info.csv')
df = df.drop(df.columns[0], axis=1)
df2 = df.sample(n=28, random_state=4242) # 3 ; 5 ; 111 ; 1111 ; 2222 ; 4242 ; 33 ; 3333 ; 1234
fn = list(df2.filename)
fn_mask = list(df2.mask_filename)
bodypart = list(df2.body_part)

################################################################################
# %%
################################################################################

fig, axs = plt.subplots(4, 6, figsize=(16,12), gridspec_kw={'hspace':0.15, 'wspace':0.05})
if transparent: fig.set_alpha(0)
for f, fm, bp, ax in zip(fn, fn_mask, bodypart, axs.reshape(-1)):
    ax.set_axis_off()
    ax.set_title(bp.title(), fontsize=12)
    img = imread(DATA_PATH+'RAW/'+f)
    ax.imshow(img, cmap='gray')

#fig.tight_layout()
if save_fig: fig.savefig(FIGURE_PATH+'raw_sample.pdf', dpi=FIG_RES, bbox_inches='tight')
plt.show()
