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
FIG_RES = 300 # dpi
transparent = True
save_fig = True

# %% Load info and some example images
################################################################################
df = pd.read_csv(DATA_PATH+'data_info.csv')
df = df.drop(df.columns[0], axis=1)
#df2 = df.sample(n=24, random_state=2222) # 3 ; 5 ; 111 ; 1111 ; 2222 ; 4242 ; 33 ; 3333 ; 1234
df2 = df.iloc[[1, 11670, 737, 38420, 30090,35600,16900]]#286 38414
fn = list(df2.filename)
fn_mask = list(df2.mask_filename)
bodypart = list(df2.body_part)

################################################################################
m_color, m_alpha = 'limegreen', 0.5


fig, axs = plt.subplots(1, 7, figsize=(8,2), gridspec_kw={'hspace':0.18, 'wspace':0.01, 'width_ratios':[0.18, 0.12, 0.15, 0.12, 0.13, 0.18, 0.12]})
#if transparent: fig.set_alpha(0)
for f, fm, bp, ax in zip(fn, fn_mask, bodypart, axs.reshape(-1)):
    ax.set_axis_off()
    ax.set_title(bp.title(), fontsize=12)
    img, mask = imread(DATA_PATH+'PROCESSED/'+f), imread(DATA_PATH+'PROCESSED/'+fm)
    m = np.ma.masked_where(mask == 0, mask)
    ax.imshow(img, cmap='Greys_r')#, vmin=0, vmax=1)
    ax.imshow(m, cmap = matplotlib.colors.ListedColormap(['white', m_color]), \
              vmin=0, vmax=1, alpha=m_alpha, zorder=1)

# legend
handles, labels = [matplotlib.patches.Patch(fc=m_color, alpha=m_alpha)], ['segmentation mask']
lgd = fig.legend(handles, labels, ncol=1, fontsize=12, loc='lower center',
                 bbox_to_anchor=[0.5, -0.1], bbox_transform=fig.transFigure, frameon=False)
#fig.tight_layout()
if save_fig: fig.savefig(FIGURE_PATH+'segmentation_sample.pdf', dpi=FIG_RES, bbox_inches='tight', bbox_extra_artist=(lgd,))
plt.show()
