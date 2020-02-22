import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.io import imread
import matplotlib
import matplotlib.pyplot as plt

#import src.preprocessing.segmentation as seg

DATA_PATH = r'../data/'
OUTPUT_PATH = r'../Outputs/'
FIGURE_PATH = r'../Figures/'
FIG_RES = 200 # dpi
transparent = True
save_fig = True

# %% Load info and some example images
################################################################################
df = pd.read_csv(DATA_PATH+'data_info.csv')
df = df.drop(df.columns[0], axis=1)
df2 = df.sample(n=16, random_state=1111) # 3 ; 5 ; 111 ; 1111 ; 2222
fn = list(df2.filename)
fn_mask = list(df2.mask_filename)
bodypart = list(df2.body_part)
# %%
################################################################################
m_color, m_alpha = 'limegreen', 0.25

fig, axs = plt.subplots(4, 4, figsize=(18,18))
if transparent: fig.set_alpha(0)
for f, fm, bp, ax in zip(fn, fn_mask, bodypart, axs.reshape(-1)):
    ax.set_axis_off()
    ax.set_title(bp.title())
    img, mask = imread(DATA_PATH+'PROCESSED/'+f), imread(DATA_PATH+'PROCESSED/'+fm)
    m = np.ma.masked_where(mask == 0, mask)
    ax.imshow(img, cmap='Greys_r')#, vmin=0, vmax=1)
    ax.imshow(m, cmap = matplotlib.colors.ListedColormap(['white', m_color]), \
              vmin=0, vmax=1, alpha=m_alpha, zorder=1)

# legend
handles, labels = [matplotlib.patches.Patch(fc=m_color, alpha=m_alpha)], ['segmentation mask']
lgd = fig.legend(handles, labels, ncol=1, fontsize=12, loc='lower center',
                 bbox_to_anchor=[0.5, -0.03], bbox_transform=fig.transFigure)
fig.tight_layout()
if save_fig: fig.savefig(FIGURE_PATH+'segmentation_sample.png', dpi=FIG_RES, bbox_inches='tight', bbox_extra_artist=(lgd,))
plt.show()
