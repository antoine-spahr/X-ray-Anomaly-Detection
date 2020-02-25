import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.io import imread
import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

import src.preprocessing.cropping_rect as croprect

DATA_PATH = r'../../data/'
OUTPUT_PATH = r'../../Outputs/'
FIGURE_PATH = r'../../Figures/'
FIG_RES = 200 # dpi
transparent = True
save_fig = True

# %% Load info and some example images
################################################################################
df = pd.read_csv(DATA_PATH+'data_info.csv')
df = df.drop(df.columns[0], axis=1)

df_cropped = df[df.uncropped == 0].reset_index()
df_uncropped = df[df.uncropped == 1].reset_index()

cropped_idx = [11000, 19000, 7000, 2000, 3000, 4000]
uncropped_idx = [9000, 8000, 6000, 4500, 6500, 7500]
#fn = df_cropped.loc[5000,'filename']
#fn = df_uncropped.loc[9500,'filename']
fn_crop = [df_cropped.loc[i,'filename'] for i in cropped_idx]
fn_uncrop = [df_uncropped.loc[i,'filename'] for i in uncropped_idx]

# %%
img = imread(DATA_PATH+'RAW/'+fn_crop[0])

rects = croprect.find_squares(img, min_area=40000)
img_cropped = croprect.crop_squares(rects, img)

fig, axs = plt.subplots(1,2,figsize=(10,6))
if transparent: fig.set_alpha(0)

axs[0].imshow(img, cmap='Greys_r')
for r in rects[1:]:
    axs[0].add_patch(matplotlib.patches.Polygon(r, lw=1, ec='red', fc=(0,0,0,0)))

axs[0].add_patch(matplotlib.patches.Polygon(rects[0], lw=2, ec='dodgerblue', fc=(0,0,0,0)))
axs[0].set_title('Original image with detected rectangles')
axs[0].set_axis_off()
handles = [matplotlib.patches.Patch(lw=1, ec='red', fc=(0,0,0,0)),
           matplotlib.patches.Patch(lw=2, ec='dodgerblue', fc=(0,0,0,0))]
labels = ['Detected rectangles', 'Selected rectangle']
lgd = axs[0].legend(handles, labels, loc='lower left', ncol=1, fontsize=12,
                frameon=False, bbox_to_anchor=(0, 0))
for t in lgd.get_texts(): t.set_color('lightgray')

axs[1].imshow(img_cropped, cmap='Greys_r')
axs[1].set_title('Extracted images')
axs[1].set_axis_off()
fig.tight_layout()

if save_fig: fig.savefig(FIGURE_PATH+'rect_cropping_sample.png', dpi=FIG_RES, bbox_inches='tight', bbox_extra_artist=(lgd,))
plt.show()
