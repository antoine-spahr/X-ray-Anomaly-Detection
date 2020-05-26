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
FIG_RES = 300 # dpi
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
img_good = imread(DATA_PATH+'RAW/'+fn_crop[0]) # 0
img_fail = imread(DATA_PATH+'RAW/'+fn_crop[1])

rects_good = croprect.find_squares(img_good, min_area=40000)
img_cropped_good = croprect.crop_squares(rects_good, img_good)

rects_fail = croprect.find_squares(img_fail, min_area=40000)
img_cropped_fail = croprect.crop_squares(rects_fail, img_fail)

fig, axs = plt.subplots(1,4,figsize=(8,4), gridspec_kw={'wspace': 0.25, 'width_ratios': [0.25,0.2, 0.4, 0.12]})
#fig, axs = plt.subplots(1,2,figsize=(3.5,3.5), gridspec_kw={'wspace': 0.65, 'width_ratios': [0.56,0.44]})
#if transparent: fig.set_alpha(0)

for i, rects, img, img_cropped in zip([0, 2], [rects_good, rects_fail], [img_good, img_fail], [img_cropped_good, img_cropped_fail]):
#for i, rects, img, img_cropped in zip([0, 2], [rects_good], [img_good], [img_cropped_good]):
    axs[i].imshow(img, cmap='Greys_r')
    for r in rects[1:]:
        axs[i].add_patch(matplotlib.patches.Polygon(r, lw=2, ec='red', fc=(0,0,0,0)))

    axs[i].add_patch(matplotlib.patches.Polygon(rects[0], lw=2, ec='dodgerblue', fc=(0,0,0,0)))
    axs[i].set_title('Original', fontsize=12, fontweight='bold')
    axs[i].set_axis_off()

    axs[i+1].imshow(img_cropped, cmap='Greys_r')
    axs[i+1].set_title('Extracted', fontsize=12, fontweight='bold')
    axs[i+1].set_axis_off()

    arrow = matplotlib.patches.ConnectionPatch(xyA=(1, 0.5), xyB=(0, 0.5),
                         coordsA='axes fraction', coordsB='axes fraction',
                         axesA=axs[i], axesB=axs[i+1],
                         arrowstyle='simple, head_length=1.0, head_width=1.5, tail_width=0.7', connectionstyle="arc3")
    arrow.set_facecolor('k')
    axs[i].add_artist(arrow)

handles = [matplotlib.patches.Patch(lw=2, ec='red', fc=(0,0,0,0)),
           matplotlib.patches.Patch(lw=2, ec='dodgerblue', fc=(0,0,0,0))]
labels = ['Detected rectangles', 'Selected rectangle']
lgd = fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=12,
                    frameon=False, bbox_to_anchor=(0.5, 0.15), bbox_transform=fig.transFigure)

#fig.tight_layout()

if save_fig: fig.savefig(FIGURE_PATH+'rect_cropping_sample_ieee.pdf', dpi=FIG_RES, bbox_inches='tight', bbox_extra_artist=(lgd,))
plt.show()
