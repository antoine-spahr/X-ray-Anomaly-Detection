import pandas as pd
import os
import sys
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

DATA_PATH = r'../../data/'
OUTPUT_PATH = r'../../Outputs/'
FIGURE_PATH = r'../../Figures/'

# Get Data
df = pd.read_csv(DATA_PATH+'data_info.csv')
df = df.drop(df.columns[0], axis=1)
df = df[df.low_contrast == 0]
df = df[df.abnormal_XR == 1]


# Plot
fig, axs = plt.subplots(1,5, figsize=(8,3), gridspec_kw=dict(width_ratios=[0.14,0.17,0.32,0.175,0.195]))
for i, ax in zip([15004, 4940, 5358, 9999, 9275], axs.reshape(-1)):
    img = skimage.io.imread(DATA_PATH + 'PROCESSED/' + df.iloc[i]['filename'])
    ax.imshow(img, cmap='gray')
    ax.set_axis_off()
fig.tight_layout()
fig.savefig(FIGURE_PATH + 'abnormal_sample.pdf', dpi=300, bbox_inches='tight')
plt.show()
