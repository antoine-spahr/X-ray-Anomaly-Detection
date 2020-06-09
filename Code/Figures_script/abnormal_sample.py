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
fig, axs = plt.subplots(1,4, figsize=(8,4), gridspec_kw=dict(width_ratios=[0.2,0.1,0.45,0.25]))
for i, ax in zip([15004, 4899, 5358, 9999], axs.reshape(-1)):
    img = skimage.io.imread(DATA_PATH + 'PROCESSED/' + df.iloc[i]['filename'])
    ax.imshow(img, cmap='gray')
    ax.set_axis_off()
fig.tight_layout()
fig.savefig(FIGURE_PATH + 'abnormal_sample.pdf', dpi=300, bbox_inches='tight')
plt.show()
