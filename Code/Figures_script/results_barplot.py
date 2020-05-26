import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.io import imread
import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

from src.utils.results_processing import metric_barplot

OUTPUT_PATH = r'../../Outputs/'
FIGURE_PATH = r'../../Figures/'
FIG_RES = 200 # dpi
transparent = True
save_fig = True

# {experiment name : [experiment folder , score name]}
expe_semisupervised = {'DSAD': ['DeepSAD_2020_02_25_11h12', 'embedding'],
                       'Joint\nDSAD': ['JointDeepSAD_2020_03_20_19h20_milestone4080', 'embedding'],
                       'Joint\nDSAD\nSubspace': ['Joint_DeepSAD_Subspace_2020_03_26_11h44', 'embedding'],
                       'DROCC-LF': ['DROCC-LF_2020_04_04_08h49', 'DROCC-LF anomaly score']}

expe_unsupervised = {'DSVDD': ['DeepSVDD_2020_03_02_16h35', 'embedding'],
                     'Joint\nDSVDD' : ['JointDeepSVDD_2020_03_23_09h14_milestone4080', 'embedding'],
                     'Autoencoder \n(JointDSVDD)': ['JointDeepSVDD_2020_03_23_09h14_milestone4080', 'reconstruction'],
                     'Joint\nDSVDD\nSubspace': ['Joint_DeepSVDD_Subspace_2020_03_28_12h40', 'embedding'],
                     'DROCC': ['DROCC_2020_04_02_11h56', 'DROCC anomaly score'],
                     'ARAE': ['ARAE_2020_04_09_20h13', 'ARAE anomaly score']}

#%%#############################################################################
#                                 Load Data                                    #
################################################################################

data_auc = []
data_auprc = []
names = []

for name, folder in expe_semisupervised.items():
    names.append(name)
    data_auc.append(pd.read_csv(OUTPUT_PATH + folder[0] + '/analysis/AUC_Tables/Test_AURC.csv',
                           index_col=0).loc[folder[1]][:-2])
    data_auprc.append(pd.read_csv(OUTPUT_PATH + folder[0] + '/analysis/AUC_Tables/Test_AUPRC.csv',
                             index_col=0).loc[folder[1]][:-2])

for name, folder in expe_unsupervised.items():
    names.append(name)
    data_auc.append(pd.read_csv(OUTPUT_PATH + folder[0] + '/analysis/AUC_Tables/Test_AURC.csv',
                           index_col=0).loc[folder[1]][:-2])
    data_auprc.append(pd.read_csv(OUTPUT_PATH + folder[0] + '/analysis/AUC_Tables/Test_AUPRC.csv',
                             index_col=0).loc[folder[1]][:-2])

data_auc = pd.concat(data_auc, axis=1).values
data_auprc = pd.concat(data_auprc, axis=1).values

#%%#############################################################################
#                                   Figures                                    #
################################################################################

fig, ax = plt.subplots(1, 1, figsize=(16,8))
if transparent: fig.set_alpha(0)

metric_barplot([data_auc, data_auprc], ['AUC', 'AUPRC'], names, fontsize=14,
               colors=['salmon', 'skyblue'], w=0.45, ax=ax, gap=len(expe_semisupervised),
               legend_kwargs=dict(loc='upper center', ncol=2, frameon=True, framealpha=1.0,
               fontsize=13, bbox_to_anchor=(0.5, -0.15), bbox_transform=ax.transAxes))
#ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
ax.set_ylabel('AUC [-] , AUPRC [-]')

ax.annotate('Semi-Supervised', xy=(0.215, 1.02), xytext=(0.215, 1.1), xycoords='axes fraction',
            fontsize=13, ha='center', va='center', rotation=0, bbox=dict(boxstyle='square', fc='white', lw=0),
            arrowprops=dict(arrowstyle='-[, widthB=15, lengthB=0.5', lw=1))

ax.annotate('Unsupervised', xy=(0.69, 1.02), xytext=(0.69, 1.1), xycoords='axes fraction',
            fontsize=13, ha='center', va='center', rotation=0, bbox=dict(boxstyle='square', fc='white', lw=0),
            arrowprops=dict(arrowstyle='-[, widthB=22, lengthB=0.5', lw=1))

fig.tight_layout()
if save_fig: fig.savefig(FIGURE_PATH+'results_barplot.pdf', dpi=FIG_RES, bbox_inches='tight')
plt.show()
