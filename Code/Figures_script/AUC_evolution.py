import numpy as np
import pandas as pd
import glob

import matplotlib
import matplotlib.pyplot as plt

FIGURE_PATH = '../../Figures/'
OUTPUT_PATH = '../../Outputs/'
expe_folders = ['AE_DSAD_2020_06_05_01h15',
                'AE_DMSAD_2020_05_25_08h33',
                'SimCLR_DSAD_2020_06_01_10h52',
                'SimCLR_DMSAD_2020_05_19_09h11']

def extract_valid_auc(path_to_logs):
    """
    Parse the AUC evolution from the log files.
    """
    all_auc = []
    for log_file in glob.glob(path_to_logs + '/*.txt'):
        with open(log_file) as f:
            valid_auc = []
            for line in f:
                pos = line.find('Valid AUC')
                if pos != -1:
                    valid_auc.append(float(line[pos+10:pos+16]))
        all_auc.append(valid_auc)

    return all_auc

def get_valid_auc_df(all_auc):
    """
    Generate a DataFrame from the list of AUC list and compute the 95% CI.
    """
    df = pd.DataFrame(all_auc).transpose()
    df['m'] = df.mean(axis=1)
    df['s'] = df.std(axis=1)
    df['CI_inf'] = df.m - 1.96*df.s
    df['CI_sup'] = df.m + 1.96*df.s
    return df

# %%
df_CDSAD = get_valid_auc_df(extract_valid_auc(OUTPUT_PATH + 'SimCLR_DSAD_2020_06_01_10h52' + '/logs'))
df_DSAD = get_valid_auc_df(extract_valid_auc(OUTPUT_PATH + 'AE_DSAD_2020_06_05_01h15' + '/logs'))

df_CDMSAD = get_valid_auc_df(extract_valid_auc(OUTPUT_PATH + 'SimCLR_DMSAD_2020_05_19_09h11' + '/logs'))
df_DMSAD = get_valid_auc_df(extract_valid_auc(OUTPUT_PATH + 'AE_DMSAD_2020_05_25_08h33' + '/logs'))

# %% PLOT
fontsize=12
colors = ['xkcd:vermillion', 'xkcd:azure']

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.set_xticks(np.arange(0,160,10))
ax.set_yticks(np.arange(0,100,5))
ax.grid(True, axis='y')
for color, df, name in zip(colors, [df_CDSAD, df_DSAD], ['CDSAD', 'DSAD']):
    epochs = np.arange(1,len(df.m)+1,1)
    ax.plot(epochs, df.m, color=color, lw=2, label=name)
    ax.fill_between(epochs, df.CI_sup, df.CI_inf, fc=color, alpha=0.4, ec=None)

ax.set_xlim([1, max([len(df_CDSAD), len(df_DSAD)])])
ax.set_ylim([45,90])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('Validation AUC [%]', fontsize=fontsize)
ax.set_xlabel('Epochs [-]', fontsize=fontsize)
ax.legend(loc='upper right', ncol=1, fontsize=fontsize, frameon=True, facecolor='white', edgecolor='lightgray', framealpha=1)
fig.savefig(FIGURE_PATH + 'AUC_evolution.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
# fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharey=True)
# for ax, df_list in zip(axs.reshape(-1), [[df_CDSAD, df_DSAD], [df_CDMSAD, df_DMSAD]]):
#     for color, df in zip(colors, df_list):
#         epochs = np.arange(1,len(df.m)+1,1)
#         ax.plot(epochs, df.m, color=color, lw=2)
#         ax.fill_between(epochs, df.CI_sup, df.CI_inf, fc=color, alpha=0.5, ec=None)
#         # for col in df.columns[:-4]:
#         #     ax.plot(epochs, df[col], color=color, lw=2)
#     ax.set_xlim([1, max([len(df_list[0]), len(df_list[1])])])
#     ax.set_ylim(top=90)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.set_ylabel('Validation AUC [%]', fontsize=fontsize)
#     ax.set_xlabel('Epochs [-]', fontsize=fontsize)
#
# plt.show()
