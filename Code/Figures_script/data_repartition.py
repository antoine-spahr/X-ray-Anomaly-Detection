import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

DATA_PATH = r'../../data/'
OUTPUT_PATH = r'../../Outputs/'
FIGURE_PATH = r'../../Figures/'
FIG_RES = 200 # dpi
transparent = False

def human_format(num, pos=None):
    """
    Format large number using a human interpretable unit (kilo, mega, ...).
    ----------
    INPUT
        |---- num (int) -> the number to reformat
    OUTPUT
        |---- num (str) -> the reformated number
    """
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0

    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

# %% Load info
################################################################################
df = pd.read_csv(DATA_PATH+'data_info.csv')
df = df.drop(df.columns[0], axis=1)

# %% Get Patient's bodypart numbers
################################################################################
df_grp = df.groupby(['body_part', 'patientID']).mean()
df_patient = df_grp.reset_index().groupby(['body_part', 'body_part_abnormal']).count()
df_patient = df_patient.reset_index(level=1)

body_parts = [name.title() for name in df_patient.index.unique()]
normal_count = np.array(df_patient.loc[df_patient.body_part_abnormal == 0.0, 'patientID'])
mixt_count = np.array(df_patient.loc[df_patient.body_part_abnormal == 0.5, 'patientID'])
abnormal_count = np.array(df_patient.loc[df_patient.body_part_abnormal == 1.0, 'patientID'])

# %% Get Xrays numbers
################################################################################
df_xr = df.groupby(['body_part', 'abnormal_XR']).count() \
          .reset_index(level=1)
normal_xr_count = np.array(df_xr.loc[df_xr.abnormal_XR == 0, 'patientID'])
abnormal_xr_count = np.array(df_xr.loc[df_xr.abnormal_XR == 1, 'patientID'])

# %% get overall numbers
################################################################################
df_patient = df.groupby(['body_part', 'patientID']).mean()
patient_normal = df_patient.body_part_abnormal.value_counts()[0.0]
patient_mixt = df_patient.body_part_abnormal.value_counts()[0.5]
patient_abnormal = df_patient.body_part_abnormal.value_counts()[1.0]
xr_normal = df.abnormal_XR.value_counts()[0]
xr_abnormal = df.abnormal_XR.value_counts()[1]

# %%
################################################################################
width = 0.7
height = 0.5
colors1 = ['gainsboro', 'silver', 'gray']
colors2 = ['gainsboro', 'gray']

#fig, axs = plt.subplots(2,2,figsize=(14,7), gridspec_kw={'height_ratios':[0.85, 0.15]})
fig = plt.figure(figsize=(16,8))
if transparent: fig.patch.set_alpha(0) # transparent background
gs = fig.add_gridspec(2, 2, wspace=0.3, hspace=0.1, height_ratios=[0.85, 0.15])
#===============================================================================
# stacked barplot per patients
axp = fig.add_subplot(gs[0,0])
if transparent: axp.patch.set_alpha(0)
axp.bar(body_parts, normal_count, width, bottom=0, color=colors1[0],
           lw=1, ec='black', label='Normal studies only')
axp.bar(body_parts, mixt_count, width, bottom=normal_count, color=colors1[1],
           lw=1, ec='black', label='Mixt studies')
axp.bar(body_parts, abnormal_count, width, bottom=(mixt_count+normal_count),
           color=colors1[2], lw=1, ec='black', label='Abnormal studies only')
# axis name and title
axp.set_title('Number of patients per body part categorized \n by the presence of anomaly per body part')
axp.set_ylabel('Number of patients [-]', fontsize=12)
# legend
handles, labels = axp.get_legend_handles_labels()
axp.legend(handles[::-1], labels[::-1], loc='upper left', ncol=1, fontsize=12, frameon=False, )
# goodies
axp.spines['right'].set_visible(False)
axp.spines['top'].set_visible(False)
axp.tick_params(axis='both', labelsize=12)
axp.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(human_format))
axp.text(0, 1.02, 'A', fontsize=20, fontweight='bold', transform=axp.transAxes)
#===============================================================================
# Overall plot
axp_all = fig.add_subplot(gs[1,0])
if transparent: axp_all.patch.set_alpha(0)
axp_all.barh(['overall'], patient_normal, height=height, left=0, color=colors1[0],
           lw=1, ec='black', label='Normal studies only')
axp_all.barh(['overall'], patient_mixt, height=height, left=patient_normal, color=colors1[1],
           lw=1, ec='black', label='Mixt studies')
axp_all.barh(['overall'], patient_abnormal, height=height, left=patient_normal+patient_mixt, color=colors1[2],
           lw=1, ec='black', label='Abnormal studies only')
# goodies
axp_all.set_xlabel('Overall number of patients [-]', fontsize=12)
axp_all.spines['right'].set_visible(False)
axp_all.spines['top'].set_visible(False)
axp_all.tick_params(axis='both', labelsize=12)
axp_all.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(human_format))
axp_all.set_ylim([-0.5,0.5])
#===============================================================================
# stacked bar plot per image
axxr = fig.add_subplot(gs[0,1])
if transparent: axxr.patch.set_alpha(0)
axxr.bar(body_parts, normal_xr_count, width, bottom=0, color=colors2[0],
           lw=1, ec='black', label='Normal Xrays')
axxr.bar(body_parts, abnormal_xr_count, width, bottom=normal_xr_count, color=colors2[1],
           lw=1, ec='black', label='Abnormal Xrays')
# axis name and title
axxr.set_title('Number of Xrays images per body part \ncategorized by the presence of anomaly')
axxr.set_ylabel('Number of Xrays images [-]', fontsize=12)
# legend
handles, labels = axxr.get_legend_handles_labels()
axxr.legend(handles[::-1], labels[::-1], loc='upper left', ncol=1, fontsize=12, frameon=False, )
# goodies
axxr.spines['right'].set_visible(False)
axxr.spines['top'].set_visible(False)
axxr.tick_params(axis='both', labelsize=12)
axxr.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(human_format))
axxr.text(0, 1.02, 'B', fontsize=20, fontweight='bold', transform=axxr.transAxes)
#===============================================================================
# Overall plot
axxr_all = fig.add_subplot(gs[1,1])
if transparent: axxr_all.patch.set_alpha(0)
axxr_all.barh(['overall'], xr_normal, height=height, left=0, color=colors2[0],
           lw=1, ec='black', label='Normal studies only')
axxr_all.barh(['overall'], xr_abnormal, height=height, left=xr_normal, color=colors2[1],
           lw=1, ec='black', label='Abnormal studies only')
# goodies
axxr_all.set_xlabel('Overall number of Xrays images [-]', fontsize=12)
axxr_all.spines['right'].set_visible(False)
axxr_all.spines['top'].set_visible(False)
axxr_all.tick_params(axis='both', labelsize=12)
axxr_all.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(human_format))
axxr_all.set_ylim([-0.5,0.5])
#===============================================================================
fig.savefig(FIGURE_PATH+'data_repartition.pdf', dpi=FIG_RES, bbox_inches='tight')
plt.show()
