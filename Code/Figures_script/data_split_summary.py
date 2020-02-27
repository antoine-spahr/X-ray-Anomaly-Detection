import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

import src.datasets.MURADataset as MURA

DATA_PATH = r'../../data/'
OUTPUT_PATH = r'../../Outputs/'
FIGURE_PATH = r'../../Figures/'
FIG_RES = 200 # dpi
transparent = False

def get_patient_bodypart_counts(df, by_bodyparts=False):
    """

    """
    #grp_val = ['body_part'] if by_bodyparts else []
    df_grp = df.groupby(['body_part','patientID']).mean()
    if by_bodyparts:
        df_patient = df_grp.reset_index().groupby(['body_part','body_part_abnormal']).count()
    else:
        df_patient = df_grp.reset_index().groupby(['body_part_abnormal']).count()
    df_patient = df_patient.reset_index(level=1) if by_bodyparts else df_patient.reset_index()

    counts = {'normal':None, 'mixt':None, 'abnormal':None}
    if by_bodyparts :
        counts['body_parts'] = [name.title() for name in df_patient.index.unique()]
    #body_parts = [name.title() for name in df_patient.index.unique()]
    counts['normal'] = np.array(df_patient.loc[df_patient.body_part_abnormal == 0.0, 'patientID'])
    counts['mixt'] = np.array(df_patient.loc[df_patient.body_part_abnormal == 0.5, 'patientID'])
    counts['abnormal'] = np.array(df_patient.loc[df_patient.body_part_abnormal == 1.0, 'patientID'])
    counts = {k:(v if v.size > 0 else np.array([0])) for k, v in counts.items()}
    return counts

def get_xr_counts(df, names=True):
    """

    """
    df_xr = df.groupby(['body_part', 'abnormal_XR']).count().reset_index(level=1)
    counts = {'normal':None, 'abnormal':None}
    body_parts_name = [name.title() for name in df_xr.index.unique()]
    counts['normal'] = np.array(df_xr.loc[df_xr.abnormal_XR == 0, 'patientID'])
    counts['abnormal'] = np.array(df_xr.loc[df_xr.abnormal_XR == 1, 'patientID'])
    counts = {k:(v if v.size > 0 else np.array([0])) for k, v in counts.items()}
    if names:
        return counts, body_parts_name
    else:
        return counts

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

def draw_cruved_rect(x1, x2, h1, h2, offset1, offset2, ax, fc='lightgray', ec='gray', lw=1, alpha=0.3):
    """

    """
    if h1 != 0 or h2 != 0:
        x05 = (x2+x1)/2
        v = np.array([[x1, offset1],
                      [x05, offset1],
                      [x05, offset2],
                      [x2, offset2],
                      [x2, offset2 + h2],
                      [x05, offset2 + h2],
                      [x05, offset1 + h1],
                      [x1, offset1 + h1]])

        p = matplotlib.path.Path(v, codes = [1,4,4,4,2,4,4,4], closed=True)
        ax.add_patch(matplotlib.patches.PathPatch(p, lw=lw, ec=ec, fc=fc, alpha=alpha, zorder=-1))

def plot_set_distribution(df, ax, colors, title=False, ticklabel=True, return_counts=True):
    width = 0.5

    counts, body_parts_name = get_xr_counts(df, names=True)

    ax.bar(range(1,len(body_parts_name)+1), counts['normal'], width, bottom=0, color=colors[0],
               lw=1, ec='black', label='Normal X-rays image')
    ax.bar(range(1,len(body_parts_name)+1), counts['abnormal'], width, bottom=counts['normal'],
               color=colors[2], lw=1, ec='black', label='Abnormal X-rays image')

    # if title: ax.set_title('Number of X-rays images per set per body parts')
    ax.set_ylabel('X-rays', fontsize='12')
    # goodies
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if ticklabel:
        ax.xaxis.set_ticklabels(['']+body_parts_name, rotation=25, ha='right')
    else:
        ax.xaxis.set_ticklabels([])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(human_format))
    ax.tick_params(axis='both', labelsize=10)
    if return_counts:
        return counts

# %% Get data info
df = pd.read_csv(DATA_PATH+'data_info.csv')
df = df.drop(df.columns[0], axis=1)
df = df[df.low_contrast == 0]

# %% split data
spliter = MURA.MURA_TrainValidTestSplitter(df, train_frac=0.5, ratio_known_normal=0.05, ratio_known_abnormal=0.05, random_state=42)
spliter.split_data(verbose=True)

train_df = spliter.get_subset('train')
valid_df = spliter.get_subset('valid')
test_df = spliter.get_subset('test')

# %% get counts
all_counts = get_patient_bodypart_counts(df, False)
train_counts = get_patient_bodypart_counts(train_df, False)
valid_counts = get_patient_bodypart_counts(valid_df, False)
test_counts = get_patient_bodypart_counts(test_df, False)

# %%
################################################################################
width = 0.5
colors1 = ['paleturquoise', 'gold', 'lightsalmon']
offset = 0
offset_set = 5000
pos1 = 0.35
pos2 = 2.75
pos3 = 4
set_len_x = 3

################################################################################
fig = plt.figure(figsize=(14,7))
if transparent: fig.patch.set_alpha(0)
ax = fig.add_subplot(1, 1, 1)

# add stacked bar plots
ax.bar([pos1], all_counts['normal'], width, bottom=offset, color=colors1[0],
           lw=1, ec='black', label='Normal studies only')
ax.bar([pos1], all_counts['mixt'], width, bottom=offset+all_counts['normal'], color=colors1[1],
           lw=1, ec='black', label='Mixt studies')
ax.bar([pos1], all_counts['abnormal'], width, bottom=(offset+all_counts['mixt']+all_counts['normal']),
           color=colors1[2], lw=1, ec='black', label='Abnormal studies only')

offset_tr = -offset_set
ax.bar([pos2], train_counts['normal'], width, bottom=offset_tr, color=colors1[0],
           lw=1, ec='black', label='Normal studies only')
ax.bar([pos2], train_counts['mixt'], width, bottom=offset_tr+train_counts['normal'], color=colors1[1],
           lw=1, ec='black', label='Mixt studies')
ax.bar([pos2], train_counts['abnormal'], width, bottom=(offset_tr+train_counts['mixt']+train_counts['normal']),
           color=colors1[2], lw=1, ec='black', label='Abnormal studies only')

offset_v = offset_tr + sum(train_counts.values()) + offset_set
ax.bar([pos2], valid_counts['normal'], width, bottom=offset_v, color=colors1[0],
           lw=1, ec='black', label='Normal studies only')
ax.bar([pos2], valid_counts['mixt'], width, bottom=offset_v+valid_counts['normal'], color=colors1[1],
           lw=1, ec='black', label='Mixt studies')
ax.bar([pos2], valid_counts['abnormal'], width, bottom=(offset_v+valid_counts['mixt']+valid_counts['normal']),
           color=colors1[2], lw=1, ec='black', label='Abnormal studies only')

offset_te = offset_v + sum(valid_counts.values()) + offset_set
ax.bar([pos2], test_counts['normal'], width, bottom=offset_te, color=colors1[0],
           lw=1, ec='black', label='Normal studies only')
ax.bar([pos2], test_counts['mixt'], width, bottom=offset_te+test_counts['normal'], color=colors1[1],
           lw=1, ec='black', label='Mixt studies')
ax.bar([pos2], test_counts['abnormal'], width, bottom=(offset_te+test_counts['mixt']+test_counts['normal']),
           color=colors1[2], lw=1, ec='black', label='Abnormal studies only')

# Connection with Bezier curves
n_color = [colors1[0], 'gray']
m_color = [colors1[1], 'gray']
a_color = [colors1[2], 'gray']
lw = 1

x1, x2 = pos1 + width/2, pos2 - width/2
hn1 = float(train_counts['normal'])
hn2 = float(valid_counts['normal'])
hn3 = float(test_counts['normal'])
draw_cruved_rect(x1, x2, hn1, hn1, 0, offset_tr, ax=ax, fc=n_color[0], ec=n_color[1], lw=lw, alpha=0.3)
draw_cruved_rect(x1, x2, hn2, hn2, hn1, offset_v, ax=ax, fc=n_color[0], ec=n_color[1], lw=lw, alpha=0.3)
draw_cruved_rect(x1, x2, hn3, hn3, hn1+hn2, offset_te, ax=ax, fc=n_color[0], ec=n_color[1], lw=lw, alpha=0.3)

hnall = float(all_counts['normal'])
hm2 = float(valid_counts['mixt'])
hm3 = float(test_counts['mixt'])
draw_cruved_rect(x1, x2, hm2, hm2, hnall, offset_v + hn2, ax=ax, fc=m_color[0], ec=m_color[1], lw=lw, alpha=0.3)
draw_cruved_rect(x1, x2, hm3, hm3, hnall+hm2, offset_te + hn3, ax=ax, fc=m_color[0], ec=m_color[1], lw=lw, alpha=0.3)

hmall = float(all_counts['mixt'])
ha1 = float(train_counts['abnormal'])
ha2 = float(valid_counts['abnormal'])
ha3 = float(test_counts['abnormal'])
draw_cruved_rect(x1, x2, ha1, ha1, hnall + hmall, offset_tr + hn1, ax=ax, fc=a_color[0], ec=a_color[1], lw=lw, alpha=0.3)
draw_cruved_rect(x1, x2, ha2, ha2, hnall + hmall + ha1, offset_v + hn2 + hm2, ax=ax, fc=a_color[0], ec=a_color[1], lw=lw, alpha=0.3)
draw_cruved_rect(x1, x2, ha3, ha3, hnall + hmall + ha1 + ha2, offset_te + hn3 + hm3, ax=ax, fc=a_color[0], ec=a_color[1], lw=lw, alpha=0.3)

# Labels of sets
txt_offset = 400
ax.text(pos1, offset + sum(all_counts.values())[0]+txt_offset, 'All\npatients', fontsize=12, ha='center')
ax.text(pos2, offset_tr + sum(train_counts.values())[0]+txt_offset, 'Train Set', fontsize=12, ha='center')
ax.text(pos2, offset_v + sum(valid_counts.values())[0]+txt_offset, 'Valid Set', fontsize=12, ha='center')
ax.text(pos2, offset_te + sum(test_counts.values())[0]+txt_offset, 'Test Set', fontsize=12, ha='center')

# add Xrays image distribution
htr, hv, hte = 10000, 5000, 5000
ax_xr_tr = ax.inset_axes([pos3, offset_tr, set_len_x, htr], transform=ax.transData)
tr_bp_counts = plot_set_distribution(train_df, ax_xr_tr, colors1, title=False, ticklabel=True)
ax_xr_v = ax.inset_axes([pos3, offset_v[0], set_len_x, hv], transform=ax.transData)
valid_bp_counts = plot_set_distribution(valid_df, ax_xr_v, colors1, title=False, ticklabel=False)
ax_xr_te = ax.inset_axes([pos3, offset_te[0], set_len_x, hte], transform=ax.transData)
test_bp_counts = plot_set_distribution(test_df, ax_xr_te, colors1, title=True, ticklabel=False)

# add connections
draw_cruved_rect(pos2+width/2, pos3, sum(train_counts.values())[0], htr, offset_tr, offset_tr, ax=ax, fc='lightgray', ec='gray', lw=lw, alpha=0.3)
draw_cruved_rect(pos2+width/2, pos3, sum(valid_counts.values())[0], hv, offset_v, offset_v, ax=ax, fc='lightgray', ec='gray', lw=lw, alpha=0.3)
draw_cruved_rect(pos2+width/2, pos3, sum(test_counts.values())[0], hte, offset_te, offset_te, ax=ax, fc='lightgray', ec='gray', lw=lw, alpha=0.3)

# add text of total amounts
ax_xr_tr.text(0.05, 0.9,
              f"Normal : {tr_bp_counts['normal'].sum()} / Abnormal : {tr_bp_counts['abnormal'].sum()}",
              fontsize=10, transform=ax_xr_tr.transAxes)
ax_xr_v.text(0.05, 0.8,
              f"Normal : {valid_bp_counts['normal'].sum()} / Abnormal : {valid_bp_counts['abnormal'].sum()}",
              fontsize=10, transform=ax_xr_v.transAxes)
ax_xr_te.text(0.05, 0.8,
              f"Normal : {test_bp_counts['normal'].sum()} / Abnormal : {test_bp_counts['abnormal'].sum()}",
              fontsize=10, transform=ax_xr_te.transAxes)

# add axis facecolor
ax_xr_tr.patch.set_facecolor('lightgray')
ax_xr_tr.patch.set_alpha(0.3)
ax_xr_v.patch.set_facecolor('lightgray')
ax_xr_v.patch.set_alpha(0.3)
ax_xr_te.patch.set_facecolor('lightgray')
ax_xr_te.patch.set_alpha(0.3)

# Goodies
ax.set_ylim([-offset_set-500, offset_te + hte + 500])#bottom=-offset_set-500)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.yaxis.set_major_locator(plt.NullLocator())
ax.xaxis.set_major_locator(plt.NullLocator())
ax_bis = ax.inset_axes([0,0,3,15000], transform=ax.transData, zorder=-5)
ax_bis.spines['right'].set_visible(False)
ax_bis.spines['top'].set_visible(False)
ax_bis.spines['bottom'].set_visible(False)
ax_bis.set_ylim([0,15000])
ax_bis.xaxis.set_major_locator(plt.NullLocator())
ax_bis.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(human_format))
ax_bis.tick_params(axis='both', labelsize=10)
ax_bis.set_ylabel('Number of patients [-]', fontsize=12)
ax.set_xlim([0,pos3 + set_len_x])

# legend
handles = [matplotlib.patches.Patch(fc=c, ec='black', lw=1) for c in colors1]
labels = ['Normal X-rays only', 'Mixt X-rays', 'Abnormal X-rays only']
lgd = ax.legend(handles[::-1], labels[::-1], loc='lower left', ncol=1, fontsize=12,
                frameon=False, bbox_to_anchor=(0, -0.1))#, bbox_transform=fig.transFigure)

if transparent:
    ax.patch.set_alpha(0)
    ax_bis.patch.set_alpha(0)
fig.savefig(FIGURE_PATH+'unsupervized_data_split_summary.pdf', dpi=FIG_RES, bbox_inches='tight', bbox_extra_artist=(lgd,))
plt.show()
