import numpy as np
import pandas as pd
import glob
import json
import os
import sys
sys.path.append('../')

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

def load_experiment_results(path, exp_folders, exp_names):
    """
    Load the results of the specified experiments in a dictionnary.
    ----------
    INPUT
        |---- path (str) where the expeiment folder are stored.
        |---- exp_folders (list of str) the folders' name to load.
        |---- exp_names (list of str) the dict keys to use for each experiments.
    OUTPUT
        |---- results_all (dict) the dict containing all the experiement results
        |           (each key is an experiment with repetitions). The value of
        |           the dict is a list of results dict (results of one replicate).
    """
    results_all = {}
    for folder, name in zip(exp_folders, exp_names):
        results = []
        for json_file in glob.glob(path + folder + '/results/*.json'):
            with open(json_file) as fp:
                results.append(json.load(fp))
        results_all[name] = results

    return results_all

def scores_as_df(results_json, set, em_col=['index', 'label', 'scores']):
    """
    Make a pandas Dataframe from the scores in the JSON results object for the
    given set.
    ----------
    INPUT
        |---- results_json (dict) the model result of one training.
        |---- set (str) the set to convert : 'valid' or 'test'
    OUTPUT
        |---- df (pd.DataFrame) the scores in a Dataframe.
    """
    df_em = pd.DataFrame(data=np.array(results_json['embedding'][set]['scores']),
                         columns=em_col)
    df_rec = pd.DataFrame(data=np.array(results_json['reconstruction'][set]['scores']),
                          columns=['index', 'label', 'scores'])
    df = pd.merge(df_em, df_rec, how='inner', left_on='index', right_on='index', suffixes=('_em', '_rec'))
    df = df.drop(columns=['label_rec']).rename(columns={'label_em':'label'})
    return df

def get_dict_val(dictionary, path, splitchar='/'):
    """
    Return the value of the dictionnary at the given path of keys.
    ----------
    INPUT
        |---- dictionary (dict) the dict to search in.
        |---- path (str) path-like string for key access ('key1/key2/key3' for ex.).
        |---- splitchar (str) the character to use to split the path into keys.
    OUTPUT
        |---- dictionary (dict or value) the value of the dictionnary at the
        |           provided key path.
    """
    for item in path.split(splitchar):
        dictionary = dictionary[item]
    return dictionary

def plot_loss(results_list, dict_pass_to_loss, cmap_name, ax=None, lw=1, color_range=(0.25,0.75), plot_rep=True, plot_mean=False):
    """
    Plot the loss evolution for all the replicate in list of results, as well as
    the 95% confidence interval of the mean loss evolution.
    ----------
    INPUT
        |---- results_list (list of dict) list of experiemnt results where the
        |           loss evolution is stored.
        |---- dict_pass_to_loss (str) the key-path to the loss in the results.
        |---- cmap_name (str) the name of the colormap to use to plot all replicate.
        |---- ax (matplotlib.Axes) the axes where to plot.
        |---- lw (int) the line width.
        |---- color_range (tuple (lower, upper)) the range of the colormap to use.
        |---- plot_rep (bool) whether to plot all the replicate lines.
        |---- plot_mean (bool) whether to plot the mean loss evolution.
    OUTPUT
        |---- None
    """
    # find axes
    ax = plt.gca() if ax is None else ax

    # set color scheme
    cmap = matplotlib.cm.get_cmap(cmap_name)
    colors = cmap(np.linspace(color_range[0], color_range[1], len(results_list)))
    losses = []
    epochs = None
    for replicat, color in zip(results_list, colors):
        data = np.array(get_dict_val(replicat, dict_pass_to_loss))
        ax.plot(data[:,0], data[:,1], color=color, lw=lw, alpha=1)
        losses.append(data[:,1])
        epochs = data[:,0]

    losses = np.stack(losses, axis=1)
    if plot_mean: ax.plot(epochs, losses.mean(axis=1), color='black', lw=lw/2, alpha=1)
    ax.fill_between(epochs, losses.mean(axis=1) + 1.96*losses.std(axis=1),
                           losses.mean(axis=1) - 1.96*losses.std(axis=1),
                           color='gray', alpha=0.3)

def metric_barplot(metrics_scores, serie_names, group_names, colors=None, w=None, ax=None, fontsize=12, jitter=False, jitter_color=None, gap=None):
    """
    Plot a grouped barplot from the passed array, for various metrics.
    ----------
    INPUT
        |---- metric_scores (list of 2D np.array) the data to plot each element
        |           of the list is a np.array (N_replicats x N_group). The lenght
        |           of the lists gives the number of series plotted.
        |---- series_name (list of str) the names for each series (to appear in
        |           the legend).
        |---- group_names (list of str) the names of the groups (the x-ticks labels).
        |---- colors (list of str) the colors for each series. If None, colors
        |           are randomly picked.
        |---- w (float) the bar width. If None, w is automoatically computed.
        |---- ax (matplotlib Axes) the axes where to plot.
        |---- fontsize (int) the fontsize to use for the texts.
    OUTPUT
        |---- None
    """
    # find axes
    ax = plt.gca() if ax is None else ax

    n = len(metrics_scores)
    if colors is None: colors = np.random.choice(list(matplotlib.colors.CSS4_COLORS.keys()), size=n)
    if jitter_color is None: jitter_color = np.random.choice(list(matplotlib.colors.CSS4_COLORS.keys()))

    offsets = list(np.arange(-(n-1),(n-1)+2, 2))
    if w is None: w = 0.9/n
    ind = np.arange(metrics_scores[0].shape[1]) # number of different groups
    if gap:
        ind = np.where(ind + 1 > gap, ind + 0.5, ind)

    for metric, offset, name, color in zip(metrics_scores, offsets, serie_names, colors):
        means = np.nanmean(metric, axis=0)
        stds = np.nanstd(metric, axis=0)
        ax.bar(ind + offset*w/2, means, width=w, yerr=1.96*stds,
               fc=color, ec='black', lw=1, label=name)

        for i, x in enumerate(ind):
            ax.text(x + offset*w/2, means[i]-0.03, f'{means[i]:.2%}', fontsize=fontsize, ha='center', va='top', rotation=90)

        if jitter:
            for j, x in enumerate(ind):
                ax.scatter(np.random.normal(x + offset*w/2, 0.00, metric.shape[0]),
                           metric[:,j], c=jitter_color, marker='o', s=30, lw=0, zorder=5)

    handles, labels = ax.get_legend_handles_labels()
    if jitter:
        handles += [matplotlib.lines.Line2D((0,0),(0,0), lw=0, marker='o',
                    markerfacecolor=jitter_color, markeredgecolor=jitter_color, markersize=7)]
        labels += ['Measures']
    ax.legend(handles, labels, loc='upper left', ncol=1, frameon=False, framealpha=0.0,\
              fontsize=fontsize, bbox_to_anchor=(1.05, 1), bbox_transform=ax.transAxes)

    ax.set_xticks(ind)
    ax.set_xticklabels(group_names)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('', fontsize=fontsize)
    ax.set_ylim([0,1])

def plot_scores_dist1D(scores, labels, ax=None, colors=['forestgreen', 'tomato'],
                       alpha=0.5, fontsize=12, density=False, legend=False, nbin=20):
    """
    Plot the scores histogram separately by the label's value.
    ----------
    INPUT
        |---- scores (np.array) the scores vector to plot.
        |---- labels (np.array) the labels to categroize the scores.
        |---- ax (matplotlib.Axes) the axes where to plot.
        |---- colors (list of str) the two colors to use for the two categroies.
        |---- alpha (float) the transparency to use.
        |---- fontsize (int) the fontsize of the ticks.
        |---- density (bool) whether the histogram's bins sum to 1.
        |---- legend (bool) whether to add a legend (normal/abnormal) below.
        |---- nbin (int) the number of bins for the histogram.
    OUTPUT
        |---- None
    """
    # find axes
    ax = plt.gca() if ax is None else ax

    scores_0 = scores[labels == 0]
    scores_1 = scores[labels == 1]

    # abnormal distribution
    ax.hist(scores_1, bins=nbin, log=True, density=density, histtype='bar', color=colors[1], alpha=alpha, range=(scores.min(), scores.max()))
    #ax.hist(scores_1, bins=nbin, log=True, density=density, histtype='step', color=colors[1])

    # normal distribution
    ax.hist(scores_0, bins=nbin, log=True, density=density, histtype='bar', color=colors[0], alpha=alpha, range=(scores.min(), scores.max()))
    #ax.hist(scores_0, bins=nbin, log=True, density=density, histtype='step', color=colors[0])

    ax.set_ylim(ax.get_ylim())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=fontsize)

    if legend:
        handles = [matplotlib.patches.Patch(facecolor=color, edgecolor='black', alpha=alpha) for color in colors]
        labels = ['Normal', 'Abnormal']
        ax.legend(handles, labels, loc='lower center', ncol=2, frameon=False, framealpha=0.0,\
                  fontsize=fontsize, bbox_to_anchor=(0.5, -0.4), bbox_transform=ax.transAxes)

def plot_scores_dist2D(scores_1, scores_2, labels, kde=True, scatter=True, ax=None,
                       colors=['forestgreen', 'tomato'], cmaps=['Greens', 'Reds'],
                       alphas=[0.5, 0.6], fontsize=12):
    """
    Plot the two scores as a scatterplot separately by the label's value.
    ----------
    INPUT
        |---- scores_1 (np.array) the first scores vector to plot.
        |---- scores_2 (np.array) the second scores vector to plot.
        |---- labels (np.array) the labels to categroize the scores.
        |---- kde (bool) whether to plot the kde estimation by Seaborn.
        |---- scatter (bool) whether to plot the scatter points.
        |---- ax (matplotlib.Axes) the axes where to plot.
        |---- colors (list of str) the two colors to use for the two categroies.
        |---- cmaps (list of colormaps) the colormaps to use for the kde.
        |---- alphas (list of float) the transparencies to use for the points and kde.
        |---- fontsize (int) the fontsize of the ticks.
    OUTPUT
        |---- None
    """
    assert (kde != scatter) or (kde and scatter), "At least one of kde or scatter must be True."
    # find axes
    ax = plt.gca() if ax is None else ax
    # get scores per categories
    scores_1_0 = scores_1[labels == 0]
    scores_1_1 = scores_1[labels == 1]
    scores_2_0 = scores_2[labels == 0]
    scores_2_1 = scores_2[labels == 1]
    # plot kde
    if kde:
        sns.kdeplot(scores_1_1, scores_2_1, cmap=cmaps[1], shade=True,
                    shade_lowest=False, cut=0, ax=ax, alpha=alphas[1], legend=False)
        sns.kdeplot(scores_1_0, scores_2_0, cmap=cmaps[0], shade=True,
                    shade_lowest=False, cut=0, ax=ax, alpha=alphas[1], legend=False)
    if scatter:
        # add points
        ax.scatter(scores_1_1, scores_2_1, s=10, marker='o', color=colors[1], alpha=alphas[0])
        ax.scatter(scores_1_0, scores_2_0, s=10, marker='o', color=colors[0], alpha=alphas[0])
    # goodies
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=fontsize)

def metric_curves(results_list, scores_name=None, set='valid', curves=['roc', 'prc'], areas=False, ax=None, fontsize=12, em_col=['index', 'label', 'scores', 'Nsphere']):
    """
    Plot the ROC and/or Precision recall curves on the given axes for all the
    replicates on results_list.
    ----------
    INPUT
        |---- results_list
    """
    # find axes
    ax = plt.gca() if ax is None else ax

    for results in results_list:
        if scores_name is None:
            df = pd.DataFrame(data=np.array(results[set]['scores']),
                              columns=em_col)
            scores_name_tmp = 'scores'
        else:
            df = scores_as_df(results, set, em_col=em_col)
            scores_name_tmp = scores_name


        if 'roc' in curves:
            fpr, tpr, thres = roc_curve(df.label, df[scores_name_tmp])
            ax.plot(fpr, tpr, color='coral', lw=1)
            if areas : ax.fill_between(fpr, tpr, facecolor='coral', alpha=0.05)
        if 'prc' in curves:
            pre, rec, thres2 = precision_recall_curve(df.label, df[scores_name_tmp])
            ax.plot(rec, pre, color='cornflowerblue', lw=1)
            if areas : ax.fill_between(rec, pre, facecolor='cornflowerblue', alpha=0.05)

    colors = []
    xlabel, ylabel = [], []
    if 'roc' in curves:
        xlabel.append('FPR')
        ylabel.append('TPR')
        colors.append('coral')
    if 'prc' in curves:
        xlabel.append('Recall')
        ylabel.append('Precision')
        colors.append('cornflowerblue')

    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel(' ; '.join(xlabel), fontsize=fontsize)
    ax.set_ylabel(' ; '.join(ylabel), fontsize=fontsize)

    handles = [matplotlib.lines.Line2D((0,0),(1,1), color=c, lw=1) for c in colors]
    labels = [f'{name.upper()} curves' for name in curves]
    ax.legend(handles, labels, loc='lower right', ncol=1, frameon=False, fontsize=fontsize)
