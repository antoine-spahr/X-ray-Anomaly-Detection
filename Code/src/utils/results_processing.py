import numpy as np
import pandas as pd
import glob
import json
import os
import sys
sys.path.append('../')

import matplotlib
import matplotlib.pyplot as plt

def scores_as_df(results_json, set):
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
                         columns=['index', 'label', 'scores'])
    df_rec = pd.DataFrame(data=np.array(results_json['reconstruction'][set]['scores']),
                          columns=['index', 'label', 'scores'])
    df = pd.merge(df_em, df_rec, how='inner', left_on='index', right_on='index', suffixes=('_em', '_rec'))
    df = df.drop(columns=['label_rec']).rename(columns={'label_em':'label'})
    return df
