import numpy as np
import json
from sklearn.metrics import roc_auc_score

class ScoresCombiner:
    """
    Enables to combine two anomaly scores into a single one. The weights are
    ajusted to maximize the AUC on a given scores subset.
    """
    def __init__(self):
        """
        Build the ScoresCombiner.
        ----------
        INPUT
            |---- None
        OUTPUT
            |---- None
        """
        self.min_1, self.max_1 = None, None
        self.min_2, self.max_2 = None, None
        self.w_1, self.w_2 = None, None
        self.fitted = False

    def fit(self, scores_1, scores_2, labels, search_res=0.1, verbose=0, return_scores=0):
        """
        Find the parameters of the combiner based on the passed scores subset.
        It estimates the minimum and maximum of both anomaly scores to normalize
        the scores. And it grid-search the weight that maximize the AUC.
        ----------
        INPUT
            |---- scores_1 (np.array) the first set anoamly scores.
            |---- scores_2 (np.array) the second set anoamly scores.
            |---- label (np.array) the set of ground truth of anomaly (1 or 0).
            |---- search_res (float) the search resolution to use in the weight
            |           grid search. It must be 0 < search_res < 1.
            |---- verbose (int) whether to display a summary of the fitting (0 :
            |           no information displayed ; 1 : only resutls displayed ;
            |           2 : detailed display)
            |---- return_scores (int) whether to return the validation scores
            |           and/or AUC. (0 : no return ; 1 : return scores only : 2 :
            |           return AUC only ; 3 : return scores and AUC)
        OUTPUT
            |---- None
        """
        assert verbose in [0,1,2], f'Verbose = {verbose} is an invalid key. It must be one of : 0, 1, 2'
        assert return_scores in [0,1,2,3], f"The return code must be one of 0, 1, 2, or 3. {return_scores} has been passed."
        # get normalization parameter
        if verbose == 2: print('>>> Getting normalization parameters.')
        self.min_1, self.max_1 = scores_1.min(), scores_1.max()
        self.min_2, self.max_2 = scores_2.min(), scores_2.max()
        #normalized scores
        scores_1_n, scores_2_n = self._normalize_scores(scores_1, scores_2)
        if verbose == 2: print('>>> Scores Normalized.')
        # Grid search the two weights to maximize AUC
        self.w_1, self.w_2 = self._compute_weight(scores_1_n, scores_2_n, labels,
                                                  search_res=search_res, verbose=verbose)
        self.fitted = True
        # return required results
        scores = self.compute_scores(scores_1, scores_2)
        if return_scores == 1:
            return scores
        elif return_scores == 2:
            return roc_auc_score(labels, scores)
        elif return_scores == 3:
            return scores, roc_auc_score(labels, scores)

    def _normalize_scores(self, scores_1, scores_2):
        """
        Normlaize the passed scores using the fitted paramters.
        ----------
        INPUT
            |---- scores_1 (np.array) the first set anoamly scores to normalize.
            |---- scores_2 (np.array) the second set anoamly scores to normalize.
        OUTPUT
            |---- scores_n_1 (np.array) the normalized anomaly scores set 1.
            |---- scores_n_2 (np.array) the normalized anomaly scores set 2.
        """
        scores_1_n = (scores_1 - self.min_1) / (self.max_1 - self.min_1)
        scores_2_n = (scores_2 - self.min_2) / (self.max_2 - self.min_2)
        return scores_1_n, scores_2_n

    def _compute_weight(self, scores_1, scores_2, labels, search_res=0.1, verbose=False):
        """
        Grid search the weights combination that maximize the AUC of the new score.
        ----------
        INPUT
            |---- scores_1 (np.array) the first set anoamly normalized scores.
            |---- scores_2 (np.array) the second set anoamly normalized scores.
            |---- label (np.array) the set of ground truth of anomaly (1 or 0).
            |---- search_res (float) the search resolution to use in the weight
            |           grid search. It must be 0 < search_res < 1.
            |---- verbose (int) whether to display a summary of the fitting (0 :
            |           no information displayed ; 1 : only resutls displayed ;
            |           2 : detailed display)
        OUTPUT
            |---- best_weights (tuple (w_1, w_2)) the best set of weights found.
        """
        best_auc = 0.0
        best_weights = (0.0, 0.0)
        if verbose == 2: print('>>> Start weight searching.')
        for w_1 in np.arange(0, 1+search_res, search_res):
            for w_2 in np.arange(0, 1+search_res, search_res):
                if w_1 + w_2 == 1:
                    scores = w_1 * scores_1 + w_2 * scores_2
                    auc = roc_auc_score(labels, scores)
                    if verbose == 2 : print(f'>>> | w_1 = {w_1:.2f} | w_2 = {w_2:.2f} |',
                                            f'AUC = {auc:.3%} | Best AUC = {best_auc:.3%}')
                    if auc > best_auc:
                        best_auc = auc
                        best_weights = (w_1, w_2)
        if verbose in [1, 2]: print(f'>>> Best AUC of {best_auc:.3%} obtained with ',
                                    f'w_1 = {best_weights[0]:.2f} and w_2 = {best_weights[1]:.2f}.')
        return best_weights


    def compute_scores(self, scores_1, scores_2):
        """
        Compute the composite scores of the passed set using the fitted parameters.
        ----------
        INPUT
            |---- scores_1 (np.array) the first set anoamly scores.
            |---- scores_2 (np.array) the second set anoamly scores.
        OUTPUT
            |---- scores (np.array) the composite scores.
        """
        assert self.fitted, "The ScoresCombiner's parameters must be fitted before being used."
        scores_1_n, scores_2_n = self._normalize_scores(scores_1, scores_2)
        scores = self.w_1 * scores_1_n + self.w_2 * scores_2_n

        return scores

    def compute_auc(self, scores_1, scores_2, labels):
        """
        Compute the composite scores of the passed set using the fitted parameters.
        ----------
        INPUT
            |---- scores_1 (np.array) the first set anoamly scores.
            |---- scores_2 (np.array) the second set anoamly scores.
            |---- labels (np.array) the labels.
        OUTPUT
            |---- auc (float) The ROC-AUC.
        """
        assert self.fitted, "The ScoresCombiner's parameters must be fitted before being used."
        scores_1_n, scores_2_n = self._normalize_scores(scores_1, scores_2)
        scores = self.w_1 * scores_1_n + self.w_2 * scores_2_n
        auc = roc_auc_score(labels, scores)
        return auc

    def save_param(self, export_json_path):
        """
        Save the fitted parameters as JSON.
        ----------
        INPUT
            |---- export_json_path (str) the JSON filename where to save the
            |           fitted parameters.
        OUTPUT
            |---- None
        """
        with open(export_json_path, 'w') as f:
            json.dump({'score_1':{
                            'min':self.min_1,
                            'max':self.max_1,
                            'w':self.w_1},
                       'score_2':{
                            'min':self.min_2,
                            'max':self.max_2,
                            'w':self.w_2}}, f)

    def load_param(self, json_param_path):
        """
        Load the fitted parameters from a JSON file.
        ----------
        INPUT
            |---- json_param_path (str) the JSON filename to load the parameters from.
        OUTPUT
            |---- None
        """
        with open(json_param_path, 'r') as fp:
            params = json.load(fb)
            self.min_1 = params['score_1']['min']
            self.max_1 = params['score_1']['max']
            self.w_1 = params['score_1']['w']
            self.min_2 = params['score_2']['min']
            self.max_rex = params['score_2']['max']
            self.w_2 = params['score_2']['w']
            self.fitted = True
