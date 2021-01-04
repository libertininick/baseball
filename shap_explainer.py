from collections import defaultdict
import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SHAPExplainer():
    """Tools for estimating the contribution of each input feature to a 
    statistical model's predictions.

    Can be used for local explanation of input features; i.e. explaining the impact of
    of each feature for a single observation relative to some baseline or reference observation.

    Or can be used to understand global importance of each feature across a set of observations.

    Implementation is based on SHapley Additive exPlanations from "A Unified Approach to 
    Interpreting Model Predictions" (https://arxiv.org/pdf/1705.07874.pdf).

    Estimations of SHAP values implemented in a model agnostic way. 
    """
    
    def __init__(self, prediction_fxn, baseline_values, var_names=None, max_evals=2048):
        """
        Args:
            prediction_fxn (function): Wrapper for model's prediction
            baseline_values (ndarray): Baseline value to use for each feature (n_feats,)
            var_names (list): Input feature names
            max_evals (int): Maximum number of model evaluations
        """
        self.prediction_fxn = prediction_fxn
        self.n_feats = len(baseline_values)
        self.baseline_values = baseline_values[None, :]
        self.var_names = var_names
        if var_names:
            assert len(var_names) == self.n_feats, "Number of variable names must match number of features"
        
        # Baseline model prediction
        self.yh_baseline = self.prediction_fxn(self.baseline_values).item()

        # Model evaluation combinations
        self.max_evals = 2*self.n_feats + max_evals
        self.all_evals, self.feat_marginal_evals = self._enumerate_model_evaluations()

        # On/Off feature toggle
        self.feat_toggle = np.zeros((len(self.all_evals), self.n_feats))
        for i, toggle in enumerate(self.all_evals):
            self.feat_toggle[i, toggle] = 1
        
    def _enumerate_model_evaluations(self):
        """Enumerate a set of feature combinations from toggeling each input 
        feature between a specific observed value and a reference/baseline value.
        
        These feature combinations are the combinations the model needs to be 
        evaluated on to calculate the marginal impact each feature has on the 
        output of the model.

        Returns:
            all_evals (set): Unique set of model evaluations
            feat_marginal_evals (dict): List of marginal evaluations for each feature
        """

        # All possible feature combinations
        n_evals = 2**self.n_feats 

        all_evals = set()
        feat_marginal_evals = defaultdict(list)
        if n_evals <= self.max_evals:
            # Enumerate every combination
            feats = set(range(self.n_feats))
            for f_i in feats:
                # Get all other model features except feature i
                other_feats = feats.difference([f_i])

                # Enumerate each marginal calculation for feature i
                for r in range(self.n_feats):
                    # Combinations of model features before adding feature i
                    for cmb in itertools.combinations(other_feats, r):
                        # Model evaluations
                        lhs = cmb                          # Before adding feature_i
                        rhs = tuple(sorted(cmb + (f_i,)))  # After adding feature i
                        all_evals.update([lhs, rhs])

                        # Weighting for marginal sample based on # permutations
                        prior_perms = max(1, math.factorial(r))
                        after_perms = max(1, math.factorial(self.n_feats - r - 1))
                        wt = prior_perms*after_perms

                        # Tabulate marginal model evaluations
                        feat_marginal_evals[f_i].append((lhs, rhs, wt))
        else:
            # Take a sample from possible permutations
            while len(all_evals) < self.max_evals:
                perm = np.random.permutation(self.n_feats)

                for i in range(self.n_feats):
                    f_i = perm[i]

                    # Model evaluations
                    lhs = tuple(sorted(perm[:i]))
                    rhs = tuple(sorted(perm[:i+1]))
                    all_evals.update([lhs, rhs])

                    # Tabulate marginal model evaluations
                    feat_marginal_evals[f_i].append((lhs, rhs, 1))

        return all_evals, feat_marginal_evals
    
    def _evaluate_model(self, obs_values):
        # Feature combo matrix
        combos = (
            self.feat_toggle*(np.array(obs_values)[None, :]) + 
            (1 - self.feat_toggle)*(self.baseline_values)
        )
        
        # Model's prediction for each sample
        yh = self.prediction_fxn(combos)
        
        # Evaluation map
        model_evals = {
            combo: yh[i]
            for i, combo
            in enumerate(self.all_evals)
        }
        
        return model_evals
    
    def _est_feature_impacts(self, model_evals):
        """Estimate the marginal impact of each feature to the model's prediction"""
        feature_impacts = dict()
        for f, f_evals in self.feat_marginal_evals.items():
            impact, total_wt = 0, 0
            for lhs, rhs, wt in f_evals:
                impact += (model_evals.get(rhs) - model_evals.get(lhs))*wt
                total_wt += wt
            feature_impacts[f] = impact/total_wt
            
        return feature_impacts
    
    def shap_values(self, x):
        """Compute SHAP values for each observation (row) in input matrix x relative
        to baseline value for each feature.

        Args:
            x (ndarray): Matrix of input feature observations (n_obs, n_feats)

        Returns:
            output (dict):
                 features (DataFrame): Input feature values
                 shap_values (DataFrame): SHAP value for each feature for each observation in inputs
        """

        # Compute SHAP values for each row
        values = []
        for row in x:
            model_evals = self._evaluate_model(row)
            values.append(self._est_feature_impacts(model_evals))

        # Result DataFrames
        inputs = pd.DataFrame(x)
        values = pd.DataFrame(values)
        values = values[sorted(values.columns)]
        if self.var_names:
            inputs.columns = self.var_names
            values.columns = self.var_names
            
        values['total'] = np.sum(values, axis=1)
        yh = self.prediction_fxn(x)
        values['yh'] = yh

        return {
            'features': inputs,
            'shap_values': values,
        }

    def plot_global_impact(self, shap_values, relative=True, target_name=None):
        values = shap_values.loc[:, self.var_names].values
        labels = np.array(self.var_names)

        if relative:
            row_sums = np.sum(np.abs(values), axis=1) + 1e-6
            values = values/row_sums[:, None]

        # Avg impact of each feature based on absolute SHAP values
        feature_impact = np.mean(np.abs(values), axis=0)

        # Sort features largest ABS impact to smallest
        sort_idxs = np.argsort(feature_impact)
        values = np.abs(values[:, sort_idxs])
        labels = labels[sort_idxs]

        # Plot
        fig, ax = plt.subplots(figsize=(15,10))

        boxplot = ax.boxplot(
            values,
            labels=labels,
            sym='',                                 # No fliers
            whis=(5, 95),                           # 5% and 95% whiskers
            widths=0.8,
            vert=False,                             # Horizontal plot
            patch_artist=True,                      # Need this flag to color boxes
            boxprops=dict(facecolor=(0,0,0,0.10)),  # Color faces black with 10% alpha
        )

        # Color median line
        plt.setp(boxplot['medians'], color='red')

        # Add mean points
        ax.plot(
            feature_impact[sort_idxs],
            np.arange(len(feature_impact)) + 1, 
            linewidth=0,
            marker='D', 
            markersize=5,
            markerfacecolor='k',
            markeredgecolor='k'
        )

        if relative:
            ax.set_xticklabels([f'{x:.0%}' for x in ax.get_xticks()])
            ax.set_xlabel('Relative impact (%)')
        else:
            ax.set_xlabel(target_name)

        ax.set_yticks(np.arange(len(feature_impact)) + 1)
        ax.set_ylabel('Feature')

        if target_name is None:
            target_name = 'Target Variable'
        ax.set_title(label=f'Global Impact of Features on {target_name}', loc='left', fontdict={'fontsize': 16})

        return plt, ax

    def plot_local_impact(self, obs, ref_obs=None, target_name=None):
        if ref_obs is not None:
            values = obs - ref_obs
            ref_value = ref_obs['yh']
        else:
            values = obs
            ref_value = self.yh_baseline
        
        values = values[self.var_names].sort_values()
        
        labels = [''] + list(values.index)
        values = [0] + list(values)
        
        n = len(values)
        idx = np.arange(n)

        fig, ax = plt.subplots(figsize=(7,10))

        # Feature impact bars
        ax.barh(
            idx,
            width=values,
            color=['blue' if x >= 0 else 'red' for x in values],
            height=1,
            alpha=0.5
        )

        # Cumulative impact line
        ax.plot(
            np.cumsum(values), 
            idx,
            color='black',
            alpha=0.5,
            marker='o',
        )

        # Vertical lines
        ax.axvline(x=0, color='black', linestyle='--')
        total_impact = np.sum(values)
        ax.axvline(
            x=total_impact, 
            color='blue' if total_impact >= 0 else 'red', 
            linestyle='--'
        )

        # Axis labels
        ax.set_xticklabels([f'{x:.2f}' for x in ax.get_xticks() + ref_value])
        ax.set_xlabel(target_name)

        ax.set_yticks(idx)
        ax.set_yticklabels(labels)
        ax.set_ylabel('Feature')

        if target_name is None:
            target_name = 'Target Variable'
        ax.set_title(label=f'Local Impact of Features on {target_name}', loc='left', fontdict={'fontsize': 16})

        return fig, ax
