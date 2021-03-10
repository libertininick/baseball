from collections import defaultdict
import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap


class InputLevels():
    """Class for transforming the values of a single input feature into levels for embedding.
    """
    def __init__(self, values, min_obs=1, nq=None):
        """
        Args:
            values (ndarray): Input values
            min_obs (int, optional): Minimum number of observations to keep level
            nq (int, optional): Number of quantiles to buck a numerical feature into
        """
        if nq:
            self.quantized = True
            
            # Coerce to float
            values = values.astype(np.float)
            
            # Keep only finite values
            values = values[np.isfinite(values)]
            
            # Use quantiles as levels
            quantiles = np.linspace(0, 1, nq + 1)
            levels = np.quantile(values, quantiles)
            levels[0], levels[-1] = -np.inf, np.inf  # set +/-Inf as bounds
            self.levels = np.unique(levels)          # remove duplicate levels
            
            # Number of levels (bins)
            # np.digitize(right=False): bins[i-1] <= x < bins[i]; n_levels = n_bins - 1
            # +1 for NaN level
            # +1 for baseline
            self.n_levels = (len(self.levels) - 1) + 1 + 1
            
        else:
            self.quantized = False
            
            # Unique levels and counts
            levels, counts = np.unique(
                [x for x in values if x], # Remove Nones
                return_counts=True
            )
            
            # Remove levels with too few observations
            self.levels = levels[counts >= min_obs]
            
            # Number of levels
            # +1 for unknown/too few
            # +1 for baseline
            self.n_levels = len(self.levels) + 1 + 1
            
        # Embedding dim
        self.emb_dim = max(2, int((self.n_levels - 2)**0.5)//2*2)

        # Map levels to indexes starting a 2
        # Idx 0 reserved for baseline
        # Idx 1 resevered for unknown/too few/NaN
        self.levels_to_idxs = {
            lvl: i
            for i, lvl
            in enumerate(self.levels, 2)
        }

        # Map indexes back to levels
        self.idxs_to_levels = {v: k for k, v in self.levels_to_idxs.items()} 
        
    def get_indexes(self, x):
        """Map inputs to indexes to be consumed by an embedding layer
        """           
        if self.quantized:
            x = np.array(x, dtype=np.float)
            binned_data = np.digitize(x, self.levels) + 1
            binned_data[~np.isfinite(x)] = 1
            return binned_data
        else:
            return np.vectorize(lambda x_i: self.levels_to_idxs.get(x_i, 1))(x)

    def get_levels(self, x):
        """Map from embedding indexes back to input values
        """
        return np.vectorize(lambda x_i: self.idxs_to_levels.get(x_i, None))(x)


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
        for i, feats in enumerate(self.all_evals):
            # Toggle ON the features in this specific model evaluation
            # Those features not in this eval will be set to their baseline value
            self.feat_toggle[i, feats] = 1
        
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
                pred_before = model_evals.get(lhs)       # Model's prediction before adding feature_i
                pred_after = model_evals.get(rhs)        # Model's prediction after adding feature_i
                impact += (pred_after - pred_before)*wt  # Marginal impact of adding feature_i
                total_wt += wt
            feature_impacts[f] = impact/total_wt
            
        return feature_impacts
    
    def get_shap_values(self, x):
        """Compute SHAP values for each observation (row) in input matrix x relative
        to baseline value for each feature.
        
        Args:
            x (ndarray): Matrix of input feature observations (n_obs, n_feats)
        
        Returns:
            shap_values (ndarray): SHAP value for each observation, for each feature, for each output 
                                   if model has single output: (n_obs, n_feat)
                                   if model has multiple outputs: (n_obs, n_feat, n_out)
        """
        # Compute SHAP values for each row
        shap_values = []
        for row in x:
            model_evals = self._evaluate_model(row)
            shap_values.append(self._est_feature_impacts(model_evals))
            
        # Sort by feature 
        shap_values = pd.DataFrame(shap_values)
        shap_values = shap_values[sorted(shap_values.columns)].values
        
        # Reshape outputs with a 3rd dim if model has multiple outputs
        n_obs, n_feat = shap_values.shape
        if isinstance(shap_values[0,0], np.ndarray):
            shap_values = np.stack(shap_values.flatten(), axis=0).reshape((n_obs, n_feat, -1))
        else:
            shap_values = shap_values.reshape((n_obs, n_feat))
        
        return shap_values

    def _normalize_by_observation(self, shap_values, e=1e-6):
        """Normalize each feature's impact relative to the other features for each observation.
        Normalized impact can be interpreted as relative importance of each feature for each observation.

        Args:
            shap_values (ndarray): SHAP values for each feature across a range of observations (n_obs, n_feat, ...)
            e (float, optional): Small value to prevent division by 0

        Returns:
            shap_values_norm (ndarray): Normalized SHAP values in range [0,1] and each row sums to 1. (n_obs, n_feat, ...)
        """

        abs_values = np.abs(shap_values)
        obs_sums = np.sum(abs_values, axis=1) + e  # Sum values for each observation

        if shap_values.ndim == 3:
            return abs_values/obs_sums[:, None, :]
        else:
            return abs_values/obs_sums[:, None]

    def plot_global_impact(self, shap_values, relative=True, target_name=None):
        """Plot the impact/importance of each feature across a range of observations (global)

        Args:
            shap_values (ndarray): SHAP values for each feature across a range of observations (n_obs, n_feat, ..)
        """
        labels = np.array(self.var_names)
        n = len(labels)


        if relative:
            # Normalize impact to % contribution for each feature
            values = self._normalize_by_observation(shap_values)
        else:
            values = np.abs(shap_values)

        if values.ndim == 3:
            # Average impact across model's outputs
            values = np.mean(values, axis=-1)
        
        # Mean/median impact of each feature
        mean_impact = np.mean(values, axis=0)
        median_impact = np.median(values, axis=0)
        
        # Sort features smallest magnitude to largest
        sort_idxs = np.argsort(median_impact)
        values = values[:, sort_idxs]
        labels = labels[sort_idxs]
        
        # Plot
        fig, ax = plt.subplots(figsize=(15, int((n+1)*0.5)))
        boxplot = ax.boxplot(
            values,
            labels=labels,
            sym='',                                 
            whis=(5, 95),                           
            widths=0.8,
            vert=False,                             
            patch_artist=True,                      
            boxprops=dict(facecolor=(0,0,0,0.10)),
        )

        # Make sure each feature is listed on y-axis
        ax.set_yticks(np.arange(n) + 1)
        
        # Color median line
        plt.setp(boxplot['medians'], color='red')
        
        # Add mean points
        ax.plot(
            mean_impact[sort_idxs],
            np.arange(n) + 1, 
            linewidth=0,
            marker='D', 
            markersize=5,
            markerfacecolor='k',
            markeredgecolor='k'
        )
        
        if relative:
            ax.set_xlabel('Relative importance (%)')
            
            ax.set_xlim(left=0, right=None)  # Bound x axis at 0
            ax.set_xticklabels([f'{x:.0%}' for x in ax.get_xticks()])

            ax.axvline(
                x=np.median(values), 
                color='blue',
                linestyle='--', 
                label=f'Median importance ({np.median(values):.2%})'
            )
        else:
            ax.set_xlabel(f'Magnitude of impact')

            ax.axvline(
                x=np.median(values), 
                color='blue',
                linestyle='--', 
                label=f'Median impact ({np.median(values):.2f})'
            )
        
        # Plot title
        if target_name is None:
            target_name = 'Target Variable'
        ax.set_title(label=f'Global Impact of Features on {target_name}', loc='left', fontdict={'fontsize': 16})
        
        ax.legend()
        
        return plt, ax

    def plot_feature_levels_marginal_impact(
        self, 
        feat_values,
        feat_shap_values, 
        model_predictions,
        yh_ref=None,
        n_label=None,
        summary_metric='median',
        colors=('red', 'blue'),
        reverse_colors=False):
        """For a specific feature, plot the average marginal impact vs. average model prediction for each level.

        Args:
            feat_values (ndarray): Array of input values for the feature (n_obs,)
            feat_shap_values (ndarray): Array of SHAP values for the feature (n_obs,)
            model_predictions (ndarray): Overall model prediction for each observation (n_obs)
            yh_ref (float, optional): Model's prediction using baseline inputs
            n_label (int, optional): Number of "worst"/"best" levels to label 
            summary_metric (str, optional): Metric to aggregate SHAP values and model predictions 
                                            for each level ('mean', 'median', ...)
            colors (tuple): Color pair to use for coloring regions of plot
                            color_a: "Bad" levels
                            color_b: "Good" levels
            reverse_colors (bool): Set to True if lower model predictions are better
        """

        # Fill reference model prediction if None
        if yh_ref is None:
            yh_ref = np.median(model_predictions)

        # Summarize marginal impact and model prediction for each level in feature
        levels_summary = (
            pd.DataFrame({
                'level': feat_values,
                'marginal_impact': feat_shap_values,
                'model_prediction': model_predictions,
            })
            .fillna('None')
            .groupby('level')
            .agg(summary_metric)
            .reset_index()
        )

        # Enumerate perfect correlation ordering of model prediction and marginal impacts
        perfect_cor = (
            pd.DataFrame({
                'model_prediction': np.sort(levels_summary['model_prediction']), 
                'marginal_impact_pc': np.sort(levels_summary['marginal_impact']),
            })
            .groupby('model_prediction')
            .mean()
            .reset_index()
        )
        levels_summary = pd.merge(levels_summary, perfect_cor, on='model_prediction')

        # Find difference between each level's median marginal impact and the perfect correlation est of its marginal impact
        levels_summary['marginal_impact_diff'] = levels_summary['marginal_impact'] - levels_summary['marginal_impact_pc']


        y_min, y_max = min(levels_summary['marginal_impact']), max(levels_summary['marginal_impact'])
        x_min, x_max = min(levels_summary['model_prediction']), max(levels_summary['model_prediction'])

        # Plot 
        fig, ax = plt.subplots(figsize=(15,15)) 
        
        color_a, color_b = colors
        if reverse_colors:
            color_a, color_b = color_b, color_a 

        # Marginal impact vs. average model prediction scatter 
        ax.errorbar( 
            levels_summary['model_prediction'],  
            levels_summary['marginal_impact'],  
            yerr=[ 
                np.where(levels_summary['marginal_impact_diff'] > 0, levels_summary['marginal_impact_diff'], 0), 
                np.where(levels_summary['marginal_impact_diff'] < 0, -levels_summary['marginal_impact_diff'], 0) 
            ], 
            color='black', 
            alpha=0.5, 
            fmt='.', 
            ecolor='gray', 
            elinewidth=1, 
        ) 
        ax.set_xlabel('Model Prediction') 
        ax.set_ylabel('Marginal Impact') 

        # Plot line of hypothetical perfect correlation between model's prediction and marginal impact 
        ax.plot( 
            np.sort(levels_summary['model_prediction']),  
            np.sort(levels_summary['marginal_impact']), 
            "k:", 
            label='Perfect Correlation' 
        ) 
        ax.legend(loc='upper left') 

        # Basline/Reference prediction vline 
        ax.axvline(x=yh_ref, color='black', alpha=0.5) 

        # 0 Marginal impact hline 
        ax.axhline(y=0, color='black', alpha=0.5) 

        if n_label:
            # Label "worst"/"best" levels 
            for i, (_, row) in enumerate( 
                levels_summary 
                .query('marginal_impact < 0') 
                .query('marginal_impact_diff < 0') 
                .sort_values(by=['marginal_impact_diff', 'marginal_impact']) 
                .head(n_label) 
                .loc[:, ['level', 'model_prediction', 'marginal_impact']] 
                .sort_values(by='model_prediction') 
                .iterrows() 
            ): 
                label, _x, _y = row.values 
                ax.annotate( 
                    text=label,  
                    xy=(_x, _y), 
                    xytext=(_x + x_max*0.05, _y - y_max*0.15/(i+1)), 
                    fontsize=12, 
                    color=color_a, 
                    arrowprops=dict(arrowstyle='->') 
                ) 

            for i, (_, row) in enumerate( 
                levels_summary 
                .query('marginal_impact > 0') 
                .query('marginal_impact_diff > 0') 
                .sort_values(by=['marginal_impact_diff', 'marginal_impact']) 
                .tail(n_label) 
                .loc[:, ['level', 'model_prediction', 'marginal_impact']] 
                .sort_values(by='model_prediction') 
                .iterrows() 
            ): 
                label, _x, _y = row.values 
                ax.annotate( 
                    text=label,  
                    xy=(_x, _y), 
                    xytext=(_x + x_max*0.05, _y + y_max*0.15/(i+1)), 
                    fontsize=12, 
                    color=color_b, 
                    arrowprops=dict(arrowstyle='->') 
                ) 

        # Shade regions 
        ax.fill_between(  
            np.sort(levels_summary['model_prediction']),  
            np.full_like(levels_summary['marginal_impact'], fill_value=y_min),  
            np.sort(levels_summary['marginal_impact']),  
            color=color_a,   
            alpha=0.15  
        ) 
        ax.fill_between( 
            np.sort(levels_summary['model_prediction']),  
            np.sort(levels_summary['marginal_impact']),  
            np.full_like(levels_summary['marginal_impact'], fill_value=y_max),  
            color=color_b,  
            alpha=0.15 
        ) 
        ax.fill_between( 
            [x_min, yh_ref],  
            [y_min, y_min],  
            [0, 0],  
            color=color_a,  
            alpha=0.15 
        ) 
        ax.fill_between( 
            [yh_ref, x_max],  
            [0, 0],  
            [y_max,y_max], 
            color=color_b,  
            alpha=0.15 
        )

        return fig, ax

    def plot_feature_level_embeddings(
        self,
        level_embeddings,
        emb_idx_level_mapper,
        feat_values,
        feat_shap_values,
        n_label=None,
        summary_metric='median',
        cmap='RdBu',
        reverse_colors=False,
        seed=1234):
        """For a specific feature, plot a 2-dimensional projection of the embeddings for the feature's levels.
        Color each level by a summary of its marginal impact on the model's predictions.

        Args:
            level_embeddings (ndarray): Model embedding weights for feature's levels (n_levels, emb_dim)
            emb_idx_level_mapper (function): Function for mapping level indexes to level names
            feat_values (ndarray): Array of input values for the feature (n_obs,)
            feat_shap_values (ndarray): Array of SHAP values for the feature (n_obs,)
            n_label (int, optional): Number of "worst"/"best" levels to label 
            summary_metric (str, optional): Metric to aggregate SHAP values and model predictions 
                                            for each level ('mean', 'median', ...)
            cmap (str): Color map to shade points
            reverse_colors (bool): Set to True if lower model predictions are better
            seed (int): Random state seed for UMAP reducer
        """

        # Reduce embedding dim to 2D
        reducer = umap.UMAP(
            n_components=2,
            random_state=seed
        )
        level_embeddings = reducer.fit_transform(level_embeddings)
        
        # Map embedding indexes to level names
        levels = emb_idx_level_mapper(np.arange(len(level_embeddings)))
        levels[0] = 'Baseline'
        levels[1] = 'None'

        # DataFrame
        level_embeddings = pd.DataFrame({
            'level': levels,
            'emb_x': level_embeddings[:,0],
            'emb_y': level_embeddings[:,1],
        })

        # Summarize marginal impact for each level in feature
        levels_summary = (
            pd.DataFrame({
                'level': feat_values,
                'marginal_impact': feat_shap_values,
            })
            .fillna('None')
            .groupby('level')
            .agg(summary_metric)
            .reset_index()
        )

        # Merge level embeddings with marginal impact summary
        level_embeddings = (
            pd.merge(
                left=level_embeddings,
                right=levels_summary,
                on='level',
                how='left'
            )
            .fillna(0)
        )

        x_max, y_max = max(level_embeddings['emb_x']), max(level_embeddings['emb_y'])
        cmap_range = np.max(np.abs(level_embeddings['marginal_impact']))

        # Plot
        fig, ax = plt.subplots(figsize=((15,12)))

        color_map = plt.cm.get_cmap(cmap)
        if reverse_colors:
            color_map = color_map.reversed()

        scatter = ax.scatter(
            level_embeddings['emb_x'].values,
            level_embeddings['emb_y'].values,
            s=100,
            c=level_embeddings['marginal_impact'].values,
            cmap=color_map,
            vmin=-cmap_range,
            vmax=cmap_range,
            edgecolors='black'
        )

        # Remove tick labels
        ax.set_xticklabels(['' for x in ax.get_xticks()])
        ax.set_yticklabels(['' for x in ax.get_yticks()])

        # Create colorbar 
        cbar = ax.figure.colorbar(scatter, ax=ax) 
        cbar.ax.set_ylabel("Marginal Impact", rotation=-90, va="bottom")

        # Baseline cords
        ax.axvline(x=level_embeddings['emb_x'][0], color='black', alpha=0.5, linestyle='--')
        ax.axhline(y=level_embeddings['emb_y'][0], color='black', alpha=0.5, linestyle='--')

        if n_label:
            # Label "worst"/"best" levels 
            for _, row in ( 
                level_embeddings 
                .sort_values(by='marginal_impact') 
                .head(n_label) 
                .sort_values(by='marginal_impact', ascending=False) # Label largest magnitude last
                .loc[:, ['level', 'emb_x', 'emb_y']]
                .iterrows() 
            ): 
                label, _x, _y = row.values 
                ax.annotate( 
                    text=label,  
                    xy=(_x, _y), 
                    xytext=(_x + x_max*0.02, _y + y_max*0.02), 
                    fontsize=8, 
                    color='black',
                    backgroundcolor='white',
                    arrowprops=dict(arrowstyle='->') 
                ) 

            for _, row in ( 
                level_embeddings 
                .sort_values(by='marginal_impact') 
                .tail(n_label) 
                .loc[:, ['level', 'emb_x', 'emb_y']] 
                .iterrows() 
            ): 
                label, _x, _y = row.values 
                ax.annotate( 
                    text=label,  
                    xy=(_x, _y), 
                    xytext=(_x + x_max*0.02, _y + y_max*0.02), 
                    fontsize=8, 
                    color='black',
                    backgroundcolor='white',
                    arrowprops=dict(arrowstyle='->') 
                ) 

        return fig, ax

    def plot_local_impact(
        self, 
        obs_shap_values, 
        yh_ref,
        ref_shap_values=None, 
        target_name=None,
        colors=('red', 'blue'),
        reverse_colors=False):
        """
        """

        labels = np.array(self.var_names)

        if ref_shap_values is not None:
            values = obs_shap_values - ref_shap_values
        else:
            values = obs_shap_values

        # Sort features
        sort_idxs = np.argsort(values)
        values = values[sort_idxs]
        labels = labels[sort_idxs]
        
        # Add starting point
        labels = [''] + list(labels)
        values = [0] + list(values)
        
        n = len(values)
        idx = np.arange(n)

        fig, ax = plt.subplots(figsize=(15, int((n+1)*0.5)))

        color_a, color_b = colors
        if reverse_colors:
            color_a, color_b = color_b, color_a

        # Feature impact bars
        ax.barh(
            idx,
            width=values,
            color=[color_b if x >= 0 else color_a for x in values],
            edgecolor='black',
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
            color=color_b if total_impact >= 0 else color_a, 
            linestyle='--'
        )

        # Axis labels
        ax.set_xticklabels([f'{x:.2f}' for x in ax.get_xticks() + yh_ref])
        ax.set_xlabel(target_name)

        ax.set_yticks(idx)
        ax.set_yticklabels(labels)

        if target_name is None:
            target_name = 'Target Variable'
        ax.set_title(label=f'Local Impact of Features on {target_name}', loc='left', fontdict={'fontsize': 16})

        return fig, ax
