# Baseball
Analysis of historical baseball games using ML.

## SHAP Feature Impact
The `SHAPExplainer` class allows you to estimate the impact of individual features on the outcome of a target variable:

Globally across a large sample of games:
![Global Impact](./images/global_impact.png)

From the perspective of different values a specific feature could assume, and the average marginal impact of each of those values:
![Feature Levels Marginal Impact](./images/feature_levels_marginal_impact.png)
In this examples, the model would have predicted a median win probability of ~56.3% across home games for the 2019 Boston Red Sox, using all of the input features. However, isolating the marginal impact of the `team` feature (the model's quantitative assessment of the quality of the 2019 Boston Red Sox holding all other features constant), we can see that particular value for the team feature had a median marginal contribution of almost -5% to the win probability. This values is well below what we would have predicted if the model's predictions and the marginal impact of the `team` feature were perfectly correlated.

Or locally, detailing how each feature contributed to the prediction for a single game:
![Local Impact](./images/local_impact.png)