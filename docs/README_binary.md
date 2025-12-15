### Entropy Balancing for Binary Treatment

#### Introduction
A method for offline causal inference in cross-sectional data. Similar use cases to PSW (Propensity Score Weighting) and PSM (Propensity Score Matching). Uses cross-entropy and weighting to match the first moments of features between treatment and control groups, while maximizing weight uniformity.

References:
- Paper: https://web.stanford.edu/~jhain/Paper/PA2012.pdf
- R package: https://github.com/cran/ebal

#### Input
* Treatment: vector, treatment variable (e.g., treatment assignment for each user), required
* X: matrix, user features (each feature as a column), required
* Y: vector, outcome variable, required
* base_weight: ideal weights, defaults to 1 for each unit, optional

#### Initialization Parameters
* coefs: initial Lagrangian multipliers (default all zeros), not recommended to modify unless familiar with the code
* max_iterations: maximum number of algorithm iterations, default 500
* constraint_tolerance: threshold for determining convergence. Represents the maximum acceptable difference between treatment and control group features, default 0.0001
* print_level: whether to print runtime details. Set to -1 to suppress output, default 0
* lr: step size, default 1. Larger values lead to faster convergence but risk exploding gradients
* PCA: whether to apply PCA decomposition to the feature matrix. Primarily solves collinearity issues, default True
* effect: estimand type, default "ATT". Can be set to "ATT", "ATC", or "ATE"

#### Usage Example

```python
import pandas as pd
from ebal import ebal_bin

df = pd.read_csv('~/data.csv')  # load data

treatment = "treatment"
tmp_df = df[df[treatment].isin([0, 1])]         # ensure treatment values are {0, 1}
tmp_df = tmp_df.sort_values(by=[treatment])
tmp_df = tmp_df.reset_index(drop=True)

Y = tmp_df.Y.values
X = tmp_df.iloc[:, 0:10].copy()                  # features
T = tmp_df[treatment].values                     # treatment
e = ebal_bin()
out = e.ebalance(T, X, Y)

# output weighted OLS results
out['wls'].summary()

# check feature balance between treatment and control groups
e.check_balance(X, T, out['w'])
```

#### Output
* converged: boolean, whether the algorithm converged
* maxdiff: maximum constraint deviation
* w: vector, weight for each unit. Use these weights in downstream curve fitting
* wls: regression results, print with summary() (see example above)

#### Notes
* Default estimand is ATT. To estimate ATE or ATC, modify the `effect` parameter
* Binary version output weights: sum(control weights) = 1; sum(treatment weights) = 1; adjust as needed
* When PCA=False, feature matrix X must be full rank with no collinear features, otherwise an error will be raised

#### Common Errors
* loss/weights showing NaN: usually caused by step size being too large, leading to exploding gradients. Try reducing `lr` or modifying initial `coefs`
* "collinearity in covariate matrix for controls (remove collinear covariates)": feature matrix is rank deficient, need to remove collinear features or set PCA=True
* No convergence after 500 iterations: try modifying initial `coefs`
* Still getting collinearity error after PCA: try adjusting the PCA feature selection threshold in the code
