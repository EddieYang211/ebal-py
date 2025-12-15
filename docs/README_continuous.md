### Entropy Balancing for Continuous Treatment

#### Introduction
A method for offline causal inference in cross-sectional data. Designed for multi-valued or continuous treatments. Uses cross-entropy and weighting to eliminate correlation between treatment and features, while maximizing weight uniformity.

References:
- Paper: https://web.stanford.edu/~jhain/Paper/PA2012.pdf
- R package: https://github.com/cran/ebal

#### Input
* Treatment: vector, treatment variable (e.g., dosage amount for each user), required
* X: matrix, user features (each feature as a column), required
* base_weight: ideal weights, defaults to 1 for each unit, optional

#### Initialization Parameters
* coefs: initial Lagrangian multipliers (default all zeros), not recommended to modify unless familiar with the code
* max_iterations: maximum number of algorithm iterations, default 500
* constraint_tolerance: threshold for determining convergence. Represents the maximum acceptable constraint violation, default 0.0001
* print_level: whether to print runtime details. Set to -1 to suppress output, default 0
* lr: step size, default 1. Larger values lead to faster convergence but risk exploding gradients
* PCA: whether to apply PCA decomposition to the feature matrix. Primarily solves collinearity issues, default True
* max_moment_treat: order of moments to balance for treatment, default 2
* max_moment_X: order of moments to balance for covariates, currently only supports 1

#### Usage Example

```python
import pandas as pd
from ebal import ebal_con

df = pd.read_csv('~/data.csv')  # load data

Y = df.Y.values
X = df.iloc[:, 0:10].copy()                      # features
T = df.T.values                                  # treatment, e.g., dosage amount

e = ebal_con()
out = e.ebalance(T, X)

# check correlation between treatment and features
e.check_balance(X, T, out['w'])
```

#### Output
* converged: boolean, whether the algorithm converged
* maxdiff: maximum constraint deviation
* w: vector, weight for each unit. Use these weights in downstream curve fitting

#### Notes
* Continuous version output weights: sum(weights) = 1; adjust as needed for your use case
* When PCA=False, feature matrix X must be full rank with no collinear features, otherwise an error will be raised

#### Common Errors
* loss/weights showing NaN: usually caused by step size being too large, leading to exploding gradients. Try reducing `lr` or modifying initial `coefs`
* "collinearity in covariate matrix for controls (remove collinear covariates)": feature matrix is rank deficient, need to remove collinear features or set PCA=True
* No convergence after 500 iterations: try modifying initial `coefs`
* Still getting collinearity error after PCA: try adjusting the PCA feature selection threshold in the code
