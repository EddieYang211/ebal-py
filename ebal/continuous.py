import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

class ebal_con:
    """
    Implementation of Entropy Balancing for continuous treatment.

    Author: Eddie Yang, based on work of Hainmueller (2012), Xu & Yang (2021), Tübbicke (2020) and Vegetabile et al. (2021)

    Params:
        coefs: Lagrangian multipliers, refer to Hainmueller (2012)
        max_iterations: maximum number of iterations to find the solution weights, default 500
        constraint_tolerance: tolerance level for covariate difference between the treatment
            and control group, default 1e-4
        print_level: level of details to print out, default 0. Set to -1 to suppress all output.
        lr: step size, default 1. Increase to make the algorithm converge faster
            (at the risk of exploding gradient)
        PCA: whether to apply PCA to covariates for dimensionality reduction, default True
        max_moment_treat: order of moment to be balanced for the treatment vector, default 2
        max_moment_X: order of moment to be balanced for the covariates, currently only supports 1

    Output:
        converged: boolean, whether the algorithm converged
        maxdiff: maximum covariate difference between treatment and control groups
        w: solution weights for the units
    """

    def __init__(self,
        coefs = None,
        max_iterations = 500,
        constraint_tolerance = 0.0001,
        print_level=0,
        lr=1,
        PCA=True,
        max_moment_treat=2,
        max_moment_X=1):

        self._initial_coefs = coefs
        self.coefs = coefs
        self.max_iterations = max_iterations
        self.constraint_tolerance = constraint_tolerance
        self.print_level = print_level
        self.lr = lr
        self.max_moment_treat = max_moment_treat
        self.max_moment_X = max_moment_X
        self.PCA = PCA


    def ebalance(
        self,
        Treatment,
        X,
        base_weight=None):

        self.coefs = self._initial_coefs

        Treatment = np.asarray(Treatment).reshape(-1, 1)
        X = np.asarray(X)

        if np.isnan(Treatment).any():
            raise ValueError("Treatment contains missing data")

        if np.var(Treatment) == 0:
            raise ValueError("Variance of treatment indicator = 0. Treatment indicator must not be a constant")

        if len(np.unique(Treatment)) == 2:
            raise ValueError("Treatment has 2 unique values. Consider using the binary version of entropy balancing")

        if np.isnan(X).any():
            raise ValueError("X contains missing data")

        if not Treatment.shape[0] == X.shape[0]:
            raise ValueError("length(Treatment) != nrow(X)")

        if not isinstance(self.max_iterations, int):
            raise TypeError("max_iterations must be an integer")

        if self.PCA:
            pca_model = PCA()
            X_c = X - X.mean(axis=0)
            X_c_pca = pca_model.fit_transform(X_c)
            X = X_c_pca[:,(pca_model.explained_variance_>=1) | (pca_model.explained_variance_ratio_>=0.001)]
            if self.print_level >= 0:
                print(f"PCA on X successful; With {X.shape[1]} dimensions\n" + "-"*35)

        # generate higher moments and standardization
        t_tmp = PolynomialFeatures(degree=self.max_moment_treat, include_bias=False).fit(Treatment)
        t_mat = t_tmp.transform(Treatment)
        t_mat = (t_mat - t_mat.mean(axis=0))/(t_mat.std(axis=0))

        X_mean = (X - X.mean(axis=0))/(X.std(axis=0))


        gTX_int = np.multiply(X_mean, t_mat[:,0].reshape(-1, 1))
        gTX = np.column_stack((t_mat, X_mean, gTX_int))
        gTX = np.column_stack((np.ones(X_mean.shape[0]).reshape(-1, 1), gTX))

        if base_weight is None:
            base_weight = np.ones(Treatment.shape[0])[..., np.newaxis]
        else:
            base_weight = np.asarray(base_weight).reshape(-1, 1)

        # set up elements
        if not np.linalg.matrix_rank(gTX) == gTX.shape[1]:
            raise ValueError("collinearity in covariate matrix for controls (remove collinear covariates)")

        tr_total = gTX.sum(axis=0)
        tr_total[-gTX_int.shape[1]:] = 0
        tr_total[0] = 1
        tr_total = tr_total.reshape(-1, 1)

        if self.coefs is None:
            self.coefs = np.zeros(gTX.shape[1]).reshape(-1, 1)
        else:
            self.coefs = np.asarray(self.coefs)

        if not self.coefs.shape[0]==gTX.shape[1]:
            raise ValueError("coefs needs to have same length as number of covariates plus one")

        if self.print_level >= 0:
            print(f"Set-up complete. Finding weights for {gTX.shape[0]} units,  with {gTX.shape[1]} constraints:\n" + "-"*35)

        weights = self._eb(tr_total, gTX, base_weight)

        return {'converged': self.converged, 'maxdiff': self.maxdiff, 'w':weights.flatten()}


    def _eb(self, tr_total, co_x, base_weight):
        self.converged = False
        for iter in range(self.max_iterations):
            weights_temp = np.exp(co_x.dot(self.coefs)) #(n, 1)
            weights_ebal = np.multiply(weights_temp, base_weight).reshape(1, -1) #(n, 1)
            co_x_agg  = weights_ebal.reshape(1, -1).dot(co_x).reshape(-1, 1) #(p, )
            gradient  = co_x_agg - tr_total

            self.maxdiff = max(np.absolute(gradient))
            if self.maxdiff < self.constraint_tolerance:
                self.converged = True
                if self.print_level >= 0:
                    print("algorithm has converged, final loss = " + str(self.maxdiff))
                break
            hessian = co_x.T.dot((co_x * weights_ebal.reshape(-1, 1)))
            self.Coefs = self.coefs.copy()
            newton = np.linalg.solve(hessian, gradient)
            self.coefs -= newton * self.lr
            loss_new = self._line_searcher(ss=0, newton=newton, base_weight=base_weight, co_x=co_x, tr_total=tr_total, coefs=self.coefs)
            loss_old = self._line_searcher(ss=0, newton=newton, base_weight=base_weight, co_x=co_x, tr_total=tr_total, coefs=self.Coefs)

            if iter % 10==0 and self.print_level>=0:
                print("iteration = " + str(iter) + ", loss = " + str(loss_old))

            if loss_old <= loss_new:
                ss_min = minimize_scalar(self._line_searcher, bounds=(.0001, self.lr), args=(newton, base_weight, co_x, tr_total, self.Coefs), method='bounded')
                self.coefs = self.Coefs - ss_min.x*newton

        if self.converged == False:
            if self.print_level >= 0:
                print("algorithm did not converge, final loss = " + str(self.maxdiff))

        return weights_ebal


    def _line_searcher(self, ss, newton, base_weight, co_x, tr_total, coefs):
        weights_temp = np.exp(co_x.dot((coefs - newton*ss)))
        weights_temp = np.multiply(weights_temp, base_weight).reshape(1, -1)
        co_x_agg  = weights_temp.reshape(1, -1).dot(co_x).reshape(-1, 1) #(p, )
        gradient  = co_x_agg - tr_total
        return max(np.absolute(gradient))


    def check_balance(self, X, Treatment, weights):
        weights = weights.copy()
        weights = weights/np.sum(weights)
        Treatment = np.asarray(Treatment).flatten()

        # Convert numpy array to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f'X{i}' for i in range(np.asarray(X).shape[1])])

        types = np.array([self._check_binary(X[x]) for x in X])
        col_names = np.array(list(X.columns.values))
        stds = np.std(X, axis=0)
        to_keep = np.std(X, axis=0)!=0
        types = types[to_keep]
        stds = stds[to_keep]
        col_drop = col_names[to_keep==False]
        col_names = col_names[to_keep]
        X = np.asarray(X)[:,to_keep]

        before_corr = np.round(np.array([pearsonr(Treatment, x)[0] for x in X.T]), 2)
        after = [pearsonr(Treatment, (weights * x).reshape(-1,)) for x in X.T]
        after_corr = np.round(np.array([a[0] for a in after]), 2)
        after_pvalue = np.round(np.array([a[1] for a in after]), 2)

        out = {"Types": types, "Before_weighting_corr": before_corr, "After_weighting_corr": after_corr, "After_weighting_pvalue": after_pvalue}
        result_df = pd.DataFrame(data=out, index=col_names)
        if self.print_level >= 0:
            print(result_df.to_string())
            if len(col_drop) > 0:
                print(f"\n*Note: Columns {col_drop} were dropped because their standard deviations are 0")
        return result_df


    def _check_binary(self, x):
        if len(set(x))==2:
            return "binary"
        else:
            return "cont"
