import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
import statsmodels.api as sm

class ebal_bin:
    """
    Implementation of Entropy Balancing for binary treatment.

    Author: Eddie Yang, based on work of Hainmueller (2012) and Xu & Yang (2021)

    Params:
        coefs: Lagrangian multipliers, refer to Hainmueller (2012)
        max_iterations: maximum number of iterations to find the solution weights, default 500
        constraint_tolerance: tolerance level for covariate difference between the treatment
            and control group, default 1e-4
        print_level: level of details to print out, default 0. Set to -1 to suppress all output.
        lr: step size, default 1. Increase to make the algorithm converge faster
            (at the risk of exploding gradient)
        PCA: whether to apply PCA to covariates for dimensionality reduction, default True
        effect: type of treatment effect to estimate ('ATT', 'ATC', 'ATE'), default 'ATT'

    Output:
        converged: boolean, whether the algorithm converged
        maxdiff: maximum covariate difference between treatment and control groups
        w: solution weights for the control units
        wls: output from weighted OLS regression
    """

    def __init__(self,
        coefs = None,
        max_iterations = 500,
        constraint_tolerance = 0.0001,
        print_level=0,
        lr=1,
        PCA=True,
        effect="ATT"):

        self._initial_coefs = coefs
        self.coefs = coefs
        self.max_iterations = max_iterations
        self.constraint_tolerance = constraint_tolerance
        self.print_level = print_level
        self.lr = lr
        self.PCA = PCA
        self.effect = effect
        if self.effect not in ['ATT', 'ATC', 'ATE']:
            raise ValueError("Effect must be one of ATT, ATC, or ATE")


    def ebalance(
        self,
        Treatment,
        X,
        Y,
        base_weight=None):

        self.coefs = self._initial_coefs

        Treatment = np.asarray(Treatment)
        if self.effect == "ATC":
            Treatment = np.abs(Treatment-1) # revert treatment indicator so that the treatment group gets reweighted instead of the control group
            if self.print_level >= 0:
                print("Estimating ATC:\n" + "-"*35)
        elif self.effect == "ATE":
            if self.print_level >= 0:
                print("Estimating ATE:\n" + "-"*35)
        else:
            if self.print_level >= 0:
                print("Estimating ATT:\n" + "-"*35)

        X = np.asarray(X)

        if np.isnan(Treatment).any():
            raise ValueError("Treatment contains missing data")

        if np.var(Treatment) == 0:
            raise ValueError("Treatment indicator ('Treatment') must contain both treatment and control observations")

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
                print("PCA on X successful \n" + "-"*35)

        # set up elements
        ntreated  = np.sum(Treatment==1)
        ncontrols = np.sum(Treatment==0)

        if base_weight is None:
            base_weight = np.ones(ncontrols)
        elif not len(base_weight) == ncontrols:
            raise ValueError("length(base_weight) != number of controls, sum(Treatment==0)")
        else:
            base_weight = np.asarray(base_weight)


        co_x = X[Treatment==0,:]
        co_x = np.column_stack((np.ones(ncontrols),co_x))

        if not np.linalg.matrix_rank(co_x) == co_x.shape[1]:
            raise ValueError("collinearity in covariate matrix for controls (remove collinear covariates)")

        if self.effect == "ATE":
            tr_total = X.mean(axis=0)
        else:
            tr_total = X[Treatment==1,:].mean(axis=0)

        tr_total = np.insert(tr_total, 0, 1, axis=0)
        if self.coefs is None:
            self.coefs = np.insert(np.zeros(co_x.shape[1]-1), 0, np.log(1), axis=0)
        else:
            self.coefs = np.asarray(self.coefs)

        if not self.coefs.shape[0]==co_x.shape[1]:
            raise ValueError("coefs needs to have same length as number of covariates plus one")

        if self.print_level >= 0:
            print(f"Set-up complete, balancing {co_x.shape[1]-1} covariates and 1 intercept\n")
            if self.effect != "ATE":
                print(f"Start finding weights for {ncontrols} units:\n" + "-"*35)

        if self.effect == "ATE":
            if self.print_level >= 0:
                print(f"Start finding weights for control group with {ncontrols} units:\n" + "-"*35)
            weights = self._eb(tr_total, co_x, base_weight) #control group
            w = np.ones(X.shape[0])
            w[Treatment==0] = weights
            # treatment group
            if self.print_level >= 0:
                print(f"Start finding weights for treatment group with {ntreated} units:\n" + "-"*35)
            base_weight = np.ones(ntreated)
            co_x = X[Treatment==1,:]
            co_x = np.column_stack((np.ones(ntreated),co_x))
            weights = self._eb(tr_total, co_x, base_weight)
            w[Treatment==1] = weights
        else:
            weights = self._eb(tr_total, co_x, base_weight)
            w = np.ones(X.shape[0])/ntreated
            w[Treatment==0] = weights

        if self.effect == "ATC":
            Treatment = np.abs(Treatment-1) # need to revert the treatment indicator to estimate atc

        wls_results = self._get_wls_results(se_type="HC2", Treatment=Treatment, Y=Y, weights=w)

        return {'converged': self.converged, 'maxdiff': self.maxdiff, 'w':w, "wls":wls_results}


    def _eb(self, tr_total, co_x, base_weight):
        self.converged = False
        for iter in range(self.max_iterations):
            weights_temp = np.exp(co_x.dot(self.coefs)) #(n, )
            weights_ebal = np.multiply(weights_temp, base_weight) #(n, )
            co_x_agg  = weights_ebal.dot(co_x) #(p, )
            gradient  = co_x_agg - tr_total
            self.maxdiff = max(np.absolute(gradient))
            if self.maxdiff < self.constraint_tolerance:
                self.converged = True
                if self.print_level >= 0:
                    print("algorithm has converged, final loss = " + str(self.maxdiff))
                break
            hessian = co_x.T.dot((co_x*weights_ebal[:, np.newaxis]))
            self.Coefs = self.coefs.copy()
            newton = np.linalg.solve(hessian, gradient)
            self.coefs -= newton*self.lr
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
        weights_temp = np.exp(co_x.dot((coefs-ss*newton)))
        weights_temp = np.multiply(weights_temp, base_weight)
        co_x_agg  = weights_temp.dot(co_x) #(p, )
        gradient  = co_x_agg - tr_total
        return max(np.absolute(gradient))


    def _get_wls_results(self, se_type, Treatment, Y, weights):
        t = sm.add_constant(Treatment.reshape(-1,1)) # intercept + treatment
        t = pd.DataFrame(data=t, columns=["const", "treatment"])
        mod_wls = sm.WLS(Y, t, weights=weights)
        res_wls = mod_wls.fit()
        return res_wls.get_robustcov_results(cov_type=se_type)


    def check_balance(self, X, Treatment, weights):
        weights = weights.copy()
        Treatment = np.asarray(Treatment)
        weights[Treatment==1] = weights[Treatment==1]/np.sum(weights[Treatment==1])
        weights[Treatment==0] = weights[Treatment==0]/np.sum(weights[Treatment==0]) # normalize weights
        if self.effect == "ATC":
            Treatment = np.abs(Treatment-1)

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
        if self.effect == "ATE":
            tr_mean = np.mean(X, axis=0)
        else:
            tr_mean = np.dot(weights[Treatment==1], X[Treatment==1,:])
        before = np.round((tr_mean - np.mean(X[Treatment==0,:], axis=0))/stds, 2)
        after = np.round(tr_mean - np.dot(weights[Treatment==0], X[Treatment==0,:]), 2)
        out = {"Types": types, "Before_weighting": before, "After_weighting": after}
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
