import numpy as np
from sklearn.linear_model import BayesianRidge,Ridge
from scipy.linalg import pinvh, eigh
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils._param_validation import Interval 
from numbers import Real, Integral
from scipy.stats import qmc 


class DeterministicBayesianRidge(BayesianRidge):
    """
    A deterministic version of BayesianRidge that suppresses aleatoric uncertainty.

    This class inherits from sklearn's BayesianRidge and modifies the fit method
    to set the noise precision to infinity, effectively eliminating the aleatoric uncertainty in predictions. 

    To ensure compatibility with BayesianRidge the alpha parameters are retained, but they are not used.

    Parameters
    ----------
    max_iter : int, default=300
        Maximum number of iterations.
    tol : float, default=1.0e-3
        Stop the algorithm if w has converged.
    alpha_1 : float, default=1.0e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior over the alpha parameter.
    alpha_2 : float, default=1.0e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the alpha parameter.
    lambda_1 : float, default=1.0e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior over the lambda parameter.
    lambda_2 : float, default=1.0e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the lambda parameter.
    alpha_init : float, default=None
        Initial value for alpha (precision of the noise).
    lambda_init : float, default=None
        Initial value for lambda (precision of the weights).
    compute_score : bool, default=False
        If True, compute the objective function at each step of the model.
    fit_intercept : bool, default=False
        Whether to calculate the intercept for this model.
    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.
    verbose : bool, default=False
        Verbose mode when fitting the model.

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        Coefficients of the regression model (mean of distribution).
    intercept_ : float
        Independent term in decision function. Set to 0.0 if
        `fit_intercept = False`.
    alpha_ : float
        Estimated precision of the noise, set to np.inf.
    lambda_ : float
        Estimated precision of the weights.
    sigma_ : array-like of shape (n_features, n_features)
        Estimated variance-covariance matrix of the weights.
    scores_ : list
        If computed_score is True, value of the objective function (to be
        maximized).

    Notes
    -----
    This implementation sets the noise precision (alpha_) to infinity after fitting,
    which results in deterministic predictions without aleatoric uncertainty.
    """
    _parameter_constraints: dict = {
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "alpha_1": [Interval(Real, 0, None, closed="left")],
        "alpha_2": [Interval(Real, 0, None, closed="left")],
        "lambda_1": [Interval(Real, 0, None, closed="left")],
        "lambda_2": [Interval(Real, 0, None, closed="left")],
        "alpha_init": [None, Interval(Real, 0, None, closed="left")],
        "lambda_init": [None, Interval(Real, 0, None, closed="left")],
        "compute_score": ["boolean"],
        "fit_intercept": ["boolean"],
        "copy_X": ["boolean"],
        "verbose": ["verbose"],
    }
    def __init__(
        self,
        *,
        max_iter=300,
        tol=1.0e-3,
        alpha_1=1.0e-6,
        alpha_2=1.0e-6,
        lambda_1=1.0e-6,
        lambda_2=1.0e-6,
        alpha_init=None,
        lambda_init=None,
        compute_score=False,
        fit_intercept=False,
        copy_X=True,
        verbose=False,
    ):
        super().__init__(
            max_iter=max_iter,
            tol=tol,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            alpha_init=alpha_init,
            lambda_init=lambda_init,
            compute_score=compute_score,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            verbose=verbose,
        )
        if self._check_alpha_values():
            print("Warning: alpha values have been specified, but DeterministicBayesianRidge will ignore them.")
    
    def _check_alpha_values(self):
        """
        Check if alpha values are different from their default values.
        
        Returns
        -------
        bool
            True if any alpha value is different from its default, False otherwise.
        """
        default_alphas = {
            'alpha_1': 1.0e-6,
            'alpha_2': 1.0e-6,
            'alpha_init': None
        }
        
        return any(getattr(self, attr) != default for attr, default in default_alphas.items())

    def fit(self, X, y, sample_weight=None):
        """
        Fit the Deterministic Bayesian Ridge Regression model.

        This method fits the model to the given training data. It overrides the
        parent class's fit method to ensure deterministic behavior by setting
        the noise precision to np.inf after fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If given a float, every sample
            will have the same weight.

        Returns
        -------
        self : object
            Returns the instance itself.

        Notes
        -----
        This method first calls the parent class's fit method to perform the
        initial fitting process. Then, it sets the noise precision (alpha_) to
        np.inf to suppress aleatoric uncertainty, ensuring deterministic
        predictions.
        """
        # Call the parent class's fit method
        super().fit(X, y, sample_weight)
        # Set the noise precision to infinity to suppress aleatoric uncertainty
        self.alpha_ = np.inf
        return self


class POPSRegressor(DeterministicBayesianRidge):
    """
    POPSRegressor (Pointwise Optimal Parameter Sets Regressor)

    This class extends the DeterministicBayesianRidge to implement the POPS algorithm for regression tasks. It performs probabilistic optimization of predictive subspaces, allowing for efficient handling of high-dimensional data. [1]

    Parameters
    ----------
    max_iter : int, default=300
        Maximum number of iterations.
    tol : float, default=1e-3
        Stop the algorithm if w has converged.
    alpha_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior over the alpha parameter.
    alpha_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the alpha parameter.
    lambda_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior over the lambda parameter.
    lambda_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the lambda parameter.
    alpha_init : float, default=None
        Initial value for alpha (precision of the noise).
    lambda_init : float, default=None
        Initial value for lambda (precision of the weights).
    compute_score : bool, default=False
        If True, compute the objective function at each step of the model.
    fit_intercept : bool, default=False
        Whether to calculate the intercept for this model.
    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.
    verbose : bool, default=False
        Verbose mode when fitting the model.
    mode_threshold : float, default=1e-3
        Threshold for determining the mode of the posterior distribution.
    resample_density : float, default=1.0
        Density of resampling for the POPS algorithm- number of hypercube per training point. Default is the greater of 0.5 or 100/n_samples.
    resampling_method : str, default='uniform'
        Method of resampling for the POPS algorithm. 
        must be one of 'sobol', 'latin', 'halton', 'grid', or 'uniform'.
    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        Coefficients of the regression model (mean of distribution).
    intercept_ : float
        Independent term in decision function. Set to 0.0 if
        `fit_intercept = False`.
    alpha_ : float
        Estimated precision of the noise.
    lambda_ : float
        Estimated precision of the weights.
    sigma_ : array-like of shape (n_features, n_features)
        Estimated variance-covariance matrix of the weights.
    scores_ : list
        If computed, value of the objective function (to be maximized).

    Notes
    -----
    The POPS algorithm extends Bayesian Ridge Regression by incorporating
    probabilistic optimization of predictive subspaces, which can lead to
    improved performance in high-dimensional settings.

    References
    ----------
    .. [1] Swinburne, T.D and Perez, D (2024). 
           Parameter uncertainties for imperfect surrogate models in the low-noise regime, arXiv:2402.01810v3
    """
    _parameter_constraints: dict = {
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "alpha_1": [Interval(Real, 0, None, closed="left")],
        "alpha_2": [Interval(Real, 0, None, closed="left")],
        "lambda_1": [Interval(Real, 0, None, closed="left")],
        "lambda_2": [Interval(Real, 0, None, closed="left")],
        "alpha_init": [None, Interval(Real, 0, None, closed="left")],
        "lambda_init": [None, Interval(Real, 0, None, closed="left")],
        "compute_score": ["boolean"],
        "fit_intercept": ["boolean"],
        "copy_X": ["boolean"],
        "verbose": ["verbose"],
    }
    def __init__(
        self,
        *,
        max_iter=300,
        tol=1.0e-3,
        alpha_1=1.0e-6,
        alpha_2=1.0e-6,
        lambda_1=1.0e-6,
        lambda_2=1.0e-6,
        alpha_init=None,
        lambda_init=None,
        compute_score=False,
        fit_intercept=False,
        copy_X=True,
        verbose=False,
        mode_threshold=1.0e-8,
        resample_density=1.0,
        resampling_method='uniform',
    ):
        super().__init__(
            max_iter=max_iter,
            tol=tol,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            alpha_init=alpha_init,
            lambda_init=lambda_init,
            compute_score=compute_score,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            verbose=verbose,
        )
        self.mode_threshold = mode_threshold
        self.fit_intercept_flag = False
        self.resample_density = resample_density
        self.resampling_method = resampling_method
        
        if self.fit_intercept:
            print("Warning: fit_intercept is set to False for POPS regression. A constant feature will be added to the design matrix.")
            self.fit_intercept_flag = True
            self.fit_intercept = False
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit the POPS regression model.

        This method extends the fit method of BayesianRidge to include POPS-specific
        computations, such as calculating leverage scores, pointwise corrections,
        and determining the hypercube support.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns:
        --------
        self : object
            Returns the instance itself.
        """
        if self.fit_intercept_flag:
            print("Warning: fit_intercept is set to False for POPS regression. Adding a constant feature for regression.")
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        super().fit(X, y, sample_weight)
        
        # enforce deterministic data
        self.alpha_ = np.inf

        n_features = X.shape[1]
        n_samples = X.shape[0]

        # Prior is lambda_ from BayesianRidge
        prior = self.lambda_ * np.eye(n_features) / n_samples
        
        # Prior ensures that the design matrix is invertible
        inverse_design_matrix = pinvh(X.T @ X / n_samples + prior)

        # Calculate leverage and pointwise fits
        errors = y - X @ self.coef_  # errors to mean prediction
        X_inverse_design_matrix = X @ inverse_design_matrix
        
        self.leverage_scores = (X_inverse_design_matrix * X).sum(1)
        
        self.pointwise_correction = X_inverse_design_matrix \
            * (errors/self.leverage_scores)[:,None]
        
        # Determine bounding hypercube from pointwise fits
        self.hypercube_support, self.hypercube_bounds = \
            self._hypercube_fit(self.pointwise_correction)
        
        self.hypercube_samples = self._resample_hypercube()

        self.hypercube_sigma = self.hypercube_samples@self.hypercube_samples.T
        self.hypercube_sigma /= self.hypercube_samples.shape[1]

    def _hypercube_fit(self,pointwise_correction):
        """
        Fit a hypercube to the pointwise corrections.

        This method calculates the principal components of the pointwise corrections
        and determines the bounding box (hypercube) in the space of these components.

        Parameters:
        -----------
        pointwise_correction : numpy.ndarray
            Array of pointwise corrections, shape (n_samples, n_features).

        Returns:
        --------
        projections : numpy.ndarray
            The principal component vectors that define the hypercube space.
        bounds : numpy.ndarray
            The min and max bounds of the hypercube along each principal component.

        Notes:
        ------
        The method performs the following steps:
        1. Compute the eigendecomposition of the covariance matrix of pointwise corrections.
        2. Select principal components based on the mode_threshold.
        3. Project the pointwise corrections onto these components.
        4. Determine the bounding box (hypercube) in this projected space.

        The resulting hypercube represents the uncertainty in the parameter estimates,
        which can be used for subsequent resampling and uncertainty quantification.
        """
        
        e_values, e_vectors = eigh(pointwise_correction.T @pointwise_correction)
        
        mask = e_values > self.mode_threshold * e_values.max()
        e_vectors = e_vectors[:,mask]
        e_values = e_values[mask]
        
        projections = e_vectors.copy()
        projected = pointwise_correction @ projections
        bounds = np.array([projected.min(0),projected.max(0)])
        
        return projections, bounds
    
    def _resample_hypercube(self,size=None):
        """
        Resample points from the hypercube.

        This method generates new samples from the hypercube defined by the
        bounding box of the pointwise corrections. The sampling is uniform
        within the hypercube.

        Parameters:
        -----------
        size : int, optional
            The number of samples to generate. If None, the number of samples
            is determined by self.resample_density * self.leverage_scores.size.

        Returns:
        --------
        numpy.ndarray
            An array of shape (n_features, n_samples) containing the resampled
            points in the feature space.

        Notes:
        ------
        The resampling process involves the following steps:
        1. Generate uniform random numbers between 0 and 1.
        2. Scale these numbers to the range of the hypercube bounds.
        3. Project the scaled points back to the original feature space using
           the hypercube support vectors.

        This method is used to generate new possible parameter values within
        the uncertainty bounds of the model, which can be used for uncertainty
        quantification in predictions.
        """
        
        low = self.hypercube_bounds[0]
        high = self.hypercube_bounds[1]
        if size is None:
            n_resample = int(self.resample_density*self.leverage_scores.size)
        else:
            n_resample = size
        n_resample = max(n_resample,100)
        
        # Sobol sequence
        if self.resampling_method == 'latin':
            sampler = qmc.LatinHypercube(d=low.size)
            samples = sampler.random(n_resample).T 
        elif self.resampling_method == 'sobol':
            sampler = qmc.Sobol(d=low.size)
            n_resample = 2**int(np.log(n_resample)/np.log(2.0))
            samples = sampler.random(n_resample).T 
        elif self.resampling_method == 'grid':
            samples = np.linspace(0,1,n_resample).T
        elif self.resampling_method == 'halton':
            sampler = qmc.Halton(d=low.size)
            samples = sampler.random(n_resample).T 
        elif self.resampling_method == 'uniform':
            samples = np.random.uniform(size=(low.size,n_resample))
        samples = low[:,None] + (high-low)[:,None]*samples
        return self.hypercube_support@samples

    def predict(self,X,return_bounds=False,resample=False,return_epistemic_std=False):
        """
        Make predictions using the POPS model.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples for prediction.
        return_bounds : bool, default=False
            If True, return the min and max bounds of the prediction.
        resample : bool, default=False
            If True, resample the hypercube before prediction.
        return_epistemic_std : bool, default=False
            If True, return the epistemic standard deviation.

        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            The predicted mean values.
        y_std : array-like of shape (n_samples,)
            The predicted standard deviation (uncertainty) for each prediction.
        y_max : array-like of shape (n_samples,), optional
            The upper bound of the prediction interval. Only returned if return_bounds is True.
        y_min : array-like of shape (n_samples,), optional
            The lower bound of the prediction interval. Only returned if return_bounds is True.
        """
        # DeterministicBayesianRidge suppresses aleatoric uncertainty
        y_pred, y_epistemic_std = super().predict(X,return_std=True)
        
        # Combine misspecification and epistemic uncertainty
        y_misspecification_var = (X@self.hypercube_sigma * X).sum(1)
        #y_std = np.sqrt((X@self.hypercube_samples).var(1) + y_epistemic_std**2)
        y_std = np.sqrt(y_misspecification_var + y_epistemic_std**2)
        
        if resample:
            self.hypercube_samples = self._resample_hypercube()
        res = [y_pred, y_std]
        
        if return_bounds:
            y_max = (X@self.hypercube_samples).max(1) + y_pred
            y_min = (X@self.hypercube_samples).min(1) + y_pred
            res += [y_max, y_min]
        if return_epistemic_std:
            res += [y_epistemic_std]
        
        return tuple(res)