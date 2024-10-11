# POPSRegression
Regression scheme from the paper 

*Parameter uncertainties for imperfect surrogate models in the low-noise regime*

TD Swinburne and D Perez, [arXiv 2024](https://arxiv.org/abs/2402.01810v3)

```bibtex
@misc{swinburne2024,
      title={Parameter uncertainties for imperfect surrogate models in the low-noise regime}, 
      author={Thomas D Swinburne and Danny Perez},
      year={2024},
      eprint={2402.01810},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2402.01810v3}, 
}
```

## Installation
There will be a PR on `scikit-learn` soon, but in the meantime
```bash
pip install POPSRegression
```

## What is POPSRegression?

Bayesian regression for low-noise data (vanishing aleatoric uncertainty). 
}
Fits the weights of a regression model using BayesianRidge, then estimates weight uncertainties (`sigma_` in `BayesianRidge`) accounting for model misspecification using the POPS (Pointwise Optimal Parameter Sets) algorithm [1]. The `alpha_` attribute which estimates aleatoric uncertainty is not used for predictions as correctly it should be assumed negligable.

Bayesian regression is often used in computational science to fit the weights of a surrogate model which approximates some complex calcualtion. 
In many important cases the target calcualtion is near-deterministic, or low-noise, meaning the true data has vanishing aleatoric uncertainty. However, there can be large misspecification uncertainty, i.e. the model weights are instrinsically uncertain as the model is unable to exactly match training data. 

Existing Bayesian regression schemes based on loss minimization can only estimate epistemic and aleatoric uncertainties. In the low-noise limit, 
weight uncertainties (`sigma_` in `BayesianRidge`) are significantly underestimated as they only account for epistemic uncertainties which decay with increasing data. Predictions then assume any additional error is due to an aleatoric uncertainty (`alpha_` in `BayesianRidge`), which is erroneous in a low-noise setting. This has significant implications on how uncertainty is propagated using weight uncertainties. 

## Example usage
Here, usage follows `sklearn.linear_model`, inheriting `BayesianRidge`

After running `BayesianRidge.fit(..)`, the `alpha_` attribute is not used for predictions.

The `sigma_` matrix still contains epistemic weight uncertainties, whilst `misspecification_sigma_` contains the POPS uncertainties. 

```python

from POPSRegression import POPSRegression

X_train,X_test,y_train,y_test = ...

# Sobol resampling of hypercube with 1.0 samples / training point
model = POPSRegression(resampling_method='sobol',resample_density=1.)

# fit the model, sample POPS hypercube
model.fit(X_train,y_train)

# Return mean and hypercube std
y_pred, y_std = model.predict(X_test,return_std=True)

# can also return max/min 
y_pred, y_std, y_max, y_min = model.predict(X_test,return_bounds=True)

# returns std by default
y_pred, y_std = model.predict(X_test)

# can also return max/min 
y_pred, y_std, y_max, y_min = model.predict(X_test,return_bounds=True)

# can also return the epistemic uncertainty (descreases as 1/sqrt(n_samples))
y_pred, y_std, y_max, y_min, y_std_epistmic = model.predict(X_test,return_bounds=True,return_epistemic_std=True)
```

As can be seen, the final error bars give very good coverage of the test output

Extreme low-dimensional case, fitting N data points to a quartic polynomial (P=5 parameters) to some complex oscillatory function

Green: two sigma of `sigma_` weight uncertainty from Bayesian Regression (i.e. without `alpha_` term for aleatoric error)

Orange: two sigma of `sigma_` and `misspecification_sigma_` posterior from POPS Regression

<img src="https://github.com/tomswinburne/POPS-Regression/blob/main/example_image.png?raw=true"></img>
