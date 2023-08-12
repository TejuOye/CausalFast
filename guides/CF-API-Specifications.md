# CausalFast API: Specifications for use
### CausalFast is an API for Causal Inference over the DoWhy Library
```diff
Table of Contents:

1.   The CausalFast Approach
2.   Core Functions
  2-1.   DAG Maker
  2-2.   Causal Model
  2-3.   Simulator
3.   Tutorials & Notebooks
```
<br>

### 1: The Causal Fast Approach<br>
As of version 0.10 DoWhy's functionality is organized under four official APIs: Causal Inference API, GCM / Functional API, Lightweight Pandas API, and the causal prediction API. CausalFast is itself an API layer over the DoWhy Causal Inference API. 

### 2: Core Functions<br>
##### 2-1 - DAG Maker<br>
##### 2-2 - Causal Model<br>
##### 2-3 - Simulator<br>
Main Function Call:<br>
Instrumental Variable Estimand:
```
simulator(causalmodel='causalmodel',identifier='default',method_name='default',method_params='default',unit='default',full_output=True,refute=True):
```
Estimators: (method_name):<br>
Instrumental Variable Estimand:
```
method_name= iv.instrumental_variable
```
<br>
Frontdoor Estimand:

```
method_name= frontdoor.two_stage_regression
```
<br>
Backdoor Estimand:

```
method_name= backdoor.linear_regression

method_name= backdoor.generalized_linear_model
method_params= {'num_null_simulations':10,
                            'num_simulations':10,
                            'num_quantiles_to_discretize_cont_cols':10,
                            'fit_method': "statsmodels",
                            'glm_family': sm.families.Binomial(),
                            'need_conditional_estimates':False}

method_name= backdoor.econml.dr.LinearDRLearner
method_params= {"init_params":{
                            'model_propensity': LogisticRegressionCV(cv=3, solver='lbfgs', multi_class='auto')},
                            "fit_params":{}}

method_name= backdoor.econml.dml.DML
method_params= {'init_params':{
                            'model_y':GradientBoostingRegressor(),
                            'model_t': GradientBoostingRegressor(),
                            'model_final': LassoCV(fit_intercept=False),
                            'featurizer':PolynomialFeatures(degree=1, include_bias=True)},
                            'fit_params':{'inference': BootstrapInference(n_bootstrap_samples=20, n_jobs=-1)}}

method_name= backdoor.propensity_score_stratification
method_name= backdoor.propensity_score_weighting
method_name= backdoor.propensity_score_matching
```
