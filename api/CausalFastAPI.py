import logging
import typing
import warnings
from itertools import combinations

import matplotlib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import networkx as nx
from networkx import path_graph

import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingRegressor

from dowhy import CausalModel
from econml.inference import BootstrapInference

class functions:
    def __init__(
        self,
        data,
        treatment,
        outcome,
        graph=None,
        common_causes=None,
        instruments=None,
        effect_modifiers=None,
        estimand_type="nonparametric-ate",
        proceed_when_unidentifiable=False,
        missing_nodes_as_confounders=False,
        identify_vars=False,
        **kwargs,
    ):
        self._data = data
        self._treatment = parse_state(treatment)
        self._outcome = parse_state(outcome)
        self._effect_modifiers = parse_state(effect_modifiers)
        self._estimand_type = estimand_type
        self._proceed_when_unidentifiable = proceed_when_unidentifiable
        self._missing_nodes_as_confounders = missing_nodes_as_confounders
        self.logger = logging.getLogger(__name__)
        self._estimator_cache = {}

    def simulator(causalmodel='causalmodel',output='default',identifier='default',estimator='default',estimatorparams='default',unit='default'):
        idparam = identifier  #for backdoor:#exhaustive-search #minimal-adjustment #maximal-adjustment'
        estparam = estimator #for backdoor.linear_regression frontdoor.two_stage_regression iv.instrumental_variable
                            #Alt estimators: backdoor.generalized_linear_model, 
        estmethod = estimatorparams  #parameters to configure a specific estimation method
        targetunit = unit #Units to estimate: ATE, ATC, ATT
        if str(type(causalmodel)) == '<class \'dowhy.causal_model.CausalModel\'>':
            cmodel = causalmodel

        else:
            print('CausalFast simulator():')
            print('____________________________________________')
            print('')
            print('Note:')
            print('You have not provided a valid CausalModel object')
            print('Create a valid CausalModel using causalfast.makegraph()')
            print('Return this object and use it as a parameter in causalfast.simulator()')
            print('')
            print('The CausalFast simulator will attempt to automatically run DoWhy analysis.')
            print('By default, the CausalFast simulator will identify the correct estimand and assign an estimator')
            print('Syntax: causalfast.simulator(causalmodel=causalmodelobj[0])')
            print('Syntax: causalfast.simulator(causalmodel=\'causalmodel\',output=\'default\',identifier=\'default\',estimator=\'default\'')
            print('')
            print('Available DoWhy Identifier algorithms:')
            print('   minimal-adjustment, maximal-adjustment, exhaustive-search, default')
            print('')
            print('Automatic DoWhy Estimators:')
            print('   Backdoor:  backdoor.linear_regression')
            print('   Frontdoor: frontdoor.two_stage_regression')
            print('   Instrumental Variable: iv.instrumental_variable')
            print('')
            print('Other estimators can be used. Required additional parameters are automatically generated')
            print('   Backdoor + binary outcome (logistic):  backdoor.generalized_linear_model')
            print('   Backdoor + binary outcome (logistic):  backdoor.econml.dr.LinearDRLearner')
            print('   Backdoor + MachineLearning (DoubleML): backdoor.econml.dml.DML')
            print('')
            print('Propensity Scores require both binary treatment and observed common causes')
            print('   Backdoor + binary treatment (Propensity Score ATT): backdoor.propensity_score_stratification')
            print('   Backdoor + binary treatment (Propensity Score ATC): backdoor.propensity_score_matching')
            print('   Backdoor + binary treatment (Propensity Score ATE): backdoor.propensity_score_weighting')
            return

        if output != 'full' and output != 'limited':
            #Identify the causal effect
            identified_estimand = cmodel.identify_effect(proceed_when_unidentifiable=True, method_name=idparam)
            print('Simulator Mode (Default DoWhy) Parameters: ')
            print('Identification Using: identifier=\'',idparam,'\'')
            print('Estimation Using: estimator=\'',estparam,'\'')
            print('')
            estimandcheck = str(identified_estimand)
            estimandcheck2 = estimandcheck.split("Estimand expression:")
            estimandcheck = estimandcheck.replace(estimandcheck2[1], '')
            estimandcheck = estimandcheck[:-20]
            estimandcheck = estimandcheck[-30:]
            estimandcheck = estimandcheck.split("Estimand name: ")
            estimandcheck = estimandcheck[1].strip()
            if estimandcheck == 'iv': 
                print('iv defaults to the \'iv.instrumental_variable\' estimation method.')
                print('DoWhy has other iv estimation methods: ')
                print('   Binary Instrument/Wald Estimator')
                print('   Two-stage least squares')
                print('   Regression discontinuity')
                print('')
                if estimator == 'default':
                    estparam = 'iv.instrumental_variable'
                    estmethod = None
                elif estimator != 'default' and estimatorparams == 'default':
                    estmethod = None
            elif estimandcheck == 'frontdoor': 
                print('DoWhy has one frontdoor estimation method: \'frontdoor.two_stage_regression\'')
                estparam = 'frontdoor.two_stage_regression'
                estmethod = None
            elif estimandcheck == 'backdoor': 
                print('backdoor defaults to the \'backdoor.linear_regression\' estimation method.')
                print('Other backdoor estimation methods include: ')
                print('Logistic:   backdoor.generalized_linear_model')
                print('Logistic:   backdoor.econml.dr.LinearDRLearner')
                print('DoubleML:   backdoor.econml.dml.DML')
                print('')
                print('Note: Propensity Scores Require a Binary Treatment Variable')
                print('Propensity: backdoor.propensity_score_stratification')
                print('Propensity: backdoor.propensity_score_matching')
                print('Propensity: backdoor.propensity_score_weighting')
                print('')
                if estimator == 'default':
                    estparam = 'backdoor.linear_regression'
                    estmethod = None
                elif estimator == 'backdoor.generalized_linear_model':
                    if estimatorparams == 'default':
                        estmethod = {'num_null_simulations':10,
                        'num_simulations':10,
                        'num_quantiles_to_discretize_cont_cols':10,
                        'fit_method': "statsmodels",
                        'glm_family': sm.families.Binomial(),
                        'need_conditional_estimates':False}
                    else:
                        estmethod = estimatorparams
                elif estimator == 'backdoor.econml.dr.LinearDRLearner':
                    if estimatorparams == 'default':
                        estmethod = {"init_params":{
                            'model_propensity': LogisticRegressionCV(cv=3, solver='lbfgs', multi_class='auto')},
                            "fit_params":{}}
                    else:
                        estmethod = estimatorparams
                elif estimator == 'backdoor.econml.dml.DML':
                    if estimatorparams == 'default':
                        estmethod = {'init_params':{
                            'model_y':GradientBoostingRegressor(),
                            'model_t': GradientBoostingRegressor(),
                            'model_final': LassoCV(fit_intercept=False),
                            'featurizer':PolynomialFeatures(degree=1, include_bias=True)},
                            'fit_params':{'inference': BootstrapInference(n_bootstrap_samples=20, n_jobs=-1)}}
                    else: 
                        estmethod = estimatorparams
                elif estimator == 'backdoor.propensity_score_stratification':
                    estparam = 'backdoor.propensity_score_stratification' 
                    targetunit = 'att'
                    print('Propensity Score Unit: ATT')
                    estmethod = None
                elif estimator == 'backdoor.propensity_score_matching':
                    estparam = 'backdoor.propensity_score_matching' 
                    targetunit = 'atc'
                    print('Propensity Score Unit: ATC')
                    estmethod = None
                elif estimator == 'backdoor.propensity_score_weighting':
                    estparam = 'backdoor.propensity_score_weighting' 
                    targetunit = 'ate'
                    print('Propensity Score Unit: ATE')
                    estmethod = None
                    estmethod = {"weighting_scheme":"ips_weight"}
                else:
                    targetunit = None
                    estmethod = None
            elif estimandcheck != 'iv' and estimandcheck != 'frontdoor' and estimandcheck != 'backdoor':
                print('Unable to identify DAG Estimand. Please input a valid DAG.')
                return

            print('Detected Estimator method_name (DoWhy): \'',estparam,'\'')
            print('Detected Estimator method_params (DoWhy): ',estmethod)
            print('')
            #Estimate
            estimate = cmodel.estimate_effect(identified_estimand,
            method_name=estparam, test_significance=True,method_params = estmethod, target_units = targetunit)
            print(estimate)
            try:
                print('Propensity Scores:        ',estimate.__dict__.get('params').get('propensity_scores'))
            except:
                print('Propensity Scores:      ')
            print('')
            #Refute
            if estimandcheck == 'iv':
                refute_results = cmodel.refute_estimate(identified_estimand, estimate,
                method_name="data_subset_refuter", subset_fraction=0.9)
                print(refute_results)
                refute_results = cmodel.refute_estimate(identified_estimand, estimate,
                method_name="placebo_treatment_refuter", placebo_type="permute")
                print(refute_results)
                refute_results = cmodel.refute_estimate(identified_estimand, estimate,
                method_name="bootstrap_refuter")
                print(refute_results)
            if estimandcheck != 'iv':
                refute_results = cmodel.refute_estimate(identified_estimand, estimate,
                method_name="random_common_cause")
                print(refute_results)
                refute_results = cmodel.refute_estimate(identified_estimand, estimate,
                method_name="data_subset_refuter", subset_fraction=0.9)
                print(refute_results)
                refute_results = cmodel.refute_estimate(identified_estimand, estimate,
                method_name="placebo_treatment_refuter", placebo_type="permute")
                print(refute_results)
                refute_results = cmodel.refute_estimate(identified_estimand, estimate,
                method_name="bootstrap_refuter")
                print(refute_results)

        elif output == 'limited':
            print('Simulator Mode (Limited) Parameters: ')
            print('Identification Using: identifier=\'',idparam,'\'')
            print('Estimation Using: estimator=\'',estparam,'\'')
            print('')
            #Identify the causal effect
            identified_estimand = cmodel.identify_effect(proceed_when_unidentifiable=True, method_name=idparam)
            estimandcheck = str(identified_estimand)
            estimandcheck2 = estimandcheck.split("Estimand expression:")
            estimandcheck = estimandcheck.replace(estimandcheck2[1], '')
            estimandcheck = estimandcheck[:-20]
            estimandcheck = estimandcheck[-30:]
            estimandcheck = estimandcheck.split("Estimand name: ")
            estimandcheck = estimandcheck[1].strip()
            if estimandcheck == 'iv': 
                print('iv defaults to the \'iv.instrumental_variable\' estimation method.')
                print('DoWhy has other iv estimation methods: ')
                print('   Binary Instrument/Wald Estimator')
                print('   Two-stage least squares')
                print('   Regression discontinuity')
                print('')
                if estimator == 'default':
                    estparam = 'iv.instrumental_variable'
                    estmethod = None
                elif estimator != 'default' and estimatorparams == 'default':
                    estmethod = None
            elif estimandcheck == 'frontdoor': 
                print('DoWhy has one frontdoor estimation method: \'frontdoor.two_stage_regression\'')
                estparam = 'frontdoor.two_stage_regression'
                estmethod = None
            elif estimandcheck == 'backdoor': 
                print('backdoor defaults to the \'backdoor.linear_regression\' estimation method.')
                print('Other backdoor estimation methods include: ')
                print('Logistic:   backdoor.generalized_linear_model')
                print('Logistic:   backdoor.econml.dr.LinearDRLearner')
                print('DoubleML:   backdoor.econml.dml.DML')
                print('')
                print('Note: Propensity Scores Require a Binary Treatment Variable')
                print('Propensity: backdoor.propensity_score_stratification')
                print('Propensity: backdoor.propensity_score_matching')
                print('Propensity: backdoor.propensity_score_weighting')
                print('')
                if estimator == 'default':
                    estparam = 'backdoor.linear_regression'
                    estmethod = None
                elif estimator == 'backdoor.generalized_linear_model':
                    if estimatorparams == 'default':
                        estmethod = {'num_null_simulations':10,
                        'num_simulations':10,
                        'num_quantiles_to_discretize_cont_cols':10,
                        'fit_method': "statsmodels",
                        'glm_family': sm.families.Binomial(),
                        'need_conditional_estimates':False}
                    else:
                        estmethod = estimatorparams
                elif estimator == 'backdoor.econml.dr.LinearDRLearner':
                    if estimatorparams == 'default':
                        estmethod = {"init_params":{
                            'model_propensity': LogisticRegressionCV(cv=3, solver='lbfgs', multi_class='auto')},
                            "fit_params":{}}
                    else:
                        estmethod = estimatorparams
                elif estimator == 'backdoor.econml.dml.DML':
                    if estimatorparams == 'default':
                        estmethod = {'init_params':{
                            'model_y':GradientBoostingRegressor(),
                            'model_t': GradientBoostingRegressor(),
                            'model_final': LassoCV(fit_intercept=False),
                            'featurizer':PolynomialFeatures(degree=1, include_bias=True)},
                            'fit_params':{'inference': BootstrapInference(n_bootstrap_samples=20, n_jobs=-1)}}
                    else: 
                        estmethod = estimatorparams
                elif estimator == 'backdoor.propensity_score_stratification':
                    estparam = 'backdoor.propensity_score_stratification' 
                    targetunit = 'att'
                    print('Propensity Score Unit: ATT')
                    estmethod = None
                elif estimator == 'backdoor.propensity_score_matching':
                    estparam = 'backdoor.propensity_score_matching' 
                    targetunit = 'atc'
                    print('Propensity Score Unit: ATC')
                    estmethod = None
                elif estimator == 'backdoor.propensity_score_weighting':
                    estparam = 'backdoor.propensity_score_weighting' 
                    targetunit = 'ate'
                    print('Propensity Score Unit: ATE')
                    estmethod = None
                    estmethod = {"weighting_scheme":"ips_weight"}
                else:
                    targetunit = None
                    estmethod = None
            elif estimandcheck != 'iv' and estimandcheck != 'frontdoor' and estimandcheck != 'backdoor':
                print('Unable to identify DAG Estimand. Please input a valid DAG.')
                return

            print('Detected Estimator method_name (DoWhy): \'',estparam,'\'')
            print('Detected Estimator method_params (DoWhy): ',estmethod)
            print('')
            print('Identify: ')
            print('==================================================')
            print('Treatment Variable:       ',identified_estimand.__dict__.get('treatment_variable'))
            print('Outcome Variable:         ',identified_estimand.__dict__.get('outcome_variable'))
            print('Backdoor Variabless:      ',identified_estimand.__dict__.get('backdoor_variables'))
            print('Instrumental Variables:   ',identified_estimand.__dict__.get('instrumental_variables'))
            print('Frontdoor Variables:      ',identified_estimand.__dict__.get('frontdoor_variables'))
            print('Mediator Variables:       ',identified_estimand.__dict__.get('mediator_variables'))
            print('First Stage Confounders:  ',identified_estimand.__dict__.get('mediation_first_stage_confounders'))
            print('Second Stage Confounders: ',identified_estimand.__dict__.get('mediation_second_stage_confounders'))
            print('')
            print(identified_estimand)

            #Estimate
            print('Estimate: ')
            print('==================================================')
            estimate = cmodel.estimate_effect(identified_estimand,
            method_name=estparam, test_significance=True,method_params = estmethod)
            print('Estimate Value:           ',estimate.__dict__.get('value'))
            print('Intercept:                ',estimate.__dict__.get('intercept'))
            print('Estimator:                ',estimate.__dict__.get('estimator'))
            print('Target Units:             ',estimate.__dict__.get('params').get('target_units'))
            print('Effect Modifiers:         ',estimate.__dict__.get('params').get('effect_modifiers'))
            try:
                print('Propensity Scores:        ',estimate.__dict__.get('params').get('propensity_scores'))
            except:
                print('Propensity Scores:      ')
            print('')

            #Refute
            print('Refute: ')
            print('==================================================')
            if estimandcheck == 'iv':
                refute_results = cmodel.refute_estimate(identified_estimand, estimate,
                method_name="data_subset_refuter", subset_fraction=0.9)
                print('Refuter 1: ')
                print('Refuter Type:             ',refute_results.__dict__.get('refutation_type'))
                print('Estimated Effect:         ',refute_results.__dict__.get('estimated_effect'))
                print('New Effect:               ',refute_results.__dict__.get('new_effect'))
                print('Statistical Significance: ',refute_results.__dict__.get('refutation_result').get('is_statistically_significant'))
                print('P-Value:                  ',refute_results.__dict__.get('refutation_result').get('p_value'))
                print('')
                refute_results = cmodel.refute_estimate(identified_estimand, estimate,
                method_name="placebo_treatment_refuter", placebo_type="permute")
                print('Refuter 2: ')
                print('Refuter Type:             ',refute_results.__dict__.get('refutation_type'))
                print('Estimated Effect:         ',refute_results.__dict__.get('estimated_effect'))
                print('New Effect:               ',refute_results.__dict__.get('new_effect'))
                print('Statistical Significance: ',refute_results.__dict__.get('refutation_result').get('is_statistically_significant'))
                print('P-Value:                  ',refute_results.__dict__.get('refutation_result').get('p_value'))
                print('')
                refute_results = cmodel.refute_estimate(identified_estimand, estimate,
                method_name="bootstrap_refuter")
                print('Refuter 3: ')
                print('Refuter Type:             ',refute_results.__dict__.get('refutation_type'))
                print('Estimated Effect:         ',refute_results.__dict__.get('estimated_effect'))
                print('New Effect:               ',refute_results.__dict__.get('new_effect'))
                print('Statistical Significance: ',refute_results.__dict__.get('refutation_result').get('is_statistically_significant'))
                print('P-Value:                  ',refute_results.__dict__.get('refutation_result').get('p_value'))
            elif estimandcheck != 'iv':
                refute_results=cmodel.refute_estimate(identified_estimand, estimate,
                method_name="random_common_cause")
                print('Refuter 1: ')
                print('Refuter Type:             ',refute_results.__dict__.get('refutation_type'))
                print('Estimated Effect:         ',refute_results.__dict__.get('estimated_effect'))
                print('New Effect:               ',refute_results.__dict__.get('new_effect'))
                print('Statistical Significance: ',refute_results.__dict__.get('refutation_result').get('is_statistically_significant'))
                print('P-Value:                  ',refute_results.__dict__.get('refutation_result').get('p_value'))
                print('')
                refute_results = cmodel.refute_estimate(identified_estimand, estimate,
                method_name="data_subset_refuter", subset_fraction=0.9)
                print('Refuter 2: ')
                print('Refuter Type:             ',refute_results.__dict__.get('refutation_type'))
                print('Estimated Effect:         ',refute_results.__dict__.get('estimated_effect'))
                print('New Effect:               ',refute_results.__dict__.get('new_effect'))
                print('Statistical Significance: ',refute_results.__dict__.get('refutation_result').get('is_statistically_significant'))
                print('P-Value:                  ',refute_results.__dict__.get('refutation_result').get('p_value'))
                print('')
                refute_results = cmodel.refute_estimate(identified_estimand, estimate,
                method_name="placebo_treatment_refuter", placebo_type="permute")
                print('Refuter 3: ')
                print('Refuter Type:             ',refute_results.__dict__.get('refutation_type'))
                print('Estimated Effect:         ',refute_results.__dict__.get('estimated_effect'))
                print('New Effect:               ',refute_results.__dict__.get('new_effect'))
                print('Statistical Significance: ',refute_results.__dict__.get('refutation_result').get('is_statistically_significant'))
                print('P-Value:                  ',refute_results.__dict__.get('refutation_result').get('p_value'))
                print('')
                refute_results = cmodel.refute_estimate(identified_estimand, estimate,
                method_name="bootstrap_refuter")
                print('Refuter 4: ')
                print('Refuter Type:             ',refute_results.__dict__.get('refutation_type'))
                print('Estimated Effect:         ',refute_results.__dict__.get('estimated_effect'))
                print('New Effect:               ',refute_results.__dict__.get('new_effect'))
                print('Statistical Significance: ',refute_results.__dict__.get('refutation_result').get('is_statistically_significant'))
                print('P-Value:                  ',refute_results.__dict__.get('refutation_result').get('p_value'))

        elif output == 'full':
            print('Simulator Mode (Full) Parameters: ')
            print('Identification Using: identifier=\'',idparam,'\'')
            print('Estimation Using: estimator=\'',estparam,'\'')
            print('')
            #Identify the causal effect
            identified_estimand = cmodel.identify_effect(proceed_when_unidentifiable=True, method_name=idparam)
            estimandcheck = str(identified_estimand)
            estimandcheck2 = estimandcheck.split("Estimand expression:")
            estimandcheck = estimandcheck.replace(estimandcheck2[1], '')
            estimandcheck = estimandcheck[:-20]
            estimandcheck = estimandcheck[-30:]
            estimandcheck = estimandcheck.split("Estimand name: ")
            estimandcheck = estimandcheck[1].strip()
            if estimandcheck == 'iv': 
                print('iv defaults to the \'iv.instrumental_variable\' estimation method.')
                print('DoWhy has other iv estimation methods: ')
                print('   Binary Instrument/Wald Estimator')
                print('   Two-stage least squares')
                print('   Regression discontinuity')
                print('')
                if estimator == 'default':
                    estparam = 'iv.instrumental_variable'
                    estmethod = None
                elif estimator != 'default' and estimatorparams == 'default':
                    estmethod = None
            elif estimandcheck == 'frontdoor': 
                print('DoWhy has one frontdoor estimation method: \'frontdoor.two_stage_regression\'')
                estparam = 'frontdoor.two_stage_regression'
                estmethod = None
            elif estimandcheck == 'backdoor': 
                print('backdoor defaults to the \'backdoor.linear_regression\' estimation method.')
                print('Other backdoor estimation methods include: ')
                print('Logistic:   backdoor.generalized_linear_model')
                print('Logistic:   backdoor.econml.dr.LinearDRLearner')
                print('DoubleML:   backdoor.econml.dml.DML')
                print('')
                print('Note: Propensity Scores Require a Binary Treatment Variable')
                print('Propensity: backdoor.propensity_score_stratification')
                print('Propensity: backdoor.propensity_score_matching')
                print('Propensity: backdoor.propensity_score_weighting')
                print('')
                if estimator == 'default':
                    estparam = 'backdoor.linear_regression'
                    estmethod = None
                elif estimator == 'backdoor.generalized_linear_model':
                    if estimatorparams == 'default':
                        estmethod = {'num_null_simulations':10,
                        'num_simulations':10,
                        'num_quantiles_to_discretize_cont_cols':10,
                        'fit_method': "statsmodels",
                        'glm_family': sm.families.Binomial(),
                        'need_conditional_estimates':False}
                    else:
                        estmethod = estimatorparams
                elif estimator == 'backdoor.econml.dr.LinearDRLearner':
                    if estimatorparams == 'default':
                        estmethod = {"init_params":{
                            'model_propensity': LogisticRegressionCV(cv=3, solver='lbfgs', multi_class='auto')},
                            "fit_params":{}}
                    else:
                        estmethod = estimatorparams
                elif estimator == 'backdoor.econml.dml.DML':
                    if estimatorparams == 'default':
                        estmethod = {'init_params':{
                            'model_y':GradientBoostingRegressor(),
                            'model_t': GradientBoostingRegressor(),
                            'model_final': LassoCV(fit_intercept=False),
                            'featurizer':PolynomialFeatures(degree=1, include_bias=True)},
                            'fit_params':{'inference': BootstrapInference(n_bootstrap_samples=20, n_jobs=-1)}}
                    else: 
                        estmethod = estimatorparams
                elif estimator == 'backdoor.propensity_score_stratification':
                    estparam = 'backdoor.propensity_score_stratification' 
                    targetunit = 'att'
                    print('Propensity Score Unit: ATT')
                    estmethod = None
                elif estimator == 'backdoor.propensity_score_matching':
                    estparam = 'backdoor.propensity_score_matching' 
                    targetunit = 'atc'
                    print('Propensity Score Unit: ATC')
                    estmethod = None
                elif estimator == 'backdoor.propensity_score_weighting':
                    estparam = 'backdoor.propensity_score_weighting' 
                    targetunit = 'ate'
                    print('Propensity Score Unit: ATE')
                    estmethod = None
                    estmethod = {"weighting_scheme":"ips_weight"}
                else:
                    targetunit = None
                    estmethod = None
            elif estimandcheck != 'iv' and estimandcheck != 'frontdoor' and estimandcheck != 'backdoor':
                print('Unable to identify DAG Estimand. Please input a valid DAG.')
                return

            print('Detected Estimator method_name (DoWhy): \'',estparam,'\'')
            print('Detected Estimator method_params (DoWhy): ',estmethod)
            print('')
            print('Identify: ')
            print('==================================================')
            print('Treatment Variable:       ',identified_estimand.__dict__.get('treatment_variable'))
            print('Outcome Variable:         ',identified_estimand.__dict__.get('outcome_variable'))
            print('Backdoor Variabless:      ',identified_estimand.__dict__.get('backdoor_variables'))
            print('Instrumental Variables:   ',identified_estimand.__dict__.get('instrumental_variables'))
            print('Frontdoor Variables:      ',identified_estimand.__dict__.get('frontdoor_variables'))
            print('Mediator Variables:       ',identified_estimand.__dict__.get('mediator_variables'))
            print('First Stage Confounders:  ',identified_estimand.__dict__.get('mediation_first_stage_confounders'))
            print('Second Stage Confounders: ',identified_estimand.__dict__.get('mediation_second_stage_confounders'))
            print('')
            print(identified_estimand)

            #Estimate
            print('Estimate: ')
            print('==================================================')
            estimate = cmodel.estimate_effect(identified_estimand,
            method_name=estparam, test_significance=True,method_params = estmethod)
            print('Estimate Value:           ',estimate.__dict__.get('value'))
            print('Estimand Expression:      ',estimate.__dict__.get('realized_estimand_expr'))
            print('Control Value:            ',estimate.__dict__.get('control_value'))
            print('Treatment Value:          ',estimate.__dict__.get('treatment_value'))
            print('Conditional Estimate:     ',estimate.__dict__.get('conditional_estimates'))
            print('Intercept:                ',estimate.__dict__.get('intercept'))
            print('Effect Strength:          ',estimate.__dict__.get('effect_strength'))
            print('Estimator:                ',estimate.__dict__.get('estimator'))
            print('Intercept:                ',estimate.__dict__.get('params').get('intercept'))
            print('Estimand Type:            ',estimate.__dict__.get('params').get('estimand_type'))
            print('Estimator Class:          ',estimate.__dict__.get('params').get('estimator_class'))
            print('Test Significance:        ',estimate.__dict__.get('params').get('test_significance'))
            print('Evaluate Effect Strength: ',estimate.__dict__.get('params').get('evaluate_effect_strength'))
            print('Confidence Intervals:     ',estimate.__dict__.get('params').get('confidence_intervals'))
            print('Target Units:             ',estimate.__dict__.get('params').get('target_units'))
            print('Effect Modifiers:         ',estimate.__dict__.get('params').get('effect_modifiers'))
            try:
                print('Propensity Scores:        ',estimate.__dict__.get('params').get('propensity_scores'))
            except:
                print('Propensity Scores:      ')
            print('')

            #Refute
            print('Refute: ')
            print('==================================================')
            if estimandcheck == 'iv':
                refute_results = cmodel.refute_estimate(identified_estimand, estimate,
                method_name="data_subset_refuter", subset_fraction=0.9)
                print('Refuter 1: ')
                print('Refuter Type:             ',refute_results.__dict__.get('refutation_type'))
                print('Estimated Effect:         ',refute_results.__dict__.get('estimated_effect'))
                print('New Effect:               ',refute_results.__dict__.get('new_effect'))
                print('Statistical Significance: ',refute_results.__dict__.get('refutation_result').get('is_statistically_significant'))
                print('P-Value:                  ',refute_results.__dict__.get('refutation_result').get('p_value'))
                print('')
                refute_results = cmodel.refute_estimate(identified_estimand, estimate,
                method_name="placebo_treatment_refuter", placebo_type="permute")
                print('Refuter 2: ')
                print('Refuter Type:             ',refute_results.__dict__.get('refutation_type'))
                print('Estimated Effect:         ',refute_results.__dict__.get('estimated_effect'))
                print('New Effect:               ',refute_results.__dict__.get('new_effect'))
                print('Statistical Significance: ',refute_results.__dict__.get('refutation_result').get('is_statistically_significant'))
                print('P-Value:                  ',refute_results.__dict__.get('refutation_result').get('p_value'))
                print('')
                refute_results = cmodel.refute_estimate(identified_estimand, estimate,
                method_name="bootstrap_refuter")
                print('Refuter 3: ')
                print('Refuter Type:             ',refute_results.__dict__.get('refutation_type'))
                print('Estimated Effect:         ',refute_results.__dict__.get('estimated_effect'))
                print('New Effect:               ',refute_results.__dict__.get('new_effect'))
                print('Statistical Significance: ',refute_results.__dict__.get('refutation_result').get('is_statistically_significant'))
                print('P-Value:                  ',refute_results.__dict__.get('refutation_result').get('p_value'))
            elif estimandcheck != 'iv':
                refute_results=cmodel.refute_estimate(identified_estimand, estimate,
                method_name="random_common_cause")
                print('Refuter 1: ')
                print('Refuter Type:             ',refute_results.__dict__.get('refutation_type'))
                print('Estimated Effect:         ',refute_results.__dict__.get('estimated_effect'))
                print('New Effect:               ',refute_results.__dict__.get('new_effect'))
                print('Statistical Significance: ',refute_results.__dict__.get('refutation_result').get('is_statistically_significant'))
                print('P-Value:                  ',refute_results.__dict__.get('refutation_result').get('p_value'))
                print('')
                refute_results = cmodel.refute_estimate(identified_estimand, estimate,
                method_name="data_subset_refuter", subset_fraction=0.9)
                print('Refuter 2: ')
                print('Refuter Type:             ',refute_results.__dict__.get('refutation_type'))
                print('Estimated Effect:         ',refute_results.__dict__.get('estimated_effect'))
                print('New Effect:               ',refute_results.__dict__.get('new_effect'))
                print('Statistical Significance: ',refute_results.__dict__.get('refutation_result').get('is_statistically_significant'))
                print('P-Value:                  ',refute_results.__dict__.get('refutation_result').get('p_value'))
                print('')
                refute_results = cmodel.refute_estimate(identified_estimand, estimate,
                method_name="placebo_treatment_refuter", placebo_type="permute")
                print('Refuter 3: ')
                print('Refuter Type:             ',refute_results.__dict__.get('refutation_type'))
                print('Estimated Effect:         ',refute_results.__dict__.get('estimated_effect'))
                print('New Effect:               ',refute_results.__dict__.get('new_effect'))
                print('Statistical Significance: ',refute_results.__dict__.get('refutation_result').get('is_statistically_significant'))
                print('P-Value:                  ',refute_results.__dict__.get('refutation_result').get('p_value'))
                print('')
                refute_results = cmodel.refute_estimate(identified_estimand, estimate,
                method_name="bootstrap_refuter")
                print('Refuter 4: ')
                print('Refuter Type:             ',refute_results.__dict__.get('refutation_type'))
                print('Estimated Effect:         ',refute_results.__dict__.get('estimated_effect'))
                print('New Effect:               ',refute_results.__dict__.get('new_effect'))
                print('Statistical Significance: ',refute_results.__dict__.get('refutation_result').get('is_statistically_significant'))
                print('P-Value:                  ',refute_results.__dict__.get('refutation_result').get('p_value'))

    def makegraph(function='main',edges='edges',graph='graph',dataset='dataset',treatment='treatment',outcome='outcome',model='model', digraph='digraph',eda=False):
        samples = 1500
        rng = np.random.default_rng()
        if function == 'tutorial':
            if model == 'backdoor':
                U = rng.standard_normal(size = samples, dtype = np.float32)
                X = 0.19*U + rng.standard_normal(size = samples, dtype = np.float32)
                Y = 0.36*U + 0.65*X + 0.25*rng.standard_normal(size = samples, dtype = np.float32)
                data = pd.DataFrame()
                data['X'] = X
                data['Y'] = Y
                data['U'] = U
                strValue ="""digraph {
                U;
                X;
                Y;
                U -> X;
                U -> Y;
                X -> Y;
                }
                """
                print('CausalFast makegraph() Tutorial')
                print('____________________________________________')
                print('This function returns a tuple: ')
                print('makegraph[0] is a complete DoWhy model')
                print('makegraph[1] is a string in the DOT form to create a digraph')
                print('makegraph[2] is the dataset')
                print('')
                print('____________________________________________')
                print('This graph shows the backdoor criterion.')
                print('treatment:              \'X\'')
                print('outcome:                \'Y\'')
                print('observed confounder:    \'U\'')
                print('data:')
                print(data.head())
                print('')
                print('Start of Digraph code:')
                print('')
                n = 7
                replacementStr = '\"\"\"digraph'
                strValue2 = replacementStr + strValue[n:]
                endstr = '\"\"\"'
                strValue2 = strValue2 + endstr
                print(strValue2)
                print(':End of Digraph code.')
                print('')
                print('Note: Copy the entire string, including start \"\"\" and end \"\"\".')
                print('Note: Do not wrap this string in quotes when assigning it to a varible for use in DoWhy CausalModel')
                cmodel = CausalModel(
                    data=data,
                    treatment='X',
                    outcome='Y',
                    graph=strValue)
                cmodel.view_model()
                print('')
                print('Returned: [0] CausalModel, [1] Digraph string, and [2] dataset')
                return cmodel, strValue, data
            
            elif model == 'frontdoor':
                U = rng.standard_normal(size = samples, dtype = np.float32)
                X = 0.19*U + rng.standard_normal(size = samples, dtype = np.float32)
                Z = 0.75*X + 0.2*rng.standard_normal(size = samples, dtype = np.float32)
                Y = 0.36*U + 0.90*Z + 0.25*rng.standard_normal(size = samples, dtype = np.float32)
                data = pd.DataFrame()
                data['X'] = X
                data['Y'] = Y
                data['Z'] = Z
                strValue ="""digraph {
                X;
                Y;
                U;
                Z;
                X -> Z;
                Z -> Y;
                U -> X;
                U -> Y;
                }
                """
                print('CausalFast makegraph() Tutorial')
                print('____________________________________________')
                print('This function returns a tuple: ')
                print('makegraph[0] is a complete DoWhy model')
                print('makegraph[1] is a string in the DOT form to create a digraph')
                print('makegraph[2] is the dataset')
                print('')
                print('____________________________________________')
                print('This graph shows the frontdoor criterion.')
                print('treatment:              \'X\'')
                print('outcome:                \'Y\'')
                print('observed confounder:    \'U\'')
                print('unobserved confounder:  \'Z\'')
                print('data:')
                print(data.head())
                print('')
                print('Start of Digraph code:')
                print('')
                n = 7
                replacementStr = '\"\"\"digraph'
                strValue2 = replacementStr + strValue[n:]
                endstr = '\"\"\"'
                strValue2 = strValue2 + endstr
                print(strValue2)
                print(':End of Digraph code.')
                print('')
                print('Note: Copy the entire string, including start \"\"\" and end \"\"\".')
                print('Note: Do not wrap this string in quotes when assigning it to a varible for use in DoWhy CausalModel')
                cmodel = CausalModel(
                    data=data,
                    treatment='X',
                    outcome='Y',
                    graph=strValue)
                cmodel.view_model()
                print('')
                print('Returned: [0] CausalModel, [1] Digraph string, and [2] dataset')
                return cmodel, strValue, data
            
            elif model == 'iv':
                U = rng.standard_normal(size = samples, dtype = np.float32)
                Z = rng.standard_normal(size = samples, dtype = np.float32)
                X = 0.19*U + 0.70*Z + rng.standard_normal(size = samples, dtype = np.float32)
                Y = 0.36*U + 0.90*X + 0.25*rng.standard_normal(size = samples, dtype = np.float32)
                data = pd.DataFrame()
                data['X'] = X
                data['Y'] = Y
                data['Z'] = Z
                strValue ="""digraph {
                X;
                Y;
                U;
                Z;
                Z -> X;
                X -> Y;
                U -> X;
                U -> Y;
                }
                """
                print('CausalFast makegraph() Tutorial')
                print('____________________________________________')
                print('This function returns a tuple: ')
                print('makegraph[0] is a complete DoWhy model')
                print('makegraph[1] is a string in the DOT form to create a digraph')
                print('makegraph[2] is the dataset')
                print('')
                print('____________________________________________')
                print('This graph shows an instrumental variable.')
                print('treatment:              \'X\'')
                print('outcome:                \'Y\'')
                print('iv:                     \'U\'')
                print('unobserved confounder:  \'Z\'')
                print('data:')
                print(data.head())
                print('')
                print('Start of Digraph code:')
                print('')
                n = 7
                replacementStr = '\"\"\"digraph'
                strValue2 = replacementStr + strValue[n:]
                endstr = '\"\"\"'
                strValue2 = strValue2 + endstr
                print(strValue2)
                print(':End of Digraph code.')
                print('')
                print('Note: Copy the entire string, including start \"\"\" and end \"\"\".')
                print('Note: Do not wrap this string in quotes when assigning it to a varible for use in DoWhy CausalModel')
                cmodel = CausalModel(
                    data=data,
                    treatment='X',
                    outcome='Y',
                    graph=strValue)
                cmodel.view_model()
                print('')
                print('Returned: [0] CausalModel, [1] Digraph string, and [2] dataset')
                return cmodel, strValue, data
            
            else:
                print('CausalFast makegraph() Tutorial')
                print('')
                print('____________________________________________')
                print('Help with tutorials: makegraph has three tutorial modes that return CausalModels used by DoWhy or CausalFast Simulator')
                print('Tutorial 1: Backdoor Criterion')
                print('Syntax: makegraph(function=\'tutorial\', model=\'backdoor\')')
                print('')
                print('Tutorial 2: Frontdoor Criterion')
                print('Syntax: makegraph(function=\'tutorial\', model=\'frontdoor\')')
                print('')
                print('Tutorial 3: Instrumental Variables')
                print('Syntax: makegraph(function=\'tutorial\', model=\'iv\')')
                print('')
                print('____________________________________________')
                print('Assign these tutorials to an object to use with CausalFast Simulator or DoWhy:')
                print('Syntax: causalfast.simulator(model=\'CausalModelObject\')')
                print('')
                print('____________________________________________')
                print('Other Help Commands:')
                print('Syntax: makegraph(function=\'help\')')
                print('Syntax: makegraph(function=\'helpcausalmodel\')')
                print('Syntax: makegraph(function=\'helpcausalgraph\')')
                print('Syntax: makegraph(function=\'helptutorial\')')
                print('')
        
        elif function == 'graphmaker':
            if edges != 'edges':
                if isinstance(edges, str):
                    print('CausalFast makegraph() Causal Graph Maker')
                    print('')
                    print('____________________________________________')
                    print('You have inputted the following as string objects:')
                    print('edges= \"',edges,'\"')
                    print('')
                    print('Please input a list object of the format:')
                    print('edgelist = [(\'X\',\'Y\'),(\'U\',\'X\'),(\'U\',\'Y\')]')
                    print('')
                    print('Command: makegraph(function=\'graphmaker\',edges=edgelist)')
                    
                elif isinstance(edges, list):
                    print('CausalFast makegraph() Causal Graph Maker')
                    print('')
                    print('____________________________________________')
                    print('Note: Please ensure your variable names precisely match the column names in your causal data.')
                    print('Note: If the variable names and column names in the causal data do not match, DoWhy will be unable to create an estimate.')
                    print('Note: The following string can be accepted by the DoWhy CausalModel object \'graph=\' arguement')
                    print('')
                    print('Edges List: ')
                    print(edges)
                    print('')
                    seed = 123
                    G = nx.DiGraph()
                    G.add_edges_from(edges)
                    pos = nx.spring_layout(G, seed=seed)
                    visual = nx.draw(G, pos, with_labels = True, width=0.4, edge_color='red', style=':', node_size=400, arrows=True)
                    DOTG = nx.nx_pydot.to_pydot(G).to_string()
                    n = 14
                    replacementStr = '\"\"\"digraph'
                    strValue = replacementStr + DOTG[n:]
                    strValue = strValue.replace("--", "->")
                    endstr = '\"\"\"'
                    strValue = strValue + endstr
                    print('')
                    print('Start of Digraph string:')
                    print('')
                    print(strValue)
                    print(':End of Digraph string')
                    print('')
                    print('Note: Copy the entire string, including start \"\"\" and end \"\"\".')
                    print('Note: Do not wrap this string in quotes when assigning it to a varible for use in DoWhy CausalModel')
                    print('')
                    print('Returned: [0] Digraph string')
                    return strValue
                
            else:
                print('CausalFast makegraph() Causal Graph Maker')
                print('')
                print('____________________________________________')
                print('Help with making Causal Graphs: Create a Directed Acyclic Graph / DAG using NetworkX to visualize a causal system')
                print('Returns a digraph string that can be used later in DoWhy.')
                print('Note: Please verify the directionality of the digraph string')
                print('')
                print('Provide the following parameters (as a list object): edges')
                print('')
                print('Example: ')
                print('edgelist = [(\'X\',\'Y\'),(\'U\',\'X\'),(\'U\',\'Y\')]')
                print('Syntax: makegraph(function=\'graphmaker\', edges=edgelist)')
                print('')
                print('____________________________________________')
                print('Other Help Commands:')
                print('Syntax: makegraph(function=\'help\')')
                print('Syntax: makegraph(function=\'helpcausalmodel\')')
                print('Syntax: makegraph(function=\'helpcausalgraph\')')
                print('Syntax: makegraph(function=\'helptutorial\')')
                print('')

        elif function == 'makecausalmodel':
            if isinstance(dataset, pd.DataFrame):
                if eda == True:
                    if graph == 'graph':
                        print('CausalFast makegraph() Causal Model Maker: EDA Mode (No Digraph Data)')
                        print('')
                        print('____________________________________________')
                        print('Tool for Making Causal Models: Input Data EDA + Causal Model')
                        print('Include a digraph string to the  to view your graph string')
                        print('Set the \'eda=False\' and include a \'graph=graphmodel\' parameter to return a CausalModel')
                        print('Note: It is very important the column names in the dataset, the digraph variable names, and the treatment and outcome variable names all match precisely. If these do not match then DoWhy will be unable to generate a causal estiamte')
                        print('')
                        print('Dimensions: ', dataset.shape)
                        print('Column Names:', list(dataset.columns))
                        print('')
                        print('Basic EDA: ')
                        for (columnName, columnData) in dataset.items():
                            if is_numeric_dtype(dataset[columnName]) and dataset[columnName].isna().sum() == 0:
                                print('NA Values: ',dataset[columnName].isna().sum(),' Datatype: ',dataset[columnName].dtype,' Column Name: ', columnName,' MIN: ', round(min(columnData),2),' AVG: ', round(columnData.mean(),2),' MAX: ', round(max(columnData),2))
                            else:
                                print('NA Values: ',dataset[columnName].isna().sum(),' Datatype: ',dataset[columnName].dtype,' Column Name: ', columnName)
                        print('')
                        
                    elif graph != 'graph':
                        print('CausalFast makegraph() Causal Model Maker: EDA Mode (With Digraph)')
                        print('')
                        print('____________________________________________')
                        print('Tool for Making Causal Models: Input Data EDA + Digraph')
                        print('Set the \'eda=False\' and include a \'graph=graphmodel\' parameter to return a CausalModel')
                        print('Note: It is very important the column names in the dataset, the digraph variable names, and the treatment and outcome variable names all match precisely. If these do not match then DoWhy will be unable to generate a causal estiamte')
                        print('')
                        print('Dimensions: ', dataset.shape)
                        print('Column Names:', list(dataset.columns))
                        print('')
                        print('Basic EDA: ')
                        for (columnName, columnData) in dataset.items():
                            if is_numeric_dtype(dataset[columnName]) and dataset[columnName].isna().sum() == 0:
                                print('NA Values: ',dataset[columnName].isna().sum(),' Datatype: ',dataset[columnName].dtype,' Column Name: ', columnName,' MIN: ', round(min(columnData),2),' AVG: ', round(columnData.mean(),2),' MAX: ', round(max(columnData),2))
                            else:
                                print('NA Values: ',dataset[columnName].isna().sum(),' Datatype: ',dataset[columnName].dtype,' Column Name: ', columnName)
                        print('')
                        print('Treatment: ', treatment, '  (Verify your treatment parameter with the Digraph and Dataset column name)')
                        print('Outcome: ', outcome, '  (Verify your outcome parameter with the Digraph and Dataset column name)')
                        print('')
                        print('Graph String: ')
                        print(graph)
                        print('')
                        
                elif eda == False and graph != 'graph':
                    print('CausalFast makegraph() Causal Model Maker:  Make a CausalModel Object (EDA Turned Off)')
                    print('')
                    print('____________________________________________')
                    print('Tool for Making Causal Models: Returns a CausalModel + Digraph + Data')
                    print('Note: It is very important the column names in the dataset, the digraph variable names, and the treatment and outcome variable names all match precisely. If these do not match then DoWhy will be unable to generate a causal estiamte')
                    print('')
                    print('Treatment: ', treatment, '  (Verify your treatment parameter with the Digraph and Dataset column name)')
                    print('Outcome: ', outcome, '  (Verify your outcome parameter with the Digraph and Dataset column name)')
                    print('')
                    print('Graph String: ')
                    print(graph)
                    cmodel = CausalModel(
                        data=dataset,
                        treatment=treatment,
                        outcome=outcome,
                        graph=graph)
                    cmodel.view_model()
                    print('')
                    print('Returned: [0] CausalModel, [1] Digraph string, and [2] dataset')
                    return cmodel, graph, dataset
                    
                else:
                    print('EDA Mode: Not a valid eda=True parameter')
                    print('CausalModel Maker Mode: not a valid dataset (pandas dataframe) or graph (DOT format digraph string)')
                    
            else:
                print('CausalFast makegraph() Help Mode: making a CausalModel')
                print('')
                print('____________________________________________')
                print('Help with making a CausalModels: Create a CausalModel object used by DoWhy or CausalFast Simulator')
                print('EDA Syntax: makegraph(function=\'makecausalmodel\', dataset=dataset, eda=True)')
                print('Model Syntax: makegraph(function=\'makecausalmodel\', graph=graph, dataset=dataset, eda=True)')
                print('EDA + Model Syntax: makegraph(function=\'makecausalmodel\', graph=graph, dataset=dataset, treatment=treatment, outcome=outcome, eda=False)')
                print('')
                print('Note: the \'graph=graph\' object should be parameterized to a list of the pattern: ')
                n = 7
                strValue ="""digraph {
                U;
                X;
                Y;
                U -> X;
                U -> Y;
                X -> Y;
                }
                """
                replacementStr = 'graph = \"\"\"digraph'
                strValue2 = replacementStr + strValue[n:]
                endstr = '\"\"\"'
                strValue2 = strValue2 + endstr
                print(strValue2)

        elif function == 'helpcausalmodel':
            print('CausalFast makegraph() Help Mode')
            print('')
            print('____________________________________________')
            print('Help with making Causal Models: Create a CausalModel object used by DoWhy or CausalFast Simulator')
            print('EDA Syntax: makegraph(function=\'makecausalmodel\', dataset=dataset, eda=True)')
            print('EDA + graph Syntax: makegraph(function=\'makecausalmodel\', graph=graph, dataset=dataset)')
            print('CausalModel Maker Syntax: makegraph(function=\'makecausalmodel\', graph=graph, dataset=dataset, treatment=\'treatment\', outcome=\'outcome\')')
            print('')
            print('Note: the \'graph=graph\' object should be parameterized to a list of the pattern: ')
            n = 7
            strValue ="""digraph {
            U;
            X;
            Y;
            U -> X;
            U -> Y;
            X -> Y;
            }
            """
            replacementStr = 'graph = \"\"\"digraph'
            strValue2 = replacementStr + strValue[n:]
            endstr = '\"\"\"'
            strValue2 = strValue2 + endstr
            print(strValue2)
            print('')
            print('Use the causal graph maker feature for assistance: ')
            print('')
            print('____________________________________________')
            print('')
            
        elif function == 'helpcausalgraph':
            print('CausalFast makegraph() Help Mode')
            print('')
            print('____________________________________________')
            print('Help with making Causal Graphs: Create a Directed Acyclic Graph / DAG using NetworkX to visualize a causal system')
            print('Returns a digraph string that can be used later in DoWhy.')
            print('Note: Please verify the directionality of the digraph string')
            print('')
            print('Provide the following parameters (as a list object): edges')
            print('')
            print('Example: ')
            print('edgelist = [(\'X\',\'Y\'),(\'U\',\'X\'),(\'U\',\'Y\')]')
            print('Syntax: makegraph(function=\'graphmaker\', edges=edgelist)')
            print('')
            print('____________________________________________')
            print('Other Help Commands:')
            print('Syntax: makegraph(function=\'help\')')
            print('Syntax: makegraph(function=\'helpcausalmodel\')')
            print('Syntax: makegraph(function=\'helpcausalgraph\')')
            print('Syntax: makegraph(function=\'helptutorial\')')
            print('')
                
        elif function == 'helptutorial':
            print('CausalFast makegraph() Help Mode')
            print('')
            print('____________________________________________')
            print('Help with tutorials: makegraph has three tutorial modes that return CausalModels used by DoWhy or CausalFast Simulator')
            print('Tutorial 1: Backdoor Criterion')
            print('Syntax: makegraph(function=\'tutorial\', model=\'backdoor\')')
            print('')
            print('Tutorial 2: Frontdoor Criterion')
            print('Syntax: makegraph(function=\'tutorial\', model=\'frontdoor\')')
            print('')
            print('Tutorial 3: Instrumental Variables')
            print('Syntax: makegraph(function=\'tutorial\', model=\'iv\')')
            print('')
            print('____________________________________________')
            print('Assign these tutorials to an object to use with CausalFast Simulator or DoWhy:')
            print('Syntax: causalfast.simulator(model=\'CausalModelObject\')')
            print('')
            print('____________________________________________')
            print('Other Help Commands:')
            print('Syntax: makegraph(function=\'help\')')
            print('Syntax: makegraph(function=\'helpcausalmodel\')')
            print('Syntax: makegraph(function=\'helpcausalgraph\')')
            print('Syntax: makegraph(function=\'helptutorial\')')
            print('')
            
        else:
            print('CausalFast makegraph() Help Mode')
            print('')
            print('____________________________________________')
            print('1) DoWhy CausalModel Maker:')
            print('   Make a CausalModel for use in DoWhy or in CausalFast Simulator.')
            print('   Provide the following parameters: graph, data, treatment, outcome')
            print('   Syntax: makegraph(function=\'makecausalmodel\', graph=\'graph\', dataset=\'dataset\', treatment=\'treatment\', outcome=\'outcome\')')
            print('')
            print('   Additional Help: ')
            print('   Syntax: makegraph(function=\'helpcausalmodel\')')
            print('')
            print('____________________________________________')
            print('2) Causal Graph Maker:')
            print('   Build a Causal Graph from a list object describing the edges between nodes')
            print('   Provide the following parameters: edges')
            print('   Syntax: makegraph(function=\'graphmaker\', edges=edgelist)')
            print('')
            print('   Additional Help: ')
            print('   Syntax: makegraph(function=\'helpcausalgraph\')')
            print('')
            print('____________________________________________')
            print('3) Tutorial Mode:')
            print('   This generates DoWhy CausalModels of the three estimand criterion for use in the simulator')
            print('   Acceptable model parameters include: model=\'backdoor\', model=\'frontdoor\', model=\'iv\' ')
            print('   Syntax: makegraph(function=\'tutorial\', model=\'backdoor\')')
            print('   Syntax: makegraph(function=\'tutorial\', model=\'frontdoor\')')
            print('   Syntax: makegraph(function=\'tutorial\', model=\'iv\')')
            print('')
            print('   Additional Help: ')
            print('   Syntax: makegraph(function=\'helptutorials\')')
            print('')
            print('____________________________________________')
            print('4) Help Mode:')
            print('   Provide the helpmode arguments to enter helpmode)')
            print('   Syntax: makegraph(function=\'help\')')
            print('   Syntax: makegraph(function=\'helpcausalmodel\')')
            print('   Syntax: makegraph(function=\'helpcausalgraph\')')
            print('   Syntax: makegraph(function=\'helptutorial\')')
            print('')
