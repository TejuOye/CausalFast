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
        **kwargs,):
        self._data = data
        self._treatment = parse_state(treatment)
        self._outcome = parse_state(outcome)
        self._effect_modifiers = parse_state(effect_modifiers)
        self._estimand_type = estimand_type
        self._proceed_when_unidentifiable = proceed_when_unidentifiable
        self._missing_nodes_as_confounders = missing_nodes_as_confounders
        self.logger = logging.getLogger(__name__)
        self._estimator_cache = {}

    def simulator(causalmodel='causalmodel',identifier='default',method_name='default',method_params='default',unit='default',full_output=True,refute=True):
        def newline():
            print('')
        def menubreak():
            newline()
            print('____________________________________________')
        def helpmenu0():
            print('CausalFast Simulator():')
        def idbreak():
            print('Identify: ')
            print('==================================================')
        def estbreak():
            print('Estimate: ')
            print('==================================================')
        def refbreak():
            print('Refute: ')
            print('==================================================')
        def ivstatement():
            print('Note: method_name=\'default\' uses \'iv.instrumental_variable\'')
            print('DoWhy has other iv estimation methods: ')
            print('   Binary Instrument/Wald Estimator')
            print('   Two-stage least squares')
            print('   Regression discontinuity')
            print('')
        def frontdoorstatement():
            print('DoWhy has one frontdoor estimation method: \'frontdoor.two_stage_regression\'')
        def backdoorstatement():
            print('Note: method_name=\'default\' uses \'backdoor.linear_regression\' or \'backdoor.generalized_linear_model\'')
            print('Note: method_name=\'default\' override options include: ')
            print('   OLS Linear:  \'backdoor.linear_regression\'')
            print('   LogisticGLM: \'backdoor.generalized_linear_model\'')
            print('   LogisticML:  \'backdoor.econml.dr.LinearDRLearner\'')
            print('   DoubleML:    \'backdoor.econml.dml.DML\'')
            print('   Propensity:  \'backdoor.propensity_score_stratification\'')
            print('   Propensity:  \'backdoor.propensity_score_matching\'')
            print('   Propensity:  \'backdoor.propensity_score_weighting\'')
            newline()
            print('Note: Propensity Scores require both binary outcome and binary treatment variables')
            print('Note: generalized_linear_model (Logistic) & econml.dr.LinearDRLearner require a binary outcome variable')
        def nodag():
            print('Unable to identify DAG Estimand. Please input a valid DAG.')
        def returnstatement():
            print('Returned: simulator[0] = DoWhyEstimateObj, simulator[1] = PropensityScores')
        def mainmenu():
            helpmenu0()
            menubreak()
            print('Note:')
            print('You have not provided a valid CausalModel object')
            print('Create a valid CausalModel using causalfast makegraph()')
            print('Return this object and use it as a parameter in causalfast.simulator()')
            newline()
            print('The CausalFast simulator will attempt to automatically run DoWhy analysis.')
            print('By default, the CausalFast simulator will identify the correct estimand and assign a relevant estimator')
            print('Syntax: causalfast.simulator(causalmodel=makegraphobj[0])')
            print('Syntax: causalfast.simulator(causalmodel=\'causalmodel\',identifier=\'default\',method_name=\'default\',full_output=True')
            newline()
            print('Available DoWhy Identifier algorithms:')
            print('   minimal-adjustment, maximal-adjustment, exhaustive-search, default')
            newline()
            print('Automatically Selected DoWhy Estimators:')
            print('   Backdoor:  backdoor.linear_regression')
            print('   Frontdoor: frontdoor.two_stage_regression')
            print('   Instrumental Variable: iv.instrumental_variable')
            newline()
            print('Overrideable Estimators: Additional Required parameters (method_params) are automatically generated')
            print('   Backdoor + binary outcome (logistic):  backdoor.generalized_linear_model')
            print('   Backdoor + binary outcome (logistic):  backdoor.econml.dr.LinearDRLearner')
            print('   Backdoor + MachineLearning (DoubleML): backdoor.econml.dml.DML')
            print('   Backdoor + binary treatment (Propensity Score ATT): backdoor.propensity_score_stratification')
            print('   Backdoor + binary treatment (Propensity Score ATC): backdoor.propensity_score_matching')
            print('   Backdoor + binary treatment (Propensity Score ATE): backdoor.propensity_score_weighting')
            newline()
            print('Note: Logistic Regression requires a binary outcome variable')
            print('Note: Propensity Scores require both binary outcome and binary treatment variables')
            return
        def estcheck(identified_estimand):
                estimandchk = str(identified_estimand)
                estimandchk2 = estimandchk.split("Estimand expression:")
                estimandchk = estimandchk.replace(estimandchk2[1], '')
                estimandchk = estimandchk[:-20]
                estimandchk = estimandchk[-30:]
                estimandchk = estimandchk.split("Estimand name: ")
                estimandchk = estimandchk[1].strip()
                return estimandchk
        #note ESTIMAND CHECK IS BROKEN IF WRONG DAG IS USED (CORRUPTED OUTPUT WHEN TWO ESTIMAND ARE DETECTED BACKDOOR + IV/FRONTDOOR)

        def idstep(identified_estimand):
            print('Treatment Variable:       ',identified_estimand.__dict__.get('treatment_variable'))
            print('Outcome Variable:         ',identified_estimand.__dict__.get('outcome_variable'))
            print('Backdoor Variabless:      ',identified_estimand.__dict__.get('backdoor_variables'))
            print('Instrumental Variables:   ',identified_estimand.__dict__.get('instrumental_variables'))
            print('Frontdoor Variables:      ',identified_estimand.__dict__.get('frontdoor_variables'))
            print('Mediator Variables:       ',identified_estimand.__dict__.get('mediator_variables'))
            print('First Stage Confounders:  ',identified_estimand.__dict__.get('mediation_first_stage_confounders'))
            print('Second Stage Confounders: ',identified_estimand.__dict__.get('mediation_second_stage_confounders'))
            newline()
            print('Estimand:', identified_estimand)
        def eststep(model, identified_estimand, estmethod, estmethodparam,verbose=True):
            estimate = model.estimate_effect(identified_estimand,
            method_name=estmethod, test_significance=True,method_params = estmethodparam)
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
            print('Propensity Scores:        ',estimate.__dict__.get('params').get('propensity_scores'))
            pscores = estimate.__dict__.get('params').get('propensity_scores')
            return estimate, pscores
        def refuterbattery(estimate,method,placebo_type2='default',subset_fraction2='default'):
            if placebo_type2 != 'default':
                refute_results=cmodel.refute_estimate(identified_estimand, estimate=estimate,method_name=method, placebo_type=placebo_type2)
            elif subset_fraction2 != 'default':
                refute_results=cmodel.refute_estimate(identified_estimand, estimate=estimate,method_name=method, subset_fraction=subset_fraction2)
            else:
                refute_results=cmodel.refute_estimate(identified_estimand, estimate=estimate,method_name=method)
            print('Refuter Type:             ',refute_results.__dict__.get('refutation_type'))
            print('Estimated Effect:         ',refute_results.__dict__.get('estimated_effect'))
            print('New Effect:               ',refute_results.__dict__.get('new_effect'))
            print('Statistical Significance: ',refute_results.__dict__.get('refutation_result').get('is_statistically_significant'))
            print('P-Value:                  ',refute_results.__dict__.get('refutation_result').get('p_value'))
            newline()

        idparam = identifier
        estmethod = method_name
        estmethodparam = method_params
        unitparam = unit

        if str(type(causalmodel)) == '<class \'dowhy.causal_model.CausalModel\'>':
            cmodel = causalmodel
        else:
            mainmenu()
            return
        if full_output == False:
            identified_estimand = cmodel.identify_effect(proceed_when_unidentifiable=True, method_name=idparam)
            estimandcheck = estcheck(identified_estimand)
            print('Simulator Mode (Default DoWhy) Parameters: ')
            print('Identification Using: identifier=\'',idparam,'\'')
            print('Estimation Using: method_name=\'',estmethod,'\'')
            newline()
            idbreak()
            print('Detected Estimand:',estimandcheck,'- Estimation Options:')
            if estimandcheck == 'iv': 
                ivstatement()
                if method_name == 'default':
                    estmethod = 'iv.instrumental_variable'
                    estmethodparam = None
                elif method_name != 'default' and method_params == 'default':
                    estmethod = 'iv.instrumental_variable'
                    estmethodparam = None
            elif estimandcheck == 'frontdoor': 
                frontdoorstatement()
                estmethod = 'frontdoor.two_stage_regression'
                estmethodparam = None
            elif estimandcheck == 'backdoor': 
                backdoorstatement()
                unitparam = None
                if estmethod == 'default' or estmethod == 'backdoor.generalized_linear_model' or estmethod == 'backdoor.linear_regression':
                    if cmodel._data[cmodel._outcome].isin([0,1]).all()[0] == True:
                        print('Detected binary outcome: Selecting GLM logistic regression')
                        estmethod = 'backdoor.generalized_linear_model'
                        if estmethodparam == 'default':
                            estmethodparam = {'num_null_simulations':10,
                            'num_simulations':10,
                            'num_quantiles_to_discretize_cont_cols':10,
                            'fit_method': "statsmodels",
                            'glm_family': sm.families.Binomial(),
                            'need_conditional_estimates':False}
                    else:
                        print('Detected nonbinary outcome: Selecting OLS linear regression')
                        estmethod = 'backdoor.linear_regression'
                        estmethodparam = None
                elif estmethod == 'backdoor.econml.dr.LinearDRLearner':
                    if estmethodparam == 'default':
                        estmethodparam = {"init_params":{
                            'model_propensity': LogisticRegressionCV(cv=3, solver='lbfgs', multi_class='auto')},
                            "fit_params":{}}
                elif estmethod == 'backdoor.econml.dml.DML':
                    if estmethodparam == 'default':
                        estmethodparam = {'init_params':{
                            'model_y':GradientBoostingRegressor(),
                            'model_t': GradientBoostingRegressor(),
                            'model_final': LassoCV(fit_intercept=False),
                            'featurizer':PolynomialFeatures(degree=1, include_bias=True)},
                            'fit_params':{'inference': BootstrapInference(n_bootstrap_samples=20, n_jobs=-1)}}
                elif estmethod == 'backdoor.propensity_score_stratification' or estmethod == 'backdoor.propensity_score_matching'  or estmethod == 'backdoor.propensity_score_weighting':
                    if cmodel._data[cmodel._outcome].isin([0,1]).all()[0] == False  or  cmodel._data[cmodel._treatment].isin([0,1]).all()[0] == False:
                        unitparam = None
                        print('Failed to detect both binary treatment and binary outcome')
                        print('Reverting Propensity Score Estimation to Regression')
                        if cmodel._data[cmodel._outcome].isin([0,1]).all()[0] == True:
                            print('Detected binary outcome: Selecting GLM logistic regression')
                            estmethod = 'backdoor.generalized_linear_model'
                            if estmethodparam == 'default':
                                estmethodparam = {'num_null_simulations':10,
                                'num_simulations':10,
                                'num_quantiles_to_discretize_cont_cols':10,
                                'fit_method': "statsmodels",
                                'glm_family': sm.families.Binomial(),
                                'need_conditional_estimates':False}
                        else:
                            print('Detected nonbinary outcome: Selecting OLS linear regression')
                            estmethod = 'backdoor.linear_regression'
                            estmethodparam = None
                    elif estmethod == 'backdoor.propensity_score_stratification':
                        unitparam = 'att'
                        print('Propensity Score Unit: ATT')
                        estmethodparam = None
                    elif estmethod == 'backdoor.propensity_score_matching':
                        unitparam = 'atc'
                        print('Propensity Score Unit: ATC')
                        estmethodparam = None
                    elif estmethod == 'backdoor.propensity_score_weighting':
                        unitparam = 'ate'
                        print('Propensity Score Unit: ATE')
                        estmethodparam = {"weighting_scheme":"ips_weight"}
                #else:
                #    unitparam = None
                #    estmethodparam = None
            elif estimandcheck != 'iv' and estimandcheck != 'frontdoor' and estimandcheck != 'backdoor':
                nodag()
                return
            print('Detected Estimator method_name (DoWhy): \'',estmethod,'\'')
            print('Detected Estimator method_params (DoWhy): ',estmethodparam)
            newline()
            estbreak()
            estimate = cmodel.estimate_effect(identified_estimand,
            method_name=estmethod, test_significance=True,method_params = estmethodparam, target_units = unitparam)
            print(estimate)
            print('Propensity Scores:        ',estimate.__dict__.get('params').get('propensity_scores'))
            pscores = estimate.__dict__.get('params').get('propensity_scores')
            newline()
            refbreak()
            if refute==True:
                if estimandcheck == 'iv':
                    refuterbattery(estimate=estimate,method='data_subset_refuter', subset_fraction2=0.9)
                    refuterbattery(estimate=estimate,method='placebo_treatment_refuter', placebo_type2="permute")
                    refuterbattery(estimate=estimate,method='bootstrap_refuter')
                elif estimandcheck != 'iv':
                    refuterbattery(estimate=estimate,method='random_common_cause')
                    refuterbattery(estimate=estimate,method='data_subset_refuter', subset_fraction2=0.9)
                    refuterbattery(estimate=estimate,method='placebo_treatment_refuter', placebo_type2="permute")
                    refuterbattery(estimate=estimate,method='bootstrap_refuter')
            returnstatement()
            checkdat = cmodel._data
            cols_to_keep = [c for c in checkdat.columns if c != 'propensity_score' and c != 'strata' and c != 'dbar' and c != 'd_y' and c != 'dbar_y']
            cmodel._data = cmodel._data[cols_to_keep]
            return estimate, pscores

        elif full_output == True:
            identified_estimand = cmodel.identify_effect(proceed_when_unidentifiable=True, method_name=idparam)
            estimandcheck = estcheck(identified_estimand)
            print('Simulator Mode (Full) Parameters: ')
            print('Identification Using: identifier=\'',idparam,'\'')
            print('Estimation Using: method_name=\'',estmethod,'\'')
            newline()
            idbreak()
            print('Detected Estimand:',estimandcheck,'- Estimation Options:')
            if estimandcheck == 'iv': 
                ivstatement()
                if method_name == 'default':
                    estmethod = 'iv.instrumental_variable'
                    estmethodparam = None
                elif method_name != 'default' and method_params == 'default':
                    estmethod = 'iv.instrumental_variable'
                    estmethodparam = None
            elif estimandcheck == 'frontdoor': 
                frontdoorstatement()
                estmethod = 'frontdoor.two_stage_regression'
                estmethodparam = None
            elif estimandcheck == 'backdoor': 
                backdoorstatement()
                unitparam = None
                if estmethod == 'default' or estmethod == 'backdoor.generalized_linear_model' or estmethod == 'backdoor.linear_regression':
                    if cmodel._data[cmodel._outcome].isin([0,1]).all()[0] == True:
                        print('Detected binary outcome: Selecting GLM logistic regression')
                        estmethod = 'backdoor.generalized_linear_model'
                        if estmethodparam == 'default':
                            estmethodparam = {'num_null_simulations':10,
                            'num_simulations':10,
                            'num_quantiles_to_discretize_cont_cols':10,
                            'fit_method': "statsmodels",
                            'glm_family': sm.families.Binomial(),
                            'need_conditional_estimates':False}
                    else:
                        print('Detected nonbinary outcome: Selecting OLS linear regression')
                        estmethod = 'backdoor.linear_regression'
                        estmethodparam = None
                elif estmethod == 'backdoor.econml.dr.LinearDRLearner':
                    if estmethodparam == 'default':
                        estmethodparam = {"init_params":{
                            'model_propensity': LogisticRegressionCV(cv=3, solver='lbfgs', multi_class='auto')},
                            "fit_params":{}}
                elif estmethod == 'backdoor.econml.dml.DML':
                    if estmethodparam == 'default':
                        estmethodparam = {'init_params':{
                            'model_y':GradientBoostingRegressor(),
                            'model_t': GradientBoostingRegressor(),
                            'model_final': LassoCV(fit_intercept=False),
                            'featurizer':PolynomialFeatures(degree=1, include_bias=True)},
                            'fit_params':{'inference': BootstrapInference(n_bootstrap_samples=20, n_jobs=-1)}}
                elif estmethod == 'backdoor.propensity_score_stratification' or estmethod == 'backdoor.propensity_score_matching'  or estmethod == 'backdoor.propensity_score_weighting':
                    if cmodel._data[cmodel._outcome].isin([0,1]).all()[0] == False  or  cmodel._data[cmodel._treatment].isin([0,1]).all()[0] == False:
                        unitparam = None
                        print('Failed to detect both binary treatment and binary outcome')
                        print('Reverting Propensity Score Estimation to Regression')
                        if cmodel._data[cmodel._outcome].isin([0,1]).all()[0] == True:
                            print('Detected binary outcome: Selecting GLM logistic regression')
                            estmethod = 'backdoor.generalized_linear_model'
                            if estmethodparam == 'default':
                                estmethodparam = {'num_null_simulations':10,
                                'num_simulations':10,
                                'num_quantiles_to_discretize_cont_cols':10,
                                'fit_method': "statsmodels",
                                'glm_family': sm.families.Binomial(),
                                'need_conditional_estimates':False}
                        else:
                            print('Detected nonbinary outcome: Selecting OLS linear regression')
                            estmethod = 'backdoor.linear_regression'
                            estmethodparam = None
                    elif estmethod == 'backdoor.propensity_score_stratification':
                        unitparam = 'att'
                        print('Propensity Score Unit: ATT')
                        estmethodparam = None
                    elif estmethod == 'backdoor.propensity_score_matching':
                        unitparam = 'atc'
                        print('Propensity Score Unit: ATC')
                        estmethodparam = None
                    elif estmethod == 'backdoor.propensity_score_weighting':
                        unitparam = 'ate'
                        print('Propensity Score Unit: ATE')
                        estmethodparam = {"weighting_scheme":"ips_weight"}
            elif estimandcheck != 'iv' and estimandcheck != 'frontdoor' and estimandcheck != 'backdoor':
                nodag()
                return
            print('Detected Estimator method_name (DoWhy): \'',estmethod,'\'')
            print('Detected Estimator method_params (DoWhy): ',estmethodparam)
            newline()
            estbreak()
            idstep(identified_estimand=identified_estimand)
            estimate = eststep(model=cmodel,identified_estimand=identified_estimand,estmethod=estmethod,estmethodparam=estmethodparam)
            newline()
            refbreak()
            if refute==True:
                if estimandcheck == 'iv':
                    refuterbattery(estimate=estimate[0],method='data_subset_refuter', subset_fraction2=0.9)
                    refuterbattery(estimate=estimate[0],method='placebo_treatment_refuter', placebo_type2="permute")
                    refuterbattery(estimate=estimate[0],method='bootstrap_refuter')
                elif estimandcheck != 'iv':
                    refuterbattery(estimate=estimate[0],method='random_common_cause')
                    refuterbattery(estimate=estimate[0],method='data_subset_refuter', subset_fraction2=0.9)
                    refuterbattery(estimate=estimate[0],method='placebo_treatment_refuter', placebo_type2="permute")
                    refuterbattery(estimate=estimate[0],method='bootstrap_refuter')
            returnstatement()
            checkdat = cmodel._data
            cols_to_keep = [c for c in checkdat.columns if c != 'propensity_score' and c != 'strata' and c != 'dbar' and c != 'd_y' and c != 'dbar_y']
            cmodel._data = cmodel._data[cols_to_keep]
        else:
            mainmenu()

    def makegraph(function='main',edges='edges',digraph='graph',dataset='dataset',treatment='treatmentX0',outcome='outcomeY0',model='model',eda=False,verbose=True):
        def newline():
            print('')
        def menubreak():
            newline()
            print('____________________________________________')
        def option1():
            print('1.')
        def option2():
            print('2.')
        def helpmenu0():
            print('CausalFast makegraph() Assistance')
        def helpmenu1():
            print('   DoWhy CausalModel Maker:')
            print('   Create a DAG for use in the CausalModel object')
            print('   Syntax: makegraph(function=\'makecausalmodel\', edges=edgelist, verbose=True)')
            newline()
            print('   Create a CausalModel for causal analysis in CausalFast Simulator or DoWhy')
            print('   Syntax: makegraph(function=\'makecausalmodel\',digraph=\'graph\',dataset=\'dataset\',treatment=\'treatment\',outcome=\'outcome\')')
            newline()
        def helpmenu2():
            print('   Tutorial Mode:')
            print('   This generates DoWhy CausalModels of the three estimand criterion for use in the simulator')
            print('   Syntax: makegraph(function=\'tutorial\', model=\'backdoor\')')
            print('   Syntax: makegraph(function=\'tutorial\', model=\'frontdoor\')')
            print('   Syntax: makegraph(function=\'tutorial\', model=\'iv\')')
            newline()
        def causalmodelmenu0():
            print('CausalFast makegraph() Causal Model Maker')
        def causalmodelmenu1():
            print('CausalFast makegraph() Tutorials')
        def variablelabels():
            print('treatment:              \'X\'')
            print('outcome:                \'Y\'')
            print('(un)observed-confounder:\'U\'')
        def edalabel(data,edalen=False):
            print('Basic EDA:')
            print('Dimensionality: ', data.shape)
            print('Column Names:', list(data.columns))
            if edalen==True:
                newline()
                print(data.head())
            newline()
            print('Rounded - 2 decimals / Numeral Length - CtNa:4, Min/Avg/Max:11')
            for (columnName, columnData) in data.items():
                naopt = str(data[columnName].isna().sum())
                nodat = '--      '
                if len(naopt) <= 4:
                    naopt = naopt + (" " * (4-len(naopt)))
                elif len(naopt) > 4:
                    naopt = naopt[:4]
                dtopt = str(data[columnName].dtype)
                if len(dtopt) <= 7:
                    dtopt = dtopt + (" " * (7-len(dtopt)))
                elif len(dtopt) > 7:
                    dtopt = dtopt[:7]
                minopt = str(round(min(columnData),2))
                if len(minopt) <= 11:
                    minopt = minopt + (" " * (11-len(minopt)))
                elif len(minopt) > 11:
                    minopt = minopt[:11]
                avgopt = str(round(columnData.mean(),2))
                if len(avgopt) <= 11:
                    avgopt = avgopt + (" " * (11-len(avgopt)))
                elif len(avgopt) > 11:
                    avgopt = avgopt[:11]
                maxopt = str(round(max(columnData),2))
                if len(maxopt) <= 11:
                    maxopt = maxopt + (" " * (11-len(maxopt)))
                elif len(maxopt) > 11:
                    maxopt = maxopt[:11]
                if is_numeric_dtype(data[columnName]) and data[columnName].isna().sum() == 0:
                    print(dtopt,'CtNA:',naopt,'MIN:', minopt,'AVG:',avgopt,'MAX:',maxopt,'Col:',columnName)
                else:
                    print(dtopt,'CountNA:',naopt,'MIN:',nodat,'AVG:',nodat,'MAX:',nodat,'Col:',columnName)
            menubreak()
        def startstring():
            print('Start of Digraph string:')
        def endstring():
            print(':End of Digraph string.')
        def stringcomment():
            print('Note: Copy the entire string, including start \"\"\" and end \"\"\".')
            print('Note: Do not wrap this string in quotes when assigning it to a varible for use in DoWhy CausalModel')
        def returnstatement():
            print('Returned: makecausalmodel[0] = CausalModel, makecausalmodel[1] = Digraph string, makecausalmodel[2] = dataset')
        def makemodelfailed():
            print('   Syntax: makegraph(function=\'makecausalmodel\', dataset=dataset, digraph=\'graph\', treatment=\'treatment\', outcome=\'outcome\', eda=True, verbose=True)')
            newline()
            print('   Note: Creating a DoWhy CausalModel object requires the following 4 parameters:')
            print('      dataset   - A pandas dataframe of your causal data')
            print('      digraph   - A string containing digraph information in DOT format (must match dataset variables)')
            print('      treatment - A string containing the treatment variable name (must match digraph and dataset variable)')
            print('      outcome   - A string containing the outcome variable name (must match digraph and dataset variable)')
        def causalmodeldesc():
            option1()
            print('   Create a Directed Acyclic Graph / DAG using NetworkX to visualize a causal system')
            print('   Returns a digraph as a string object used to build the casualmodel object')
            newline()
            print('   Provide the following parameters (as a list object): edges')
            print('   edgelist = [(\'X\',\'Y\'),(\'U\',\'X\'),(\'U\',\'Y\')]')
            newline()
            print('   Syntax: makegraph(function=\'makecausalmodel\', edges=edgelist, verbose=True)')
            newline()
            print('   Note: the \'digraph=graph\' object should be parameterized to a string object of the pattern: ')
            n = 7
            strValue ="""digraph = \"\"\"digraph {
            U;
            X;
            Y;
            U -> X;
            U -> Y;
            X -> Y;
            }
            \"\"\""""
            print(strValue)
            newline()
            option2()
            print('   Create a CausalModel object used by DoWhy or CausalFast Simulator')
            print('   Perform EDA on Data and/or returns a CausalModel object')
            newline()
            makemodelfailed()
            newline()

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
                if verbose == True:
                    causalmodelmenu1()
                    menubreak()
                    print('This graph shows the backdoor criterion.')
                    variablelabels()
                    newline()
                if eda == True:
                    edalabel(data=data,edalen=True)
                if verbose == True:
                    startstring()
                    n = 7
                    replacementStr = '\"\"\"digraph'
                    strValue2 = replacementStr + strValue[n:]
                    endstr = '\"\"\"'
                    strValue2 = strValue2 + endstr
                    print(strValue2)
                    endstring()
                    newline()
                    stringcomment()
                cmodel = CausalModel(data=data,treatment='X',outcome='Y',graph=strValue)
                cmodel.view_model()
                newline()
                returnstatement()
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
                if verbose == True:
                    causalmodelmenu1()
                    menubreak()
                    print('This graph shows the frontdoor criterion.')
                    variablelabels()
                    print('unobserved confounder:  \'Z\'')
                    newline()
                if eda == True:
                    edalabel(data=data,edalen=True)
                if verbose == True:
                    startstring()
                    n = 7
                    replacementStr = '\"\"\"digraph'
                    strValue2 = replacementStr + strValue[n:]
                    endstr = '\"\"\"'
                    strValue2 = strValue2 + endstr
                    print(strValue2)
                    endstring()
                    newline()
                    stringcomment()
                cmodel = CausalModel(data=data,treatment='X',outcome='Y',graph=strValue)
                cmodel.view_model()
                newline()
                returnstatement()
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
                if verbose == True:
                    causalmodelmenu1()
                    menubreak()
                    variablelabels()
                    print('instrumental variable:  \'Z\'')
                    newline()
                if eda == True:
                    edalabel(data=data,edalen=True)
                if verbose == True:
                    startstring()
                    n = 7
                    replacementStr = '\"\"\"digraph'
                    strValue2 = replacementStr + strValue[n:]
                    endstr = '\"\"\"'
                    strValue2 = strValue2 + endstr
                    print(strValue2)
                    endstring()
                    newline()
                    stringcomment()
                cmodel = CausalModel(data=data,treatment='X',outcome='Y',graph=strValue)
                cmodel.view_model()
                newline()
                returnstatement()
                return cmodel, strValue, data
            else:
                helpmenu0()
                option1()
                helpmenu1()
                option2()
                helpmenu2()

        elif function == 'makecausalmodel':
            if verbose == True:
                causalmodelmenu0()
                menubreak()
            if isinstance(dataset, pd.DataFrame):
                if eda == True:
                    edalabel(data=dataset,edalen=False)
                if digraph != 'graph' and treatment != 'treatmentX0' and outcome != 'outcomeX0':
                    strValue = digraph
                    n = 7
                    replacementStr = '\"\"\"digraph'
                    strValue2 = replacementStr + strValue[n:]
                    endstr = '\"\"\"'
                    strValue2 = strValue2 + endstr
                    if verbose == True:
                        print('Dataframe Detected')
                        print('Digraph Detected')
                        print('Treatment: ', treatment, '  (Verify your treatment parameter with the Digraph and Dataset column name)')
                        print('Outcome: ', outcome, '  (Verify your outcome parameter with the Digraph and Dataset column name)')
                        newline()
                        startstring()
                        print(strValue2)
                        endstring()
                        newline()
                        stringcomment()
                        newline()
                    cmodel = CausalModel(data=dataset,treatment=treatment,outcome=outcome,graph=digraph)
                    cmodel.view_model()
                    newline()
                    returnstatement()
                    strValue2 = strValue2[3:-3]
                    return cmodel, strValue2, dataset
                else:
                    print('Insufficient Parameters Supplied:')
                    newline()
                    makemodelfailed()
                    return
            if isinstance(edges, str):
                if edges == 'edges':
                    causalmodeldesc()
                else:
                    print('Edges not detected: Unable to create a digraph')
                    print('edges= \"',edges,'\"')
                    newline()
                    print('You inputted a string object. Please input a list object of the format:')
                    print('edgelist = [(\'X\',\'Y\'),(\'U\',\'X\'),(\'U\',\'Y\')]')
                    newline()
                    print('Syntax: makegraph(function=\'makecausalmodel\',edges=edgelist)')
                    newline()
                return
            if isinstance(edges, list):
                if verbose == True:
                    print('Note: Variables names (data,digraph,causalmodel) must precisely match or DoWhy cannot estimate')
                    newline()
                    print('Edges List: ')
                    print(edges)
                    newline()
                seed = 42
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
                newline()
                if verbose == True:
                    startstring()
                    newline()
                    print(strValue)
                    endstring()
                    newline()
                    stringcomment()
                    print('Note: Please verify the directionality of the digraph string')
                    newline()
                    print('Returned: Digraph string')
                strValue = strValue[3:-3]
                return strValue 
        else:
            helpmenu0()
            menubreak()
            option1()
            helpmenu1()
            menubreak()
            option2()
            helpmenu2()
            menubreak()
