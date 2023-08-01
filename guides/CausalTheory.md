# Primer of Causal History and Theory
### For data scientists to more easily learn causal analysis
```diff
Table of Contents:

1.   Lesson 1: Causality Defined
2.   Lesson 2: Correlation and Causation
3.   Lesson 3: Pearl's Theory of Causation
4.   Lesson 4: Causal Models
5.   Lesson 5: Graphical Causal Models
6.   Lesson 6: Causal Effects Estimation
7.   Lesson 7: DoWhy + EconML
8.   Lesson 8: Causal Modeling in Python Using DoWhy, EconML, CausalFast
```
### Lesson 1: Causality Defined

For centuries philosophers and scientists have considered the meaning of causal reasoning. In Plato’s Timaeus it is written “everything that becomes or changes must do so owing to some cause; for nothing can come without a cause”. Since then, many philosophers including David Hume and Immanuel Kant have made contributions the concept of cause, which ultimately can be separated into two dominant conceptions:
1.	Aristotle's interpretation: Metaphysical cause, to understand why phenomena occurs we must understand its 1) physical nature, 2) its purpose, 3) method of change, 4) the outcome <sup><sub>[1]</sup></sub>
2)	Scientific interpretation: Causes are nodes in law-like chain: A causes B if, given B, A must have occurred <sup><sub>[2]</sup></sub>

**_Causality, causation, or cause and effect_** is the influence of one state on the production of another state where the cause has partial responsibility for the effect. A process may have numerous causes in its past.

**_Causal inference_** is the process of determining the effects of a phenomenon within a system. The gold standard for determining causal effect is through the use of randomized controlled trials (RCT). However, RCT are costly and complex and in many situations may be impossible to perform. When an RCT cannot be performed, a natural experiment/observational study may provide data that can be used for causal analysis.

**_Causal analysis_** can be thought of as the practice of applying experimental design and statistics to establish cause and effect. Causal analysis can be performed on observational data. However, due to issues such as confounding, quasi-experimental approaches using statistics require assumptions to produce ‘good’ estimates with observational data.

### Lesson 2: Correlation and Causation
**_"Correlation does not imply causation"_** is a phrase most people have heard before, and the reference is that association between two variables, based on correlation alone, does not establish a cause-and-effect relationship. Indeed, **_spurious correlations_** can exist which are not causal. For this reason, while association can be used in statistical models for prediction or classification, it is not sufficient for use in high assurance situations where decisions are being made based on reliable information. There are four main causal elements: 
1.  Correlation
2.  A sequence of events, with the cause before effect
3.  A plausible mechanism for an effect to follow the cause
4.  The ability to eliminate common or alternative causes (special causes).

Correlation, in a causal context, that is correlation between variables with a causal sequence and mechanism, can be measured and modelled with the correct data, and when these assumptions are met it is possible to obtain reasonable answers to causal questions. Loosely defined causal models based on these four main causal elements may provide valuable insight. However, from a epidemiological perspective the four main causal elements alone would not be sufficient. An example of causal theory applied to epidemiology would be the Bradford Hill Criteria for causation. Hill proposed 9 'aspects of association' that establish epidemiologic evidence of a causal relationship between a cause and an observed effect. These nine criteria are:
1.  Strength (effect size, small associations can be causal, but strong associations has greater likelihood)
2.  Consistency (reproducibility)
3.  Specificity (no other likely explanation)
4.  Temporality (the effect must occur after the cause)
5.  Biological Gradient (greater exposure should generally lead to greater effects)
6.  Plausibility (knowledge of the causal mechanism is helpful)
7.  Coherence (between epidemiological and laboratory findings)
8.  Experiment (If ethical experiments can be used to demonstrate a causality)
9.  Analogy (similarities between observed association and other associations) <sup><sub>[3]</sup></sub>

### Lesson 3: Pearl's Theory of Causation
**_Judea Pearl_** is a computer scientist who developed probabilistic Bayesian Networks, but he also contributed to causality through development of structural models and do-calculus, a notation for describing causal relationships. According to Pearl, statistical expressions and causal expressions should not be defined in terms of the other, and causal relationships require their own mathematical notation. An example is that while statistical dependence has a notation based in probability, such as P(disease|symptom), this expression is insufficient to quantify the causal dependence as there is no causal expression in probability calculus. This lack of consensus regarding causal theory and notation was the main barrier to acceptance of causal analysis among statistical professionals. <sup><sub>[4]</sup></sub>

Pearl described causality as having three levels and he called this **_The Ladder of Causation_**.
This is an example about the causal relationships between a buyer shopping for toothpaste and floss.
Bottom level: Association (Seeing): What does a symptom tell me about a disease?
Middle Level: Intervention (Intervening): What if I take aspirin, will my headache be cured?
Highest level: Counterfactual (Retrospection): Was it aspirin that stopped my headache? <sup><sub>[5]</sup></sub>

Pearl also created a general theory of causation satisfies the following conditions:
1.  Have a mathematical language that represents causal questions
2.  A precise language to describe assumptions necessary to answer causal questions
3.  Able to systematically answer these questions or label others as ‘unanswerable’
4.  Able to determine what assumptions or information is needed to answer ‘unanswerable questions’
5.  Subsume all other theories that explore causation: become the foundational theory that unifies special cases <sup><sub>[4]</sup></sub>

### Lesson 4: Causal Models
Generally speaking, a causal model for analysis would be a is a framework for describing the causal mechanism in a system using quantative methods. Causal models have rules (assumptions) that allow researchers to answer causal questions from observational data, but the rules of a model will be different depending on the data and causal question. **_The Rubin Causal Model (RCM)_** is describes the approach to quantitatively analyze cause and effect using a **_Potential Outcomes Framework_**. The potential outcomes framework is a counterfactual conditional model to determine what would an outcome have been if the cause or treatment had been different or intervened on. With observational data alone it is impossible to know for certain what any other potential outcome would have been. This is known as the **_Fundamental Problem of Causal Inference_**. And while this is true for unit level analysis, the use of randomized experiments on a sample/population can estimate an average causal effect between two groups (treatment & control). This estimate is known as the **_Average Treatment Effect (ATE)_**. <sup><sub>[6]</sup></sub>

Whichever causal model is used for inference will have its own assumptions. Generally speaking, all observations must have the possibility of receiving both treatments. Also, there should be a **_Stable Unit Treatment Value Assumption (SUTVA)_** which implies that the assignment of treatment and outcome for one observation should not have an effect on the assignment of treatment to other observations. This could be thought of as unobserved confounders. Example: Assigning treatment causes Person A to change their behavior, and this has an effect on Person B which changes Person B's covariates that affects treatment assignment. Causal assumptions vary depending on the causal questions and approach for data analysis. Additional common assumptions to observational studies: Indentifiability, Exchangability, Positivity, Consistency. <sup><sub>[7]</sup></sub>

### Lesson 5: Graphical Causal Models
DoWhy seems to have been inspired by Judea Pearl's **_Structural Causal Model (SCM)_** which have 3 constitutant parts: **_Exogenous Variables_**, **_Endogenous Variables_** and **_Structural Equations_**. Exogenous variables are independent whereas endogenous variables are dependent. The structural equations are used to describe the causal effect between each variable. In DoWhy, these SCM are encoded into a graph model. Graphical Causal Models (GCM)_** are generally depicted as **_Directed Acyclic Graphics (DAGs)_** which have causal assumptions encoded within the connections between parent and ancestor nodes. Nodes are linked by arrows that display the direction of information flows between the nodes. These causal assumptions are:
1.  The graph must be a DAG
2.  Must satisfy the Causal Markov condition
3.  Must have faithfulness/conditional independence<br>

A DAG is a required assumption because causal relationships are one-directional while statistical relationships are two-directional. The DAG specifies the direction of causal information, and this is used to determine the causal estimand, the causal question (covariates, treatment, outcome). The Causal Markov condition states that all nodes must be independent of all other variables which are not its own parent/ancestor. Last, the faithfulness condition states that there must be no special independence between variables. In a rare situation where two variables that would otherwise have a dependent relationship, but instead have causal effects that cancel eachother out to become independent, would still be a violation of the markov condition. Simply put, there can only be conditional independence as implied by the markov condition and through no other special case.

These three assumptions are encoded into the GCM which is then analyzed by DoWhy. DoWhy looks at the structures within the DAG and determines which variables need to be controlled for to maintain conditional independence so as to guarantee that causal information is flowing one-directionally. This allows DoWhy to generate a reasonable causal estimate using regression. When the paths between dependent variables are controlled for the DAG is said to have become **_d-separated_**. It is not always intuitive to recognize the best variable to control for, and in some cases there may be options of variables that can be controlled for in a single DAG to achieve conditional independence, but it is necessary to ensure that the causal estimand can be identified before beginning estimation. There are three types of structures that must be considered:<br><br>
![alt text](https://raw.githubusercontent.com/TejuOye/CausalFast/main/api/images/chain_small.png "Chain")<br>
This is a chain structure showing X and Z that can become conditionally independent after controlling on Y.
<br><br>
![alt text](https://raw.githubusercontent.com/TejuOye/CausalFast/main/api/images/fork_small.png "Fork")<br>
This is fork structure also shows X and Z can become conditionally independent after controlling on Y, and Y can be thought of as a common cause of X and Z.
<br><br>
![alt text](https://raw.githubusercontent.com/TejuOye/CausalFast/main/api/images/collider_small.png "Collider")<br>
This collider structure is different from chains and forks in that X and Z have no parent node, and so X and Z are already conditionally independent since it is not possible for information to propogate from X to Y.
<br>

### Lesson 6: Propensity Score Matching
Since RCT are often impossible to create, observational data alone can be used to determine the effect of a treatment, policy, or intervention through **_Propensity Score Matching_**. Propensity score matching uses statistical approaches to control for bias in the covariates that predict receiving treatment. Bias occurs due to ‘confounding variables’ that have an effect on the outcome and are associated with both the outcome and the treatment. That is to say, the outcome may be caused by something that predicts treatment rather than caused by the treatment itself. To control for this bias, a method of regression on the covariates to the treatment (X to T), without including outcome (Y), is performed, and this results in propensity scores (measure of likelihood to receive treatment). This method commonly uses logistic regression on a binary treatment variable. <br>

Using this propensity score, it is possible to match similar propensity scores in the treatment and control groups. This can be thought of as a dimension reduction technique as it is generally sufficient to match observations based on propensity scores rather than to match based on numerous covariates. By matching propensity scores, the covariate distributions of treatment and control groups should be balanced. Then, using this balanced data, it is possible to calculate the average outcome for both the treatment and control group. By taking the difference of these average outcome values we can obtain a reasonably good ATE. This is based on the assumption that similarly matched covariates will have similar outcomes (We've removed as much bias as possible).

While randomization in a RCT will balance out this bias, observational studies do not have truly random assignment of treatment to subjects. However, with a large enough observational data, and the right causal data at that, of subjects that received treatment v subjects that did not receive treatment, it is possible to obtain a reasonably good causal estimate. While this is true for low precision uses, in high precision situations or when the causal pathway is unknown, such as is commonly seen in epidemiology, there will generally be more experimentation needed to obtain reliable estimates. <sup><sub>[8]</sup></sub>

### Lesson 7: DoWhy + EconMl
DoWhy is the premier python library for causal inference using graphical causal models. It has become popular due to its simple four step process of modeling and testing causal assumptions. The four steps of causal inference in DoWhy are:
1)	Create a causal model from data and a graph
2)	Identify the causal effect in the model and return the estimand
3)	Estimate the target estimand using a statistical method
4)	Refute the obtained estimate to determine robustness of the estimate

DoWhy has 3 mainline API. The Causal Inference API is the focus of causalfast, but DoWhy is developing a GCM API to perform more granular experimentation on data encoded into the causal graph. The last mainline API is the Pandas API which utilizes the Pandas package to perform causal analysis. Using the causal inference API, the first step is to build a causal model. This involves a data EDA process and then generating a DAG using domain knowledge of the causal paths between variables. When this causal model is created, DoWhy then performs the identification step. Identification uses algorithms to evalaute the DAG based on encoded causal assumptions within the structures, and DoWhy will declare the DAG to belong to at least of of three main estimation methods ('Estimand'). These are the Backdoor Estimand, The Frontdoor Estimand, and Instrumental Variable.<br><br>
![alt text](https://raw.githubusercontent.com/TejuOye/CausalFast/main/api/images/backdoor_small.png "Backdoor / Frontdoor")<br>
![alt text](https://raw.githubusercontent.com/TejuOye/CausalFast/main/api/images/IV_small.png "Instrumental Variable")<br>
The Backdoor Estimand 

### Lesson 8: Causal Analysis in Python Using DoWhy, EconML, CausalFast
->>Packages/libraries/specification or system requirements: Python, Dowhy, EconMl, CausalFast
->>Link to Jupyter Notebooks

<br><br>
Further Reading:<br>
[DoWhy User Guide](https://www.pywhy.org/dowhy/main/user_guide/intro.html)
<br>
[Hernan & Robins - What If: Causal Inference](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/) - Harvard University, T.H. Chan School of Public Health

References:
[<sup><sub>[1]</sup></sub>](https://plato.stanford.edu/entries/aristotle-causality/) 
[<sup><sub>[2]</sup></sub>](https://see.library.utoronto.ca/SEED/Vol4-2/Hulswit.htm) 
[<sup><sub>[3]</sup></sub>](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4589117/) 
[<sup><sub>[4]</sup></sub>](https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf) 
[<sup><sub>[5]</sup></sub>](https://ftp.cs.ucla.edu/pub/stat_ser/r481.pdf) 
[<sup><sub>[6]</sup></sub>](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4782596/pdf/nihms737705.pdf) 
[<sup><sub>[7]</sup></sub>](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/) 
[<sup><sub>[8]</sup></sub>](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2943670/pdf/nihms200640.pdf) 
