# Primer of Causal History and Theory
```diff
1.   Lesson 1: Causality Defined
2.   Lesson 2: Correlation and Causation
```
### Lesson 1: Causality Defined

For centuries philosophers and scientists have considered the meaning of causal reasoning. In Plato’s Timaeus it is written “everything that becomes or changes must do so owing to some cause; for nothing can come without a cause”. Since then, many philosophers including David Hume and Immanuel Kant have made contributions the concept of cause, which ultimately can be separated into two dominant conceptions:
1.	Aristotelian interpretation: Causes are the active initiators of change
  a.	A is the cause of B means A initiates change in B
2)	Scientific interpretation: Causes are inactive nodes in a law-like implication chain
  a.	A is the cause of B if, given the occurrence of B, A must have occurred

**_Causality, causation, or cause and effect_** is the influence of one process/state on the production of another process/state where the cause is partially responsible for the effect, and the effect is partially responsible for the cause, and generally a process has several causes.

**_Causal inference_** is the process behind determining the effects of a phenomenon within a larger system. The most effective way to determine the causal effect is through the use of randomized controlled trials (RCT). As RCT are costly and complex, and in some situations may be impossible to perform. When an RCT cannot be performed a natural experiment/observational study may provide inference. 

**_Causal analysis_** can be thought of as the practice of applying experimental design and statistics to establish cause and effect. There are usually four elements: correlation, a sequence of cause before effect, a plausible mechanism for an effect to follow the cause, the ability to eliminate common or alternative causes (special causes). Causal analysis can be performed by observational studies, but due to issues such as confounding, quasi-experimental approaches using statistics require assumptions to produce ‘good’ estimates with observational data. 

### Lesson 2: Correlation and Causation
Judea Pearl is a computer scientist who developed probabilistic Bayesian Networks, but has also contributed to causality through development of structural models and do-calculus, a notation for describing causal relationships. According to Pearl, statistical expressions and causal expressions should not be defined in terms of the other, and causal relationships required its own mathematical notation. An example is that while statistical dependence has a notation based in probability, such as P(disease|symptom), this expression is insufficient to quantify the causal dependence as there is no causal expression in probability calculus. This lack of consensus regarding causal theory and notation was the main barrier to acceptance of causal analysis among statistical professionals.
Pearl described causality as having three levels and he called this ‘The Ladder of Causation’.
This is an example about the causal relationships between a buyer shopping for toothpaste and floss.
Lowest level: Association P(floss|toothpaste)
Ex: Shoppers who buy toothpaste are more likely to buy dental floss
Mid Level: Intervention P(floss|do(toothpaste))
Ex: Probability of buying floss given an intervention on toothpaste price. (keep in mind, price cannot be used alone to estimate probability, since a confounder, such as a tariff, could increase the price of toothpaste and floss together).
Highest level: Counterfactual P(floss|toothpaste,price*2)
Ex: Probability of buying floss given the store doubled the price of toothpaste?

**_Pearl_** proposed that a general theory of causation must meet five conditions:
1.  Have a mathematical language that represents causal questions
2.  A precise language to describe assumptions necessary to answer causal questions
3.  Able to systematically answer these questions or label others as ‘unanswerable’
4.  Able to determine what assumptions or information is needed to answer ‘unanswerable questions’
5.  Subsume all other theories that explore causation: become the foundational theory that unifies special cases

**_"Correlation does not imply causation"_** is a phrase most people have heard before, and the reference is that association between two variables, based on correlation alone, does not establish a cause-and-effect relationship. Indeed, spurious correlations can exist which are not causal. For this reason, while association alone can be used in statistical models for prediction or classification, it is not sufficient for use in situations where decisions must be made based on reliable information. An example of causal theory applied to epidemiology would be the Bradford Hill Criteria for causation. 

Hill proposed 9 principles that establish epidemiologic evidence of a causal relationship between a cause and an observed effect. These nine criteria are:
1.  Strength (effect size, small associations can be causal, but strong associations has greater likelihood)
2.  Consistency (reproducibility)
3.  Specificity (no other likely explanation)
4.  Temporality (the effect must occur after the cause)
5.  Biological Gradient (greater exposure should generally lead to greater effects)
6.  Plausibility (knowledge of the causal mechanism is helpful)
7.  Coherence (between epidemiological and laboratory findings)
8.  Experiment (If ethical experiments can be used to demonstrate a causality)
9.  Analogy (similarities between observed association and other associations)

**_The Rubin Causal Model_** is an approach to quantitatively analyze cause and effect using a **_Potential Outcomes Framework_**. The potential outcomes framework is a counterfactual conditional model to determine what would an outcome have been if the cause or treatment had been different or intervened on. With observational data, it is impossible to know for certain what any other potential outcome would have been. This is known as the **_Fundamental Problem of Causal Inference_**. And while this is true for unit level analysis, the use of randomized experiments at the population level can estimate an average causal effect between two groups. This estimate is known as the **_Average Treatment Effect_** (ATE).

Since RCT are frequently impossible to create, observational data can be used to determine the effect of a treatment, policy, or intervention through **_Propensity Score Matching_**. Propensity score matching uses statistical approaches to control for bias in the covariates that predict receiving treatment. Bias occurs due to ‘confounding variables’ that have an effect on the outcome and are associated with both the outcome and the treatment. That is to say, the outcome may be caused by something that predicts treatment rather than caused by the treatment itself. Randomization in RCT will generally balance out this bias, but observational studies do not have truly random assignment of treatment to subjects. However, with a large enough observational data of subjects that received treatment v subjects that did not receive treatment, it is possible to use ‘matching’ to reduce bias due to confounding. Matching is a quasi-experimental approach to reduce bias by matching treated units with untreated units that have similar covariates in order to compare and estimate outcome. 

In computer science, causal and effect using observational data for two variables X -> Y or Y -> X is typically determined by incorporating noise into a model, such as Y = F(X) + E (additive noise). These models have their own assumptions, such that there are no other causes of Y, and that X and E have no common causes, and the distribution of cause is independent from causal mechanisms. 
