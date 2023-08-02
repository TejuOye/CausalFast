![Picture of golf clubs on a golf cart overlooking the fairway](https://raw.githubusercontent.com/TejuOye/CausalFast/main/api/images/golfcart.jpg)
<br>
<h3>PGA Tournament Entry: Causal Inference & Effect Estimation</h3><br>

- [`Data Available Here`](https://raw.githubusercontent.com/TejuOye/CausalFast/main/api/data/pga.csv) - Alegre, Canela, Pastoriza (2022). From: [Data in Brief Volume 41](https://www.sciencedirect.com/science/article/pii/S2352340922001639). This dataset contains the determinants for athletes choice to self-select for entry to a tournament endowed with a monetary prize. Use this notebook to perform causal inference and effect estimation using the CausalFast simulator for DoWhy.<br>

<b>Steps:</b>
1) Build a DAG describing variables that effect treatment or outcome.
2) Build a causal model using 'Entry_decision' as the treatment
3) Test different treatment variables to reveal the treatment effect present in the data
4) Practice selecting the proper estimation 'method_name' (Linear Regression v Logistic Regression v ML)
5) Evaluate the refuters to understand their insights
6) Present findings of the determinants for selection into a tournament
 
- Pastoriza, D., Alegre, I., & Canela, M.A. (2021). Conditioning the effect of prize on tournament self-selection. J. Econ. Psychol., 86, 102,414.
- 22 Dimensions: TournamentPrizeMoney, CompetitivenessTournament, AbilityRanking, CumulativeCareerMoney, ExemptNextSeason, Temperature, Injury, etc.
- 54,915 observations (1996-2006)

<br>
<b>Tutorial Notebook:</b> DoWhy Causal Inference & Effect Estimation [`available here`](https://github.com/TejuOye/CausalFast/blob/main/api/notebooks/PGA-InferenceAndEstimation.ipynb) </b>
