# Zillow_Cluster_Regression

## Project Objective 
> Document code, process data (through entire pipeline), and articulate key findings and takeways in a jupyter notebook final report 

* Create modules that faciliate project repeatability, as well as final report readability

> Ask/Answer exploratory questions of data and attributes to understand drivers of home value  

* Utilize charts, statistical tests, and various clustering models to drive linear regression models; improving baseline model

> Construct models to predict `Fraud` (Class)
* Fraud: Fraudulent transaction in the Class Feature
* A observation of fraudulent activites as identified by the Class Feature

> Make recommendations to a *fictional* data science team about how to improve predictions

> Refine work into report in form of jupyter notebook. 

> Present walkthrough of report in 5 minute recorded presentation

* Detail work done, underlying rationale for decisions, methodologies chosen, findings, and conclusions.

> Be prepared to answer panel questions about all project areas

## Project Business Goals
> Construct ML Classification model that accurately predicts `Fraud` using various techniques to create and guide feature selection for modeling</br>

> Find key drivers of `Fraud`</br>

> Deliver report that the data science team can read through and replicate, while understanding what steps were taken, why and what the outcome was.

> Make recommendations on what works or doesn't work in predicting log error, and insights gained from clustering

## Deliverables
> Github repo with a complete README.md, a final report (.ipynb), other supplemental artifacts and modules created while working on the project (e.g. exploratory/modeling notebook(s))</br>
> 5 minute recording of a presentation of final notebook</br>

## Data Dictionary
|       Target             |           Datatype       |     Definition      |
|:-------------------------|:------------------------:|-------------------:|  
Class (Fraud)              | 284807 non-null  int64   | Boolean Classifier for Fraud

|       Feature            |           Datatype       |     Definition      |
|:------------------------|:------------------------:|-------------------:|  
Time                | 284807 non-null  float64  | Time Elapsed since last transaction
V1                  | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V2                  | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V3                  | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V4                  | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V5                  | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V6                  | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V7                  | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V8                  | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V9                  | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V10                 | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V11                 | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V12                 | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V13                 | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V14                 | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V15                 | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V16                 | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V17                 | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V18                 | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V19                 | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V20                 | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V21                 | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V22                 | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V23                 | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V24                 | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V25                 | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V26                 | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V27                 | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
V28                 | 284807 non-null  float64  | Feature name hidden in source due to confidentiality
Amount              | 284807 non-null  float64  | Amount transacted
Indicate            | 284807 non-null  float64  | Feature Engineered, product of RFE determined features
Fraud               | 284807 non-null  Object   | String Values of No-Fraud and Yes-Fraud for Class boolean
cluster V-030409    | 284807 non-null  int32    | Clusters created by KMeans of V03,V04,V09
cluster V-101112    | 284807 non-null  int32    | Clusters created by KMeans of V10,V11,V12
cluster V-141617    | 284807 non-null  int32    | Clusters created by KMeans of V14,V16,V17
if1_anom            | 284807 non-null  int64    | IsolationForest result for anomolies
V141617_3           | 284807 non-null  int64    | Boolean One-Hot Encoding for the Third cluster in V141617
V101112_2           | 284807 non-null  int64    | Boolean One-Hot Encoding for the Second cluster in V101112
V030409_0           | 284807 non-null  int64    | Boolean One-Hot Encoding for the Zero cluster in V030409
-----                    

# Initial Questions and Hypotheses

##  **Hypothesis 1 - Do the sample means of V20, V23, and V28 vary for Yes-Fraud compared to the population?**
> - $H_0$: The mean values of `V20,V23,V28` for the sample `Yes-Fraud` will not be signifcantly different, to the population of `Fraud`.  
> - $H_a$: Rejection of Null ~~The mean values of `V20,V23,V28` for `Yes-Fraud` will not be signifcantly different, to the population of `Fraud`.~~  
>Conclusion: There is enough evidence to reject our null hypothesis only for V20

##  **Hypothesis 2 - Is the distribution of Fraud equal in all Cluster groups?**
> - $H_0$: `Fraud` is not different in the distribution of `Cluster` Groups. 
> - $H_a$: Rejection of Null ~~`Fraud` is not different in the distribution of `Cluster` Groups~~
>Conclusion: There is enough evidence to reject our null hypothesis for all cases

##  **Hypothesis 3 - When divided into Yes-Fraud and No-Fraud, will the mean values of Indicate remain the same?**
> - alpha = .05
> - $H_0$: The mean values of `Indicate` will not be signifcantly different, relating to `Yes-Fraud` and `No-Fraud`.  
> - $H_a$: Rejection of Null ~~The mean values of `Indicate` will not be signifcantly different, relating to `Yes-Fraud` and `No-Fraud`~~  
>Conclusion: There is enough evidence to reject our null hypothesis for this case

##  **Hypothesis 4 - Are the mean values of Each Cluster equal to each other in relation to Fraud Class**
> - alpha = .05
> - $H_0$: The mean values of `Cluster-Sample_n` will not be signifcantly different from `Cluster-Population` relating `Fraud`.  
> - $H_a$: Rejection of Null ~~The mean values of `Cluster-Sample_n` will not be signifcantly different from `Cluster-Population` relating `Fraud`.~~  
> Conclusion: There is enough evidence to reject our null hypothesis for all cases

## Summary of Key Findings and Takeaways
 - Feature `V20` holds value in determining Fraud and will be used in modeling
 - Engineer Feature `Indicate` will also be useful in modeling as a way to predict fraud with it's high deviance depending on Fraud Class
 - We have isolated 3 Clusters from our groups that will also provide benefit and will be utilized in modeling
 - Features to direct in our modeling phase: `V20`, `Indicate`, `V141617.3`, `V101112.2`, `V030409.2`
-----
</br></br></br>

# Pipeline Walkthrough
## Plan
> Create and build out project README  
> Create required as well as supporting project modules and notebooks
* `wrangle.py`, `explore.py`, `model.py`,  `Final_CC.ipynb`
* `creditcard.csv`
> Decide which features to import   
> Decide how to deal with outliers, anomolies 

> Clustering
- Decide on which features to use when crafting clusters
- Create cluster feature sets
- Add cluster labels as features  
> Statistical testing based on clustering
- Create functions that iterate through statistical tests
- Organize in Explore section 
> Explore
- Visualize cluster differences to gauge impact
- Rank clusters based on statistical weight
> Modeling
* Create functions that automate iterative model testing
    - Adjust parameters and feature makes
* Handle acquire, prepare/split and scaling in wrangle
> Verify docstring is implemented for each function within all notebooks and modules 
 

## Acquire
> Acquired csv data from appropriate sources (Kaggle link - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* Create local .csv of raw data upon initial acquisition for later use
* Review data on a preliminary basis
* Set Target Variable
> Add appropriate artifacts into `wrangle.py`

## Prepare
> Univariate exploration: 
* Basic histograms/boxplot for categories 
> Handle any possible threats of data leakage
- Split data (60,20,20)
* Scale data (MinMaxScaler)

> Feature Engineering
* `Indicate`: Feature that contains all #1 Recommendations by RFE
* `Fraud`: String Values mapped to `Class`  
* Cluster modeling: 
> - `cluster V-030409`: `V14`, `V16`, `V17`
> - `cluster V-101112`: `V10`, `V11`, `V12`
> - `cluster V-141617` :|: `V03`, `V04`, `V09`
> - `if1_anom` :|: IsolatedForest model result for anomoly detection
> - `V141617_3 ` :|: `V14`, `V16`, `V17` - Cluster# 3 One-Hot Encoded
> - `V101112_2 ` :|: `V10`, `V11`, `V12` - Cluster# 2 One-Hot Encoded
> - `V030409_0 ` :|: `V03`, `V04`, `V09` - Cluster# 0 One-Hot Encoded
> Collect and collate section *Takeaways*  
> Add appropirate artifacts into `wrangle.py` or `explore.py`

## Explore
> Bivariate exploration
* Investigate and visualize features against `Fraud`
> Identify additional possible areas for feature engineering (clustering)
* Use testing and visualizations to determine which features are significant in determining difference in `Fraud`
> Multivariate:
* Visuals exploring features as they relate to `Fraud`
> Statistical Analysis:
* SpearmanR for continous vs continous
* Two tailed T-Test (Sample vs Population) for discrete vs continous
* Chi^2 for discrete vs discrete
> Collect and collate section *Takeaways*

## Model
> Ensure all data is scaled  
> Create dummy vars of categorical columns (one-hot encoding)
> Set up comparison for evaluation metrics and model descriptions    
> Set Baseline Prediction and evaluate Accuracy and F1 scores  
> Explore various models and feature combinations.
* For initial M.V.P of each model include only single cluster label features
> Choose **Three** Best Models to add to final report

>Choose **one** model to evaluate on Test set
* Decision Tree
* Depth: 4
* Features: As determined by explore section - `V20`, `Indicate`, `V141617.3`, `V101112.2`, `V030409.2`
> Collect and collate section *Takeaways*

## Deliver
> Create project report in form of jupyter notebook  
> Finalize and upload project repository with appropriate documentation 
> Created recorded presentation for delivery
----
</br>

## Project Reproduction Requirements
> Requires personal `creditcard.csv` file containing dataset (obtained in the github or at Kaggle)
> Steps:
* Fully examine this `README.md`
* Download `wrangle.py, explore.py`, `modeling.py`, and `Fina_CC.ipynb` to working directory
* Run `Fina_CC.ipynb`