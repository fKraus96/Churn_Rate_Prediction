# Churn_Rate_Prediction
Predict customer churn rates for a mobile phone provider using several different state-of-the-art machine learning models. 

The goal of this analysis is to accurately predict whether a customer will churn or not. In this
regard a training dataset of 4000 rows 18 columns (one of them the response variable is provided).
The analysis will be structured as follows. Firstly, some general insights on the data will be provided
together with some descriptive statistics and plots aiming to identify potential non-linear
relationships in the data. Secondly, a thorough walk-through of the analysis, including the different
models and the reasoning for utilisation of these models, as well as the different hypertuning steps
will be provided. Finally, model stacking techniques are discussed and utilised in order to optimize
the final outcome of the analysis. 

Furthermore, a hypertuning class similar to GridSearchCV by Sklearn is developed with the additional functionality of incorporating tuning of the probability threshold for a given model. This is done in order to account for the weights of the different datapoints in the classification task. 
