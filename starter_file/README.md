
# Prediction of Cardiovascular Disease - Azure ML Capstone Project

This is the final project of Udacity Azure ML Nanodegree. As a part of this project we were asked to pick a dataset of our choice. We preprocess it and then use Azure ML studio tools like AutoML and Hyperdrive Configuration to derive the best model based on metrics. We compare the runs and then deploy the best model. We enable applications insights and then test the endpoints. 

<p align="center">
 
 <img src="https://github.com/MonishkaDas/nd00333-capstone/blob/master/starter_file/ScreenShots/heart.png">

</p>

## Project Set Up and Installation

The project was run and deployed on Azure ML studio. For anyone trying to replicate my work, you will need to create an Azure ML instance to run the notebook. The compute target, which in case is named "compute-cluster" is made and is used to run both the AutoML Model (_automl.ipynb_) and the HyperDrive Model (_hyperparameter-tuning.ipynb_). the files can found the repository along with the _train.py_ file that is the entry-script for the HyperDrive Configuration model, the _logs.py_ file to enable application insights and the _endpoint.py_ file to test the endpoints. To use the configurations on a different dataset, the _train.py_ and _endpoint.py_ files need to altered accordingly and the dataset has to be cleaned and processed as required. A few tweaks to the configuration must get you the best possible model that can be then deployed using the code in the  _automl.ipynb_ file.

## Dataset

### Overview

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year. CVDs are a group of disorders of the heart and blood vessels and include coronary heart disease, cerebrovascular disease, rheumatic heart disease and other conditions. Four out of 5CVD deaths are due to heart attacks and strokes, and one third of these deaths occur prematurely in people under 70 years of age.

Individuals at risk of CVD may demonstrate raised blood pressure, glucose, and lipids as well as overweight and obesity. These can all be easily measured in primary care facilities. Identifying those at highest risk of CVDs and ensuring they receive appropriate treatment can prevent premature deaths. Access to essential noncommunicable disease medicines and basic health technologies in all primary health care facilities is essential to ensure that those in need receive treatment and counselling.

### Death Rates due to Cardiovascular diseases in 2017

<p align="center">
 
 <img src="https://github.com/MonishkaDas/nd00333-capstone/blob/master/starter_file/ScreenShots/cardiovascular-disease-death-rates.png">

</p>



This Dataset (https://www.kaggle.com/sulianova/cardiovascular-disease-dataset) is from Kaggle. It's called the Cardiovascular Disease Dataset. The dataset consists of 70 000 records of patients data, 11 features + target.

### Data description
There are 3 types of input features:

1. **Objective:**    factual information

2. **Examination:**  results of medical examination

3. **Subjective:**   information given by the patient


 
### Features:

The following tables gives a detailed description of the dataset and its features.

|Feature| Type| Column Name | DataType |
|--------|-------|-----------|------|
|Age | Objective Feature | age | int (days)|
|Height | Objective Feature | height | int (cm) |
|Weight | Objective Feature | weight | float (kg) |
|Gender | Objective Feature | gender | categorical code |
|Systolic blood pressure | Examination Feature | ap_hi | int |
|Diastolic blood pressure | Examination Feature | ap_lo | int |
|Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal |
|Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal |
|Smoking | Subjective Feature | smoke | binary |
|Alcohol intake | Subjective Feature | alco | binary |
|Physical activity | Subjective Feature | active | binary |
|Presence or absence of cardiovascular disease | Target Variable | cardio | binary |


### Task

The task is to firstly, study the dataset and check for outliers, categorical values, missing values, etc. We then process the dataset and clean it. A well processed dataset gives out better results, this is why data preprocessing is an essential step despite the fact that AutoML has in-built functions to deal with improper data. After processing, we pass the dataset to the configured automl and hyperdrive models. We run the models and monitor the process using the run details widget. We then pick the best model from the two based on metrics, which in this case is the AutoML Model. We deploy the model and run _logs.py_ file to enable Applications insights. Finally, we test the endpoints of the model using the _endpoint.py_ file. We also creat a Onnx model and test it.


### Data Preprocessing

The dataset has a total of 70,000 readings with no missing data. However, there are a few outliers and invalid readings (*For example: a few readings show the diastolic pressure, _ap-lo_ to be higher than the systolic pressure, _ap-hi_*). We drop the outliers and duplicate readings and clean the invalid data, we drop the column _"id"_, we convert the feature column _gender_ to a binary, we replace the categorical columns like _cholestrol, gluc_ with the dummies and we standardize columns like, _age, height, weight, ap-hi, ap-lo_. This data is then passed to the configured experiments and run.

### Access

The dataset is uploaded to the GitHub Repository and accessed through a link using TabularDatasetFactories, it is then converted into pandas dataframes for preprocessing. After Processing the Dataset, it is uploaded to the default datastore, so it can be retrieved as a TabularDataFrame and passed as the _training_data_.

## Automated ML

Parameters like task, primary metric, experiment timeout, training data etc. are passed into the AutoMlConfig to create an optimzed pipeline run on AutoMl that test various models and displays the best ML Algorithm based on the metrics and run time. Around 187 pipelines with different ML Algorithms were run and the best one turned out to be the **Voting Ensemble** with Accuracy of _0.73125_

### Models
![Alt text](https://github.com/MonishkaDas/nd00333-capstone/blob/master/starter_file/ScreenShots/Screenshot%20(103).png?raw=true "Summary Importance")

### Metrics

![Alt text](https://github.com/MonishkaDas/nd00333-capstone/blob/master/starter_file/ScreenShots/Screenshot%20(117).png?raw=true "Summary Importance")

### Feature Importance - bar

![Alt text](https://github.com/MonishkaDas/nd00333-capstone/blob/master/starter_file/ScreenShots/Screenshot%20(104).png?raw=true "Summary Importance")

### Feature Importance - box plot

![Alt text](https://github.com/MonishkaDas/nd00333-capstone/blob/master/starter_file/ScreenShots/Screenshot%20(115).png?raw=true "Summary Importance")

### Feature Importance - swarm plot

![Alt text](https://github.com/MonishkaDas/nd00333-capstone/blob/master/starter_file/ScreenShots/Screenshot%20(113).png?raw=true "Summary Importance")

### Feature Importance - swarm plot with scaled feature value

![Alt text](https://github.com/MonishkaDas/nd00333-capstone/blob/master/starter_file/ScreenShots/Screenshot%20(114).png?raw=true "Summary Importance")


### Explanations 

![Alt text](https://github.com/MonishkaDas/nd00333-capstone/blob/master/starter_file/ScreenShots/Screenshot%20(107).png?raw=true "Summary Importance")

### Explanations - individual Datapoints - AgeMeanImputer

![Alt text](https://github.com/MonishkaDas/nd00333-capstone/blob/master/starter_file/ScreenShots/Screenshot%20(108).png?raw=true "Summary Importance")

### Explanations - individual Datapoints - WeightMeanImputer

![Alt text](https://github.com/MonishkaDas/nd00333-capstone/blob/master/starter_file/ScreenShots/Screenshot%20(109).png?raw=true "Summary Importance")

### Explanation exploration - WeightMeanImputer and WeightMeanImputer for Probability Class:0

![Alt text](https://github.com/MonishkaDas/nd00333-capstone/blob/master/starter_file/ScreenShots/Screenshot%20(111).png?raw=true "Summary Importance")

### Explanation exploration - WeightMeanImputer and WeightMeanImputer for Probability Class:1

![Alt text](https://github.com/MonishkaDas/nd00333-capstone/blob/master/starter_file/ScreenShots/Screenshot%20(112).png?raw=true "Summary Importance")




The Explanations section shed some light on which of the features had the most impact in predicting the results. In this dataset, the duration, emp.var.rate and nr.employed seem to be the most essential for making accurate predictions.

| Configuration | Reason |
| :- | :- |
| **experiment_timeout_minutes** | Maximum time that all iterations combined can take before the experiment terminates. |
|**max_concurrent_iterations**|These are the maximumm iterations occuring simultaneously, in this case the value is set as 10|
|**n_cross_validations**|We use 5 cross validations to avoid overfitting |
|**primary_metric**| The primary metric for this experiment is Accuracy|
|**task**|Classification |
|**compute_target**|This is the compute cluster we will be using for the run |
|**training_data**|This is the training dataset derived from the default datastore  |
|**label_column_name**|This is the target column, in this case **\"cardio\"**|
|**enable_onnx_compatible_model**|This is set as **True** to make the model onnx compatible|


        

### Results


## Best Model
 
The best performing model is the `VotingEnsemble`

**Accuracy :  0.73125**

**average_precision_score_weighted :** 0.78              

**f1_score_weighted :** 0.73

**best_individual_pipeline_score :**  0.7295652159597188

**ensembled_algorithms :**  ['LightGBM', 'XGBoostClassifier', 'XGBoostClassifier', 'LightGBM', 'LightGBM', 'XGBoostClassifier', 'XGBoostClassifier', 'XGBoostClassifier', 'LightGBM', 'XGBoostClassifier']
    
**ensemble_weights :** [0.07692307692307693, 0.07692307692307693, 0.07692307692307693, 0.07692307692307693, 0.07692307692307693, 0.07692307692307693, 0.3076923076923077, 0.07692307692307693, 0.07692307692307693, 0.07692307692307693]



## Hyperparameter Tuning

The pipeline includes a Random Parameter Sampler, Bandit Policy and SKLearn estimator which are used in the Hyperdrive configuration for maximum optimization. The file _train.py_ is passed to the estimator as an entry_file and the estimator _est_ along with the policy, Parameter Sampler and some other parameters like primary metric (_Accuracy_) are passed to the HyderDrive Config Method which is then submitted and the Run details are displayed using the widget.

**Random sampling**

Random sampling supports discrete and continuous hyperparameters. It supports early termination of low-performance runs. In random sampling, hyperparameter values are randomly selected from the defined search space. I used _choice_ to pass the parameters _**--C** (0.4, 0.5) , **max iter** (1000, 1100, 1200, 1300, 1400, 1500)_ to the random Sampler


**Bandit policy**

Bandit policy is based on slack factor/slack amount and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run. I chose the following parameters for the Bandit Policy _slack_factor = 0.001, evaluation_interval=1_.

We retrieve the best model based on the primary metric _Accuracy_ and register it.

### Results

**Best Run ID **:  HD_da36b743-a9e9-40f3-aaa5-d4c5d7fdad9d_1

**Accuracy:**  0.716566866267465

**Metrics: ** 

*Regularization Strength:*: 0.40955258766901365

*Max iterations*: 1300

**Parameters:**

*--C* :  0.40955258766901365

*--max_iter* :  1300


## Model Deployment

Both, the AzureML Run and HyperDrive Run are compared and the best runs in each are registered. The best model from the two is chosen and then deployed. The code for the Model deployment is available in the _automl.ipynb_ notebook. we retrive the swagger.URI and scoring.URI along with deployment status, which is shown as "healthy" in this case. The _logs.py_ and _endpoint.py_ are updated with the scoring.uri and key and run. The _logs.py_ enables Applications Insights and the _endpoint.py_ is used to test the model.

![Alt text](https://github.com/MonishkaDas/nd00333-capstone/blob/master/starter_file/ScreenShots/Screenshot%20(121).png?raw=true "Summary Importance")



## Screen Recording
The link to the screencast - https://www.youtube.com/watch?v=LPxz2xldigQ&feature=youtu.be

## Standout Suggestions

## ONNX Model

ONNX (Open Neural Network Exchange) is an open container format for the exchange of neural network models between different frameworks, providing they support ONNX import and export. ONNX was designed to enable fledgling AI systems to leave the nest, increasing their potential application base by expanding their interoperability.

ONNX's container format allows neural networks to be switched between different cloud service providers or into private clouds. More portability makes it possible to use models in new places, to the developer's benefit, and increases the range of models available to Facebook and Microsoft. It may also foster innovation and speed development by facilitating sharing and collaboration among researchers.

The got to retrieving the ONNX model and testing it is in the _automl.ipynb_ file.
