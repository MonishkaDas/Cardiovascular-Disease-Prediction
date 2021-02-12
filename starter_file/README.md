
# Prediction of Cardiovascular Disease - Azure ML Capstone Project

This is the final project of Udacity Azure ML Nanodegree. As a part of this project we were asked to pick a dataset of our choice. We preprocess it and then use Azure ML studio tools like AutoML and Hyperdrive Configuration to derive the best model based on metrics. We compare the runs and then deploy the best model. We enable applications insights using _logs.py_ and then test the endpoints using _endpoint.py_ file. 

## Project Set Up and Installation

The project was run and deployed on Azure ML studio. For anyone trying too replicate my work, you will need to craete an Azure ML instance to run the notebook. The compute target, which in case is named "compute-cluster" is made and is used to run both the AutoML Model (_automl.ipynb_) and the HyperDrive Model (_hyperparameter-tuning.ipynb_). the files can found the repository along with the _train.py_ file that is the entry-script for the HyperDrive Configuration model, the _logs.py_ file to enable application insights and the _endpoint.py_ file to test the endpoints. To use the configurations on a different dataset, the _train.py_ and _endpoint.py_ files need to altered accordingly and the dataset has to be cleaned and processed as required. A few tweaks to the configuration must get you the best possible model that cab be then deployed using the code in the  _automl.ipynb_ file.

## Dataset

### Overview
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

The task is to firstly, study the dataset and check for outliers, categorical values, missing values, etc. We then process the dataset and clean it. A well processed dataset gives out better results, this is why data preprocessing is an essential step despite the fact that AutoML has in-built functions to deal with improper data. After processing, we pass the dataset to the confired automl and hyperdrive models. We run the models and monitor the process using the run details widget. We then pick the best model from the two based on metrics, which in this case is the AutoML Model. We deploy the model and run _logs.py_ file to enable Applications insights. Finally, we test the endpoints of the model using the _endpoint.py_ file. We also creat a Onnx model and test it.


### Data Preprocessing

The dataset has a total of 70,000 readings with no missing data. However, there are a few outliers and invalid readings (*For example: a few readings show the diastolic pressure, _ap-lo_ to be higher than the systolic pressure, _ap-hi_*). We drop the outliers and duplicate readings and clean the invalid data, we drop the column _"id"_, we convert the feature column _gender_ to a binary, we replace the categorical columns like _cholestrol, gluc_ with the dummies and we standardize columns like, _age, height, weight, ap-hi, ap-lo_. This data is then passed to the configured experiments and run.

### Access

The dataset is uploaded to the GitHub Repository and accessed through a link using TabularDatasetFactories, it is then converted into pandas dataframes for preprocessing. After Processing the Dataset, it is uploaded to the default datastore, so it can be retrieved as a TabularDataFrame and passed as the _training_data_.

## Automated ML



### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
