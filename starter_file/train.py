from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

import seaborn as sns
sns.set()
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix



# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# web_path= "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
# ds = TabularDatasetFactory.from_delimited_files(path=web_path)

dataset = pd.read_csv("../input/Dataset_spine.csv")

#ds = TabularDatasetFactory.from_delimited_files(path=web_path)

run = Run.get_context()



def clean_data(data):
    #remove unnecessary column
    del dataset["Unnamed: 13"]
    
    # Change the Column names
    dataset.rename(columns = {
        "Col1" : "pelvic_incidence", 
        "Col2" : "pelvic_tilt",
        "Col3" : "lumbar_lordosis_angle",
        "Col4" : "sacral_slope", 
        "Col5" : "pelvic_radius",
        "Col6" : "degree_spondylolisthesis", 
        "Col7" : "pelvic_slope",
        "Col8" : "direct_tilt",
        "Col9" : "thoracic_slope", 
        "Col10" :"cervical_tilt", 
        "Col11" : "sacrum_angle",
        "Col12" : "scoliosis_slope", 
        "Class_att" : "y"}, inplace=True)
    
    x_df = data.to_pandas_dataframe().dropna()
    X = dataset.iloc[:, :-1]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(X)
    scaled_df = pd.DataFrame(data = scaled_data, columns = X.columns)
    y_df = x_df.pop("y").apply(lambda s: 1 if s == "Abnormal" else 0)
    return x_df,y_df

    

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    
    x, y = clean_data(ds)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=42,shuffle=True)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()
