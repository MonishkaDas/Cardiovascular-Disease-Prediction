from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import argparse
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

s_list = ["age", "height", "weight", "ap_hi", "ap_lo"]

def standartization(x):
    x_std = x.copy(deep=True)
    for column in s_list:
        x_std[column] = (x_std[column]-x_std[column].mean())/x_std[column].std()
    return x_std 

def clean_data(data):
    
    df = data.to_pandas_dataframe()
    df.drop("id", axis = 1, inplace = True)  


    df=standartization(df)


    df.gender = df.gender.replace(2,0)

    df.drop(df[(df['height'] > df['height'].quantile(0.975)) | (df['height'] < df['height'].quantile(0.025))].index,inplace=True)
    df.drop(df[(df['weight'] > df['weight'].quantile(0.975)) | (df['weight'] < df['weight'].quantile(0.025))].index,inplace=True)

    df.drop(df[(df['ap_hi'] > df['ap_hi'].quantile(0.975)) | (df['ap_hi'] < df['ap_hi'].quantile(0.025))].index,inplace=True)
    df.drop(df[(df['ap_lo'] > df['ap_lo'].quantile(0.975)) | (df['ap_lo'] < df['ap_lo'].quantile(0.025))].index,inplace=True)

    duplicated = df[df.duplicated(keep=False)]
    duplicated = duplicated.sort_values(by=['age', "gender", "height"], ascending= False)
    df.drop_duplicates(inplace=True)

    df['cholesterol']=df['cholesterol'].map({ 1: 'normal', 2: 'above normal', 3: 'well above normal'})
    df['gluc']=df['gluc'].map({ 1: 'normal', 2: 'above normal', 3: 'well above normal'})
    dummies = pd.get_dummies(df[['cholesterol','gluc']],drop_first=True)
    df = pd.concat([df,dummies],axis=1)
    df.drop(['cholesterol','gluc'],axis=1,inplace=True)
    


    y = df.pop("cardio")
    X = df
    
    return X, y

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    web_path = "https://raw.githubusercontent.com/MonishkaDas/nd00333-capstone/master/starter_file/cardio_train.csv"

    ds = TabularDatasetFactory.from_delimited_files(path=web_path,separator=";")
    
    x, y = clean_data(ds)

    #Split data into train and test sets.
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42,shuffle=True)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))


if __name__ == '__main__':
    main()