#### Import Section
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import exists

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import RFE

import warnings
warnings.filterwarnings("ignore")

# !!!!!!!! WRITE UP A MODULE DESCRIPTION
def split_data(df,target):
    ''' 
    takes in dataframe
    uses train test split on data frame using test size of 2, returns train_validate, test
    uses train test split on train_validate using test size of .3, returns train and validate
    returns train, validate test
    '''
    train_validate, test = train_test_split(df, test_size= .2, random_state=514,stratify = df[target])
    train, validate = train_test_split(train_validate, test_size= .25, random_state=514,stratify = train_validate[target])
    print(train.shape, validate.shape, test.shape)
    return train, validate, test

def scale_split_data (train, validate, test):
    ''' 
    takes in your three datasets
    applies minmax scaler to them using dtypes of number
    fits to those columns
    applies to copies of datasets
    returns datasets scaled
    '''
    #create scaler object
    scaler = MinMaxScaler()

    # create copies to hold scaled data
    train_scaled = train.copy(deep=True)
    validate_scaled = validate.copy(deep=True)
    test_scaled =  test.copy(deep=True)

    #create list of numeric columns for scaling
    num_cols = train.select_dtypes(include='number')

    #fit to data
    scaler.fit(num_cols)

    # apply
    train_scaled[num_cols.columns] = scaler.transform(train[num_cols.columns])
    validate_scaled[num_cols.columns] =  scaler.transform(validate[num_cols.columns])
    test_scaled[num_cols.columns] =  scaler.transform(test[num_cols.columns])

    return train_scaled, validate_scaled, test_scaled


def get_kmeans_cluster_features(train,train_scaled,validate,validate_scaled,test,test_scaled,dict_to_cluster):
    ''' 
    takes in your three datasets to apply the new featuers to as well as applying to the base datasets
    a dictionary (iterable lists with the last being the order of clustering (function to auto later))
    '''

    for i in range(len(list(dict_to_cluster))-1):
        #set features
        X1 = train_scaled[dict_to_cluster[list(dict_to_cluster)[i]]]
        X2 = validate_scaled[dict_to_cluster[list(dict_to_cluster)[i]]]
        X3 = test_scaled[dict_to_cluster[list(dict_to_cluster)[i]]]

        kmeans_scaled = KMeans(n_clusters=dict_to_cluster[list(dict_to_cluster)[len(list(dict_to_cluster))-1]][i],random_state=123)
        kmeans_scaled.fit(X1)

        X1["cluster"] = kmeans_scaled.predict(X1)
        X2["cluster"] = kmeans_scaled.predict(X2)
        X3["cluster"] = kmeans_scaled.predict(X3)

        train_scaled[f"cluster {list(dict_to_cluster)[i]}"] = X1["cluster"]
        train[f"cluster {list(dict_to_cluster)[i]}"] = X1["cluster"]

        validate_scaled[f"cluster {list(dict_to_cluster)[i]}"] = X2["cluster"]
        validate[f"cluster {list(dict_to_cluster)[i]}"] = X2["cluster"]

        test_scaled[f"cluster {list(dict_to_cluster)[i]}"] = X3["cluster"]
        test[f"cluster {list(dict_to_cluster)[i]}"] = X3["cluster"]

    return train,train_scaled,validate,validate_scaled,test,test_scaled

def rfe(predictors_x,target_y,n_features):
    ''' 
    takes in the predictors (X) (predictors_x), the target (y) (target_y), and the number of features to select (k) 
    returns the names of the top k selected features based on the Recursive Feature Elimination class. and a ranked df
    '''

    model = LinearRegression()
    rfe = RFE(model,n_features_to_select=n_features)
    rfe.fit(predictors_x,target_y)

    #print(pd.DataFrame({"rfe_ranking":rfe.ranking_},index=predictors_x.columns).sort_values("rfe_ranking")[:n_features])
    X_train_transformed = pd.DataFrame(rfe.transform(predictors_x),columns=predictors_x.columns[rfe.get_support()],index=predictors_x.index)
    X_train_transformed.head(3)

    var_ranks = rfe.ranking_
    var_names = predictors_x.columns.tolist()

    rfe_ranked = pd.DataFrame({'Var': var_names, 'Rank': var_ranks}).sort_values("Rank")
    
    return rfe_ranked


def summarize(df):
    '''  
    takes in dataframe
    prints out df info and df describe as well as the amount of nulls in df by column and row
    then it finds the dtypes of "number" and makes the inverse categorical
    prints out the value counts of cats, as well as binning and doing the same for numerical
    '''
    print('-----')
    print('DataFrame info:\n')
    print (df.info())
    print('---')
    print('DataFrame describe:\n')
    print (df.describe())
    print('---')
    #print('DataFrame null value asssessment:\n')
    #print('Nulls By Column:', nulls_by_col(df))
    #print('----')
    #print('Nulls By Row:', nulls_by_row(df))
    numerical_cols = df.select_dtypes(include='number').columns.to_list()
    categorical_cols = df.select_dtypes(exclude='number').columns.to_list()
    print('value_counts: \n')
    for col in df.columns:
        print(f'Column Names: {col}')
        if col in categorical_cols:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins=10, sort=False, dropna=False))
            print('---')
    print('Report Finished')
    return

def FE_IF_model(train, train_scaled, validate, validate_scaled, test, test_scaled,indicate):
    ''' 
    input datasets and list of features to run model with
    creates,fits and predicts, plots peformance against target
    returns same datasets with predictor feature
    '''
    model = IsolationForest(contamination=float(.002),random_state=123,bootstrap=True)
    model.fit(train_scaled[indicate])
    train["if1_scores"] = model.decision_function(train_scaled[indicate])
    train["if1_anom"] = model.predict((train_scaled[indicate]))
    train_scaled["if1_anom"] = model.predict((train_scaled[indicate]))
    validate["if1_anom"] = model.predict((validate_scaled[indicate]))
    validate_scaled["if1_anom"] = model.predict((validate_scaled[indicate]))
    test["if1_anom"] = model.predict((test_scaled[indicate]))
    test_scaled["if1_anom"] = model.predict((test_scaled[indicate]))

    #plt.title("anomaly class compared to target") # Title with column name.
    #train[train["Class"]==0]["if1_anom"].hist(bins=10,alpha=.5) # Display histogram for column.
    #train[train["Class"]==1]["if1_anom"].hist(bins=10,alpha=.5) # Display histogram for column.
    #plt.yscale("log")
    #plt.grid(False) # Hide gridlines
    #plt.show()

    train[["if1_anom","Class"]].groupby(["if1_anom"]).agg(["count","mean"])

    return train, train_scaled, validate, validate_scaled, test, test_scaled


def wrangle_creditcard():
    '''
    takes in nothing
    goes through the wrangle pipeline
    acquires, sets target, creates engineered features, splits, scales, clusters, maps for Fraud, and summarizes info
    returns all creations
    '''

    #get the data and set target
    df = pd.read_csv("~/Downloads/creditcard.csv")
    target="Class"

    #create a new feature that is the product of valuable features (will explore later)
    rfe_ranked=rfe(df.drop(columns=target),df[target],10)
    indicate = rfe_ranked[rfe_ranked["Rank"]==1]["Var"].tolist()
    df["indicate"] = 1
    for i in indicate:
        df["indicate"] = df['indicate']*df[i]

    #get prepared to cluster
    dict_to_cluster = { "indicate":['V3', 'V4', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17'],
                        "V-030409":['V3', 'V4', 'V9'],
                        "V-101112":['V10', 'V11', 'V12'],
                        "V-141617":['V14', 'V16', 'V17'],
                        "cluster_order":[4,4,4,4]}
    cluster_list = ["cluster "+x for x in dict_to_cluster.keys()][:-1]

    #split and scale data, cluster for FE
    train,validate,test = split_data(df,target)
    train_scaled, validate_scaled, test_scaled = scale_split_data(train,validate,test)
    train, train_scaled, validate, validate_scaled, test, test_scaled = get_kmeans_cluster_features(train,train_scaled,validate,validate_scaled,test,test_scaled,dict_to_cluster )

    #more feature engineering
    FE_IF_model(train, train_scaled, validate, validate_scaled, test, test_scaled,indicate)

    ## changing class to better describe what we're looking for in exploration, will drop
    train["Fraud"] = train.Class.map({0:"No-Fraud",1:"Yes-Fraud"})
    summarize(df)

    return df, target, train, train_scaled, validate, validate_scaled, test, test_scaled, cluster_list, indicate

