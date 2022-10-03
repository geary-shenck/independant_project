#### Import Section
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report,recall_score,precision_score,accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")


#encoding clusters and setting X_train
def encoding_for_modeling_Xy(train, train_scaled, validate, validate_scaled, test, test_scaled,target,model_feautures):
    ''' 
    code to help expedite the encoding process in a repeatble fashion
    '''
    train["V141617_3"] = np.where(train["cluster V-141617"]==3,1,0)
    train["V101112_2"] = np.where(train['cluster V-101112']==2,1,0)
    train["V030409_0"] = np.where(train["cluster V-030409"]==0,1,0)

    train_scaled["V141617_3"] = np.where(train_scaled["cluster V-141617"]==3,1,0)
    train_scaled["V101112_2"] = np.where(train_scaled['cluster V-101112']==2,1,0)
    train_scaled["V030409_0"] = np.where(train_scaled["cluster V-030409"]==0,1,0)

    validate["V141617_3"] = np.where(validate["cluster V-141617"]==3,1,0)
    validate["V101112_2"] = np.where(validate['cluster V-101112']==2,1,0)
    validate["V030409_0"] = np.where(validate["cluster V-030409"]==0,1,0)

    validate_scaled["V141617_3"] = np.where(validate_scaled["cluster V-141617"]==3,1,0)
    validate_scaled["V101112_2"] = np.where(validate_scaled['cluster V-101112']==2,1,0)
    validate_scaled["V030409_0"] = np.where(validate_scaled["cluster V-030409"]==0,1,0)

    test["V141617_3"] = np.where(test["cluster V-141617"]==3,1,0)
    test["V101112_2"] = np.where(test['cluster V-101112']==2,1,0)
    test["V030409_0"] = np.where(test["cluster V-030409"]==0,1,0)

    test_scaled["V141617_3"] = np.where(test_scaled["cluster V-141617"]==3,1,0)
    test_scaled["V101112_2"] = np.where(test_scaled['cluster V-101112']==2,1,0)
    test_scaled["V030409_0"] = np.where(test_scaled["cluster V-030409"]==0,1,0)

    ##setting the data sets based on features i want to include
    X_train = train_scaled[model_feautures]
    y_train = train_scaled[target]

    X_validate = validate_scaled[model_feautures]
    y_validate = validate_scaled[target]

    X_test = test_scaled[model_feautures]
    y_test = test_scaled[target]
    print("done")
    return X_train,y_train,X_validate,y_validate,X_test,y_test

def modeling_initial(train,target,X_train,y_train,X_validate,y_validate):
    ''' 
    runs through preset tuned models and returns the results
    '''

    ##find the baseline of churn based on mode
    baseline_predict = train[target].mode()[0]
    (train[target] == baseline_predict).mean()

    print(f"{baseline_predict} <--- mode of Class(No-Fraud) in training data / baseline prediction")
    print(f"{(train[target] == baseline_predict).mean() * 100 :.2f}% <--- accuracy of baseline prediciton in training data")
    print("----------------")
    #############################################

    ## doing a basic decison tree classification
    ## fits the model, plots it, predicts off the training, and produces results (classification and confusion matrix)

    clf = DecisionTreeClassifier(max_depth= 4, random_state= 123, criterion="gini")
    clf = clf.fit(X_train,y_train)

    y_pred_train_dt = clf.predict(X_train)
    y_pred_val_dt = clf.predict(X_validate)

    print(classification_report(y_validate, y_pred_val_dt), "\t Decision Tree classification report on validate set")
    print("----------------")
    #############################################

    ## logistic regression classifier, played with values until i found some i liked
    ## produces confusion and classification reports

    logreg = LogisticRegression(C=.1)#, class_weight={0:1, 1:99}, random_state=123, intercept_scaling=1, solver='lbfgs', max_iter=1000000000)
    logreg.fit(X_train,y_train)
    y_pred_train_array = logreg.predict(X_train)
    y_pred_logreg_val = logreg.predict(X_validate)
    print(classification_report(y_validate, y_pred_logreg_val),"\t Logistic Regression validate classification")
    print("----------------")
    #############################################

    ## random forest classifier, i played with the values until i settled on these
    ## fits the data, predicts on the training, and runs classification and confusion reports
    ## also plots the most weight of the top 10 features with a confusion matrix display below it

    rf = RandomForestClassifier(bootstrap=True,class_weight=None,criterion="gini",min_samples_leaf=1,n_estimators=100,max_depth=7,random_state=123)
    rf.fit(X_train,y_train)
    y_pred_rf = rf.predict(X_train)
    y_pred_rf_val = rf.predict(X_validate)

    print(classification_report(y_validate, y_pred_rf_val),"\t Random Forest validate classification report")
    print(round(recall_score(y_validate,y_pred_rf_val,average="weighted")*100,2),"% recall")
    print(round(precision_score(y_validate,y_pred_rf_val,average="weighted")*100,2),"% precision")
    print(round(accuracy_score(y_validate,y_pred_rf_val)*100,2),"% accuracy")
    print("----------------")

    return y_pred_val_dt,y_pred_logreg_val,y_pred_rf_val,clf,logreg,rf

def give_me_confusion(actual_series,model1_series,pos,neg):
    ''' 
    was using these a lot so made a temporary function
    taeks in the actural seris and the mdela series an does abinch
    of testas on them
    '''
    model_series = confusion_matrix(actual_series,model1_series,labels=(pos,neg))
    model_series_TN = model_series[0,0]
    model_series_FP = model_series[0,1]
    model_series_FN = model_series[1,0]
    model_series_TP = model_series[1,1]

    print (round(model_series_TN,3), "True Negative for model_series")
    print (round(model_series_FP,3), "False Positive for model_series")
    print (round(model_series_FN,3), "False Negative for model_series")
    print (round(model_series_TP,3), "True Positive for model_series")

    baseline_accuracy = (actual_series == actual_series.mode()[0]).mean()
    print(round(baseline_accuracy*100,2), "% mode accuracy for actual")

    model_series_ACC = (model_series_TP + model_series_TN) / (model_series_TP + model_series_TN + model_series_FP + model_series_FN)
    print(round(model_series_ACC*100,2), "% accuracy for model_series")

    model_series_TPR = model_series_TP / (model_series_TP + model_series_FN)
    print (round(model_series_TPR*100,2), "% recall/TPR for model_series (rate identified correctly)")

    model_series_TNR = model_series_TP/(model_series_TN+model_series_FP)
    print(round(model_series_TNR,1), "specificity/TNR for model_series")

    model_series_PPV = model_series_TP/(model_series_TP+model_series_FP)
    print(round(model_series_PPV*100,2) , "% precision/PPV for model_series (rate correctly classified)")

    model_series_NPV = model_series_TN/(model_series_TN+model_series_FN)
    print(round(model_series_NPV*100,2) , "% negative prediction value for model_series")

