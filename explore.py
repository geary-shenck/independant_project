#### Import Section
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from os.path import exists

from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, SelectKBest, f_regression

import warnings
warnings.filterwarnings("ignore")


def correlation_guidance(train,target):
    ''' 
    input train dataset and target string
    creates a correlation plot of the variables in the columns list (self defined) by class
    no return
    '''
    plt.figure(figsize=(30, 30))
    for i,col1 in enumerate(train.iloc[:,1:30].columns.tolist()):
        plt.subplot(5,6,i+1)
        x=train[train[target]==0][col1].sample(len(train[train[target]==1]))
        y=train[train[target]==1][col1]
        plt.plot(x, y, "o",color="grey")
        m,b = np.polyfit(x,y,1)
        plt.plot(x,m*x+b,label=f"regression line - f(x)={round(m,3)}x+{round(b,1)}")
        plt.xlabel("No-Fraud")
        plt.ylabel("Yes-Fraud")
        plt.legend()
        plt.title(f"Fraud by {col1} value")
    plt.show()

def chi2_target_v_groups(train,target,cluster_list):
    ''' 
    input the train dataset, string of the target, and a list of featues to run through
    does a chi2 test for indepenance (proportionality) and plots the results
    no return
    '''
    # creates a column of the payments for easy analysis, runs a crosstab to put into a chi2 independancy test.
    # produces observed and expected values
    # returns the chi2 and pval for the whole set
    for i in cluster_list[1:]:
        df1 = pd.crosstab(train["Fraud"],train[i])

        alpha = .05
        chi2, p, degf, expected = stats.chi2_contingency(df1)
        H0 = (f"{df1.index.name} is not different in the distribution of {df1.columns.name}")
        H1 = (f"{df1.index.name} is different in the distribution of {df1.columns.name}")
        print('Observed')
        print(df1.values)
        print('---\nExpected')
        dfexpected = df1.copy()
        for i in range(len(dfexpected)):
            dfexpected.iloc[i] = expected[i]
        print(dfexpected.values)
        print(f'---\nchi^2 = {chi2:.4f}, p = {p:.5f}, degf = {degf}')
        if p>alpha:
            print(f"due to p={p:.5f} > α={alpha} we fail to reject our null hypothesis\n({H0})")
        else:
            print(f"due to p = {p:.5f} < α = {alpha} we reject our null hypothesis\n( ", '\u0336'.join(H0) + '\u0336' , ")")

        #plot the results
        plt.figure(figsize=(8,4))
        plt.suptitle(f"{df1.index.name} Values for {df1.columns.name}", fontsize=12, y=0.99)

        for x,col in enumerate(df1.T.columns):
            ax = plt.subplot(1,2,x+1)
            pd.concat({'Observed': df1.T[col], 'Expected': dfexpected.T[col]}, axis=1).\
                plot.bar(color={"Observed": "grey", "Expected": "pink"}, edgecolor="black",ax=ax)
            ax.set_ylabel("Count")
            ax.set_title(f'{col} values') # Title with column name.
        plt.show()

def twosample_ttest_for_cc(train):
    ''' 
    input not required due to tailored use
    does levene test and passes flag to 2sided 2sample ttest, prints out results and plots out a barchart of the class
    returns nothing
    '''
    ## does a levene test for comparing variance, creates a boolean flag that is passed into the ttest for
    ## comparing the Fraud monthly charge and the Not Fraud monthly charge
    ## then plots the graphs for a visual

    cat = "indicate"
    target = "Class"

    H0 = "Feature Engineered of Fraud is equal to Feature Engineered of Not Fraud"
    Ha = "Feature Engineered of Fraud is less than or greater than Feature Engineered of Not Fraud"
    alpha = .05

    #compare variances to know how to run the test
    stat,pval = stats.levene(train[train[target]==1][cat],train[train[target]==0][cat])
    stat,pval
    if pval > 0.05:
        equal_var_flag = True
        print(f"we can accept that there are equal variance in these two groups with {round(pval,2)} certainty Flag=T",'stat=%.5f, p=%.5f' % (stat,pval))
    else:
        equal_var_flag = False
        print(f"we can reject that there are equal variance in these two groups with {round((1-pval),2)} certainty Flag=F",'stat=%.5f, p=%.5f' % (stat,pval))

    #runs a two tailed sample to sample T-test 
    t, p = stats.ttest_ind( train[train[target]==1][cat], train[train[target]==0][cat], equal_var = equal_var_flag )

    #print line for result
    if p > alpha: #directionality, or if t is negative
        print("\n We fail to reject the null hypothesis (",(H0) , ")",'t=%.5f, p=%.5f' % (t,p))
    else:
         print("\n We reject the null Hypothesis (", '\u0336'.join(H0) + '\u0336' ,")",'t=%.5f, p=%.5f' % (t,p))

    #plots out a grouping of the features
    plt.figure(figsize=(12,6))
    train[["Fraud","indicate"]].groupby("Fraud").agg("mean").plot.bar(rot=0,color="white",edgecolor="grey",linewidth=5)
    plt.axhline(y=train.indicate.mean(),label=f"Indicate Mean {(int(train.indicate.mean()))}",color="black",linewidth=4)
    plt.yscale("log")
    plt.ylabel("Indicate - Mean Value")
    plt.legend()
    plt.title(f"Indicate Feature means Compared in relation to Fraud Class")
    plt.show()
    
def sample_to_pop_ttest_strong_clusters(train,strong_clusters):
    ''' 
    input train dataset and list of strong cluster tuples
    does a ttest prints results, plots relation
    returns nothing
    '''
    temp1=train.copy()
    for feature,sample in strong_clusters:
        # sets variables
        target = "Class"
        alpha = .025
        population_name = feature
        sample_name = sample
        print(target,"<-target |",population_name,"<-population name |",sample_name,"<-sample name")

        #sets null hypothesis
        H0 = f"{sample_name} as a sample has equal mean values to {population_name} as a population regarding Fraud"
        Ha = f"{sample_name} as a sample does not have equal mean values to {population_name} as a population regarding Fraud"

        #runs test and prints results
        t, p = stats.ttest_1samp( train[train[population_name] == sample_name][target], train[target].mean())
        if p > alpha:
            print("We fail to reject the null hypothesis (",(H0) , ")",'t=%.5f, p=%.5f' % (t,p))
        else:
            print("We reject the null Hypothesis (", '\u0336'.join(H0) + '\u0336' ,")",'t=%.5f, p=%.5f' % (t,p))
        print("----------")

        temp1[population_name] = np.where(temp1[population_name]==sample_name,sample_name,f"other {population_name}")

    #plot the results
    plt.figure(figsize=(18,4))
    plt.suptitle(f"Cluster Sample Values Compared for Fraud", fontsize=12, y=0.99)
    i=0
    for feature,sample in strong_clusters:
        #plots out a grouping of the features
        i+=1
        ax = plt.subplot(1,3,i)
        temp1[[target,feature]].groupby(feature).agg("mean").plot.bar(rot=0,color="white",edgecolor="grey",linewidth=5,ax=ax)
        ax.axhline(y=temp1[target].mean(),label=f"Fraud Mean {(round(temp1[target].mean(),3))}",color="black",linewidth=3)
        ax.set_ylabel("% of Fraud")
        plt.legend()
        ax.set_title(f"{feature}-{sample} means Compared in relation to Fraud Class",fontsize=8)
    plt.show()

def ttest_V_features(train,v_features=["V20","V23","V28"]):
    ''' 
    input train dataframe and a list of features to check (default is gtg)
    does ttesting (two tailed), sample to population, prints results and plots the features
    returns nothing
    '''
    #starts the loop through the list to test
    for each in v_features:
        # sets variables
        target = each
        alpha = .05
        population_name = "Fraud"
        sample_name = "Yes-Fraud"
        print(target,"<-target |",population_name,"<-population name |",sample_name,"<-sample name")
        #sets null hypothesis
        H0 = f"{sample_name} as a sample has equal mean values to {population_name} as a population regarding {each}"
        Ha = f"{sample_name} as a sample does not have equal mean values to {population_name} as a population regarding {each}"
        #runs test and prints results
        t, p = stats.ttest_1samp( train[train[population_name] == sample_name][target], train[target].mean())
        if p > alpha:
            print("We fail to reject the null hypothesis (",(H0) , ")",'t=%.5f, p=%.5f' % (t,p))
        else:
            print("We reject the null Hypothesis (", '\u0336'.join(H0) + '\u0336' ,")",'t=%.5f, p=%.5f' % (t,p))
        print("----------")
    #plot the results
    plt.figure(figsize=(18,4))
    plt.suptitle(f"Features Values Compared for Fraud", fontsize=12, y=0.99)
    i=0
    for each in v_features:
        #plots out a grouping of the features
        i+=1
        ax = plt.subplot(1,3,i)
        train[[population_name,each]].groupby(population_name).agg("mean").plot.bar(rot=0,color="white",edgecolor="grey",linewidth=5,ax=ax)
        ax.axhline(y=train[each].mean(),label=f"{each} Mean {(round(train[target].mean(),3))}",color="black",linewidth=3)
        ax.set_ylabel(f"% of {each}")
        plt.legend()
        ax.set_title(f"{each} Mean Compared in relation to Fraud Class",fontsize=8)
    plt.show()
