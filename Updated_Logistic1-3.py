#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 20:26:21 2020

@author: chait
"""

import pandas as pd
import numpy as np
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import json
from sklearn.model_selection import KFold


def calculate_sigmoid(z):
    g_of_z = 1 / (1 + np.exp(-z))
    return (g_of_z)
    
def get_z(theta,X):
    return(np.dot(theta,X))
    
def get_likelihood(z,actual_y):
    actual_y_times_z = actual_y*z
    log_of_one_plus_e_power_z = np.log(1+np.exp(z))
    loglikelihood = np.sum(actual_y_times_z - log_of_one_plus_e_power_z)
    return(loglikelihood)


def cross_entropy_loss(data_matrix,actual_y,weight_matrix):
    clm = -(actual_y*np.log(calculate_sigmoid(np.dot(data_matrix, weight_matrix)))) - ((1-actual_y)*np.log(1-calculate_sigmoid(np.dot(data_matrix, weight_matrix))))
    #print(clm)
    return (np.mean(clm))


def logistic_regression(iterations,data_matrix,weight_matrix,actual_y,alpha):
    for i in range(iterations):
        z = get_z(data_matrix,weight_matrix)
        h_theta_of_x = calculate_sigmoid(z)
        column_transpose =  data_matrix.T
        error = actual_y - h_theta_of_x
        gradient = np.dot(column_transpose,error)
        weight_matrix = weight_matrix+(alpha*gradient)
        
        likelihood_value = get_likelihood(z,actual_y)
        #print(likelihood_value)
        
        cross_loss = cross_entropy_loss(data_matrix,actual_y,weight_matrix)
        #print(cross_loss)
        with open("error.txt","a+") as f:
            f.write(str(cross_loss))
            f.write("\n")
        
    return(weight_matrix)
def logistic_regression1(iterations,data_matrix,weight_matrix, actual_y, alpha, add_intercept = False):
    if add_intercept:
        intercept = np.ones((data_matrix.shape[0], 1))
        data_matrix = np.hstack((intercept, data_matrix))
        
    #weight_matrix = np.zeros(data_matrix.shape[1])
    #print(weight_matrix)
    #print("---------")
    weight_matrix,cost_list=stochastic_gradient_ascent(data_matrix,actual_y,weight_matrix,alpha,iterations)
    #print(list(cost_list))
    return weight_matrix
def stochastic_gradient_ascent(features,target,theta,learning_rate,iterations):
    m=len(target)
    cost_list=np.zeros(iterations)
    print(target)
    for it in range(iterations):
        cost=0.0
        for i in range(m):
            rand_ind = np.random.randint(0,m)
            features_i = features[rand_ind,:].reshape(1,features.shape[1])
            target_i = target[rand_ind].reshape(1,1)
            prediction=np.dot(features_i,theta)
            theta=theta+(1/m)*learning_rate*(features_i.T.dot((target_i-prediction)))
            cost+=cross_entropy_loss(features_i,target_i,theta)
    cost_list[it]=cost
    return theta,cost_list
def evaluationmetric(actual,predicted,filename):
    class_metrics = classification_report(actual.astype(int),predicted.astype(int))
    print(class_metrics)
    class_metrics_dict = classification_report(actual.astype(int),predicted.astype(int),output_dict=True)
    with open(filename,"a+") as ft:
        ft.write(json.dumps(class_metrics_dict))
        
    
    cm = confusion_matrix(actual.astype(int),predicted.astype(int))
    print("confusion matrix:\n",cm)
def cross_validation(num_folds, train_df):
    kf = KFold(n_splits=2)
    labels_arr = np.array(train_df.iloc[:,-1])
    df = train_df.iloc[:,:-1]
    i=0
    train_df_array = np.array(train_df)
    for train_index, test_index in kf.split(train_df_array):
        X_train_df, X_test_df = train_df_array[train_index], train_df_array[test_index]
        y_train_df, y_test_df = labels_arr[train_index], labels_arr[test_index]
        print(y_train_df)
        row_count = X_train_df.shape[1]

        weight_matrix = np.zeros(row_count)

        i=i+1
        trained_weight_matrix = logistic_regression1(50000,X_train_df,weight_matrix,y_train_df,0.01,False)
        predicted_values=np.round(calculate_sigmoid(np.dot(X_test_df,trained_weight_matrix)))
        evaluationmetric(labels_array_test,y_test_df,"results_cross_validation"+i+".txt")


#trained_weight_matrix = logistic_regression(50000,df_array,weight_matrix,labels_array,0.01)
def main(filename1,filename2):
    df = pd.read_csv(filename1)
    df_test=pd.read_csv(filename2)
    folds=2
    cross_validation(folds,df)
    labels_array = np.array(df.iloc[:,-1])
    df = df.iloc[:,:-1]
    df_array = np.array(df)
    
    labels_array_test = np.array(df_test.iloc[:,-1])
    df_test = df_test.iloc[:,:-1]
    df_test_array = np.array(df_test)
    
    row_count = df_array.shape[1]

    weight_matrix = np.zeros(row_count)
    

    trained_weight_matrix = logistic_regression1(10000,df_array,weight_matrix,labels_array,0.01,False)
    predicted_values=np.round(calculate_sigmoid(np.dot(df_test_array,trained_weight_matrix)))
    evaluationmetric(labels_array_test,predicted_values,"results_"+filename2+".txt")

main("stem_count_train.csv","stem_count_test.csv")
#main("nostem_count_train.csv","nostem_count_test.csv")
#main("stem_binary_train.csv","stem_binary_test.csv")
#main("nostem_binary_train.csv","nostem_binary_test.csv")
        
        
 

        
        
        
        
    
    
    
    


    
    

