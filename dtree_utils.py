# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 01:29:55 2018

@author: chinmaya
"""
def categorical_2_binary(dframe, cat_feature = None, flag = True):
    import pandas as pd
    # one hot encoding
    # dataframe in where categorical variables are present
    # cat_feature : list of categorical variable column names
    # flag == false, when cat_feature inpunt is none
    if flag == False:
        cat_dframe = dframe.select_dtypes(include =["object"]).copy()
        cat_feature = cat_dframe.columns.tolist()
        
    for feature in cat_feature:
        dframe_oh = pd.get_dummies(dframe[feature], prefix = feature)
        dframe = dframe.drop(feature, axis = 1)
        
        for col in dframe_oh.columns.tolist():
            dframe[col] = dframe_oh[col]
    return dframe
        
        
# Visualizing a decision tree model

def visualize_decision_tree(model,data):
    # model: saved model from decision tree classifier
    # data: raw data of X, contains feature column name    
    # return: tree graph
    from sklearn import tree
    import graphviz 
    from os import system
    
    dot_data = tree.export_graphviz(small_model, out_file='simple_tree.dot',
                                   feature_names=data.columns,  
                             class_names=['+1','-1'],  
                             filled=True, rounded=True,  
                             special_characters=True) 
    system("dot -Tpng simple_tree.dot -o simple_tree.png")
    
    from IPython.display import Image
    Image(filename='simple_tree.png')
           
    
def false_pos_false_neg(prediction,validation_y, class_values =[-1,1]):
    # prediction: output from model predict
    # validation_y: actual y variable
    # class used in model for binary classification, it can be [0,1]
    false_positive = ((prediction == class_values[1]) * (validation_y == class_values[0])).sum()
    false_nagative = ((prediction == class_values[0]) * (validation_y == class_values[1])).sum()
    return false_positive, false_nagative