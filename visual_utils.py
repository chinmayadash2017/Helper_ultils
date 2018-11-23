# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 21:55:53 2018

@author: chinmaya

"""

import matplotlib.pyplot as plt
import seaborn as sns

#draw the bar plot for class distribution
def classdist_barplot(dataframe,target):
    # dataframe: pandas
    #target:column name
    cnt_cls = dataframe[target].vlaue_counts()
    plt.figure(figsize=(12,4))
    sns.barplot(cnt_cls.index, cnt_cls.values, alpha=0.8)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel(target, fontsize=12)
    plt.xticks(rotation=90)
    plt.show()
    
