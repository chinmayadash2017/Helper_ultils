# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 01:29:55 2018

@author: chinmaya
"""
import numpy as np

# Convert the categorical variable to binary values(0,1 like true, false)
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

           
# calulate the false positive and false negative countfrom model prediction 
def false_pos_false_neg(prediction,validation_y, class_values =[-1,1]):
    # prediction: output from model predict
    # validation_y: actual y variable
    # class used in model for binary classification, it can be [0,1]
    false_positive = ((prediction == class_values[1]) * (validation_y == class_values[0])).sum()
    false_nagative = ((prediction == class_values[0]) * (validation_y == class_values[1])).sum()
    return false_positive, false_nagative


# compute the number of misclassified examples at intermediate node of decision tree
def intermediate_node_num_mistakes(labels_in_node):
    # intermediate_node_num_mistakes which computes the number of misclassified examples
    # of an intermediate node given the set of labels (y values)
    # of the data points contained in the node
    # Corner case: If labels_in_node is empty, return 0
    #assert(len(labels_in_node) > 0)   
    # Count the number of 1's (safe loans)   
    safe_loan = (labels_in_node==1).sum()
    # Count the number of -1's (risky loans)           
    risky_loan = (labels_in_node==-1).sum()
    # Return the number of mistakes that the majority classifier makes 
    return min(safe_loan, risky_loan)


# Function to pick best feature to split on
def best_splitting_feature(data, features, target):
    
    # target_values = data[target]
    best_feature = None # Keep track of the best feature 
    best_error = 10     # Keep track of the best error so far 
    # Note: Since error is always <= 1, we should intialize it with something larger than 1.

    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))  
    
    # Loop through each feature to consider splitting on that feature
    for feature in features:
        
        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]
        
        # The right split will have all data points where the feature value is 1
        ## YOUR CODE HERE
        right_split = data[data[feature] == 1]
            
        # Calculate the number of misclassified examples in the left split.
        # Remember that we implemented a function for this! (It was called intermediate_node_num_mistakes)
        # YOUR CODE HERE
        left_mistakes = intermediate_node_num_mistakes(left_split[target])            

        # Calculate the number of misclassified examples in the right split.
        ## YOUR CODE HERE
        right_mistakes = intermediate_node_num_mistakes(right_split[target])  
            
        # Compute the classification error of this split.
        # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
        ## YOUR CODE HERE
        error = (left_mistakes + right_mistakes) / num_data_points

        # If this is the best error we have found so far, store the feature as best_feature and the error as best_error
        ## YOUR CODE HERE
        if error < best_error:
            best_feature = feature
            best_error = error
    
    return best_feature # Return the best feature we found


# creates a leaf node given a set of target values
def create_leaf(target_values):    
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True}   
    ## YOUR CODE HERE 
   
    # Count the number of data points that are +1 and -1 in this node.
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])    

    # For the leaf node, set the prediction to be the majority class.
    # Store the predicted class (1 or -1) in leaf['prediction']
    if num_ones > num_minus_ones:
        leaf['prediction'] =  1        ## YOUR CODE HERE
    else:
        leaf['prediction'] =  -1        ## YOUR CODE HERE        

    # Return the leaf node
    return leaf

# Early stopping condition 2: Minimum node size
def reached_minimum_node_size(data, min_node_size):
    # Return True if the number of data points is less than or equal to the minimum node size.
    return len(data) <= min_node_size

# Early stopping condition: Minimum gain in error reduction
def error_reduction(error_before_split, error_after_split):
    # Return the error before the split minus the error after the split.
    return error_before_split - error_after_split

# decision tree recursively and implements 3 stopping conditions
# Stopping condition 1: All data points in a node are from the same class.
# Stopping condition 2: No more features to split on.
# Additional stopping condition: In addition to the above two stopping conditions covered in lecture,
# in this assignment we will also consider a stopping condition based on the max_depth of the tree.
# By not letting the tree grow too deep, we will save computational effort in the learning process
def decision_tree_create(data, features, target, current_depth = 0, 
                         max_depth = 10, min_node_size=1, 
                         min_error_reduction=0.0):
    
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print ("--------------------------------------------------------------------")
    print ("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))
    
    
    # Stopping condition 1: All nodes are of the same type.
    if intermediate_node_num_mistakes(target_values) == 0:
        print ("Stopping condition 1 reached. All data points have the same target value.")                
        return create_leaf(target_values)
    
    # Stopping condition 2: No more features to split on.
    if remaining_features == []:
        print ("Stopping condition 2 reached. No remaining features.")                
        return create_leaf(target_values)    
    
    # Early stopping condition 1: Reached max depth limit.
    if current_depth >= max_depth:
        print ("Early stopping condition 1 reached. Reached maximum depth.")
        return create_leaf(target_values)
    
    # Early stopping condition 2: Reached the minimum node size.
    # If the number of data points is less than or equal to the minimum size, return a leaf.
    if reached_minimum_node_size(data, min_node_size):
        print ("Early stopping condition 2 reached. Reached minimum node size.")
        return create_leaf(target_values)

    # Find the best splitting feature
    splitting_feature = best_splitting_feature(data, features, target)
    
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    
    # Early stopping condition 3: Minimum error reduction
    # Calculate the error before splitting (number of misclassified examples 
    # divided by the total number of examples)
    error_before_split = intermediate_node_num_mistakes(target_values) / float(len(data))
    
    # Calculate the error after splitting (number of misclassified examples 
    # in both groups divided by the total number of examples)
    left_mistakes = intermediate_node_num_mistakes(left_split[target])
    right_mistakes = intermediate_node_num_mistakes(right_split[target])
    error_after_split = (left_mistakes + right_mistakes) / float(len(data))
    
    # If the error reduction is LESS THAN OR EQUAL TO min_error_reduction, return a leaf.
    if error_reduction(error_before_split, error_after_split) <= min_error_reduction:
        print ("Early stopping condition 3 reached. Minimum error reduction.")
        return create_leaf(target_values)
    
    remaining_features.remove(splitting_feature)
    print ("Split on feature %s. (%s, %s)" % (\
                      splitting_feature, len(left_split), len(right_split)))
    
    
    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, 
                                     current_depth + 1, max_depth, min_node_size, min_error_reduction)        
    
    ## YOUR CODE HERE
    right_tree = decision_tree_create(right_split, remaining_features, target, 
                                     current_depth + 1, max_depth, min_node_size, min_error_reduction)
    
    
    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}

    
# Making predictions with a decision tree
def classify(tree, x, annotate = False):
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate:
             print ("At leaf, predicting %s" % tree['prediction'])
        return tree['prediction']
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate:
             print ("Split on %s = %s" % (tree['splitting_feature'], split_feature_value))
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            ### YOUR CODE HERE
            return classify(tree['right'], x, annotate)

# This function should return a prediction (class label) for each row in data using the decision tree
def evaluate_classification_error(tree, data):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x), axis=1)
    
    # Once you've made the predictions, calculate the classification error and return it
    ## YOUR CODE HERE
    
    return (data['safe_loans'] != np.array(prediction)).values.sum() *1. / len(data)

# return the classification error, other version of old (previous function)
def new_evaluate_classification_error(tree, data):
    # Apply the classify(tree, x) to each row in your data
    data['prediction'] = [classify(tree,a) for a in data.to_dict(orient = 'records')]
           
    # Once you've made the predictions, calculate the classification error and return it
    classification_error = round(float(sum(data['prediction'] != data['safe_loans']))/len(data),4)
    return classification_error

# count number of leafs (measure of complexity)
def count_leaves(tree):
    if tree['is_leaf']:
        return 1
    return count_leaves(tree['left']) + count_leaves(tree['right'])

# print out a single decision stump
def print_stump(tree, name = 'root'):
    split_name = tree['splitting_feature'] # split_name is something like 'term. 36 months'
    if split_name is None:
        print ("(leaf, label: %s)" % tree['prediction'])
        return None
    split_feature, split_value = split_name.split('_',1)
    print ('                       %s' % name)
    print ('         |---------------|----------------|')
    print ('         |                                |')
    print ('         |                                |')
    print ('         |                                |')
    print ('  [{0} == 0]               [{0} == 1]    '.format(split_name))
    print ('         |                                |')
    print ('         |                                |')
    print ('         |                                |')
    print ('    (%s)                         (%s)' \
        % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree')))
    

if __name__ == "__main__":
    print(" all useful decision tree utility functions")
