# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 01:29:55 2018

@author: chinmaya
"""
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


# decision tree recursively and implements 3 stopping conditions
# Stopping condition 1: All data points in a node are from the same class.
# Stopping condition 2: No more features to split on.
# Additional stopping condition: In addition to the above two stopping conditions covered in lecture,
# in this assignment we will also consider a stopping condition based on the max_depth of the tree.
# By not letting the tree grow too deep, we will save computational effort in the learning process
def decision_tree_create(data, features, target, current_depth = 0, max_depth = 10):
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print ("--------------------------------------------------------------------")
    print ("Subtree, depth = %s (%s data points)." % (current_depth, len(target_values)))
    

    # Stopping condition 1
    # (Check if there are mistakes at current node.
    # Recall you wrote a function intermediate_node_num_mistakes to compute this.)
    if intermediate_node_num_mistakes(target_values) == 0:
        print ("Stopping condition 1 reached.")     
        # If not mistakes at current node, make current node a leaf node
        return create_leaf(target_values)
    
    # Stopping condition 2 (check if there are remaining features to consider splitting on)
    if remaining_features == []:
        print ("Stopping condition 2 reached.")    
        # If there are no remaining features to consider, make current node a leaf node
        return create_leaf(target_values)    
    
    # Additional stopping condition (limit tree depth)
    if current_depth >= max_depth:
        print ("Reached maximum depth. Stopping for now.")
        # If the max tree depth has been reached, make current node a leaf node
        return create_leaf(target_values)

    # Find the best splitting feature (recall the function best_splitting_feature implemented above)
    ## YOUR CODE HERE
    splitting_feature = best_splitting_feature(data, remaining_features, target)
    
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    remaining_features.remove(splitting_feature)
    print ("Split on feature %s. (%s, %s)" % (\
           splitting_feature, len(left_split), len(right_split)))
    
    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print ("Creating leaf node.")
        return create_leaf(left_split[target])
    if len(right_split) == len(data):
        print ("Creating leaf node.")
        ## YOUR CODE HERE
        return create_leaf(right_split[target])
        
    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth)        
    ## YOUR CODE HERE
    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth)

    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree,
            'right'            : right_tree}
    

