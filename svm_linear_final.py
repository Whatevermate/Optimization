# -*- coding: utf-8 -*-
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

###############################################################
# Defines the kernel function. Although the linear kernel is simple 
# we need to preserve the code structure for the nonlinear kernel
def kernel(x1, x2):
    return np.dot(x1,x2)


###################################                  
# given the new w, b update E vector
def update_epsilon(n_instances, x_train, y_train):
    global b, w, e
    for i in range(n_instances):
        x = x_train[i,:]
        e[i] = np.dot(w, x) - b - y_train[i]
    return 0
             
    
###################################      
# second choice heuristic for choosing the second point of the pair
def heuristic_2(i2):
    global e, nonbound
    # makes sure that the algorithm always chooses at least one point even if all E's are 0
    difference = -1
    for i1 in nonbound:
        result = np.abs(e[i1] - e[i2]) 
        if result > difference:
            difference = result
            i = i1
    return i
    

##################################
# updates nonbound points based on their a values
def update_nonbound(i1, i2, C):
    global a, nonbound
    for i in [i1, i2]:
        # if point was nonbound and its updated a makes it bound, remove it from nonbound
        if i in nonbound:
            if a[i] == 0 or a[i] == C:
                nonbound.remove(i)
        else:
        # if point was bound and its updated a makes it nonbound, add it to nonbound
            if a[i] > 0 and a[i] < C:
                nonbound.append(i)
    return 0


###################################  
def takeStep(i1, i2, x_train, y_train, C, eps = 10**(-4)):
    global b, w, e, a, nonbound
    n_instances =  np.shape(y_train)[0]
    # don't take two identical points
    if i1 == i2:
        return 0
    
    # Lagrange multipliers
    alph1 = a[i1]
    alph2 = a[i2]
    
    # Classes
    y1 = y_train[i1]
    y2 = y_train[i2]
    
    # Errors
    e1 = e[i1]
    e2 = e[i2]
    s = y1*y2
    
    # Calculate L and H of the boxes that restrict the Lagrange multipliers
    if y1 != y2:
        l = max(0, alph2-alph1)
        h = min(C, C+alph2-alph1)
    else:
        l = max(0, alph2+alph1-C)
        h = min(C, alph2+alph1)
    if (l == h):
        return 0
    
    # Kernels
    k11 = kernel(x_train[i1,:], x_train[i1,:])
    k12 = kernel(x_train[i1,:], x_train[i2,:])
    k22 = kernel(x_train[i2,:], x_train[i2,:])

    eta = k11 + k22 - 2*k12
    
    # if eta > 0 no problem
    if eta > 0:
        a2 = alph2 + y2*(e1-e2)/eta

        if a2 < l:
            a2 = l
        elif a2 > h:
            a2 = h

    else:
        # if eta < 0 compute the new objective functions
        f1 = y1*(e1 + b) - alph1*k11 - s*alph2*k12
        f2 = y2*(e2 + b) - s*alph1*k12 - alph2*k22
        l1 = alph1 + s*(alph2 - l)
        h1 = alph1 + s*(alph2 - h)
        # objective function at a2 = L
        Lobj = l1*f1 + l*f2 + (1/2)*(l1**2)*k11 + (1/2)*(l**2)*k22 + s*l*l1*k12
        # objective function at a2 = H
        Hobj = h1*f1 + h*f2 + (1/2)*(h1**2)*k11 + (1/2)*(h**2)*k22 + s*h*h1*k12
        if Lobj < Hobj - eps:
            a2 = l
        elif Lobj > Hobj + eps:
            a2 = h
        else:
            a2 = alph2
            
    if abs(a2 - alph2) < eps*(a2 + alph2 + eps):
        return 0
    a1 = alph1 + s*(alph2 - a2)
    
    # Update threshold
    b1 = e1 + y1*(a1 - alph1)*k11 + y2*(a2-alph2)*k12 + b
    b2 = e2 + y1*(a1 - alph1)*k12 + y2*(a2-alph2)*k22 + b
    if 0 < a1 and a1 < C:
        b = b1
    elif 0 < a2 and a2 < C:
        b= b2
    # Average thresholds if both are bound
    else:
        b = (b1+b2)/2
        
        
    # Update weight vector to reflect change in a1 & a2, if SVM is linear
    x1 = x_train[i1,:]
    x2 = x_train[i2,:]
    w = w + y1*(a1-alph1)*x1 + y2*(a2-alph2)*x2    
    
    # Update error cache using new Lagrange multipliers
    update_epsilon(n_instances, x_train, y_train)
    # Update the Lagrange multipliers in the alpha array
    a[i1] = a1
    a[i2] = a2
     
    # Update nonbound vector
    update_nonbound(i1, i2, C)
    return 1


###################################  
def examineExample(i2, x_train, y_train, tol, C):
    global b, w, e, a, nonbound
    y2 = y_train[i2]
    alph2 = a[i2]
    e2 = e[i2]
    r2 = e2*y2
    if (r2 < -tol and alph2 < C) or (r2 > tol and alph2 > 0):
        # check if there are any nonbound instances left
        if len(nonbound) > 0:
            i1 = heuristic_2(i2)
            result = takeStep(i1, i2, x_train, y_train, C)
            if result:
                return result
    # if the heuristic doesn't work, loop over all nonbound starting at a random point
        shuffle_nonbound = nonbound[:]
        random.shuffle(shuffle_nonbound)
        for item in shuffle_nonbound:
            i1 = item
            result = takeStep(i1, i2, x_train, y_train, C)
            if result:  
                return result
        
    
        complementary = list(set(range(0,np.shape(y_train)[0])) - set(nonbound))
        random.shuffle(complementary)
        for item in complementary:
            i1 = item
            result = takeStep(i1, i2, x_train, y_train, C)
            if result:
                return result 
    return 0
    

###################################       
def smo(x_train, y_train, C, tol = 0.001):
    global b, w, e, a, nonbound
        
    # Initializes b, w, e, a and nonbound
    n_instances = np.shape(y_train)[0]
    n_features = np.shape(x_train)[1]
    b = 0
    w = np.zeros(n_features)
    e = np.zeros(n_instances)
    a = np.zeros(n_instances)

    update_epsilon(n_instances, x_train, y_train)

    
    # number of a pairs that were changed during a specific iteration
    numChanged = 0
    
    # Whether all instances have been examined
    # 1 is "not all were examined"
    examineAll = 1
    
    # find all nonbound instances beforehand and then update only the two
    nonbound = []
    
    while numChanged > 0 or examineAll:
        numChanged = 0
        # If not all instances have been examined check all instances 
        if examineAll:
            # loop oever all instances
            indices = list(set(range(0,np.shape(y_train)[0])))
            random.shuffle(indices)
            for i in indices:
                numChanged += examineExample(i, x_train, y_train, tol, C)
        else:
            # loop over all nonbound instances
            for i in range(len(nonbound)):
                numChanged += examineExample(i, x_train, y_train, tol, C)

        # if all instances have just been checked, check all nonbound instances
        if examineAll == 1:
            examineAll = 0
        # if all nonbound instances satisfy KKT, check all instances again
        elif numChanged == 0:
            examineAll = 1
    return w, b
#####################################################################

# classifies instances based on the resulting classifier and produces y_pred vector
def classify(x_test, y_test, w, b):
    y_pred = []
    for i in range(np.shape(y_test)[0]):
        x = x_test[i,:]
        u = np.dot(w.T, x) - b
        if u > 0:
            y_pred.append(1)
        else:
            y_pred.append(-1)
    return y_pred



#####################################################################
# Tunes parameter c over a training set 
# Takes as parameters the indices of the training and the test set 
# to slice initial sets per request
def tune_c(x_train, x_test, y_train, y_test, C, indices_train, indices_test):
    file = open("results_validate_linear", "w")
    x_train = x_train[indices_train[0]:indices_train[1],:]
    y_train = y_train[indices_train[0]:indices_train[1]]
    x_test = x_test[indices_test[0]:indices_test[1],:]
    y_test = y_test[indices_test[0]:indices_test[1]]
    for c in C:
        print(c)
        # Computes the classifier based on this value of C
        w, b = smo(x_train, y_train, C = c, tol = 0.001)
        # Classifies the given test set
        y_pred = classify(x_test, y_test, w, b)
        # Computes accuracy, precision, recall, f1 scores
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)
        # Outputs them in file
        file.write('''C = {}, accuracy on test set = {}%
                   Precision = {}, Recall = {}
                   F score = {}
                   
                   '''.format(c, accuracy*100, precision, recall, f1score))
        # y_pred = []
    file.close()
        
#####################################################################
    


# Parses the file and creates the training set
# Commented out lines were necessary for finding common features in the training and test set
y_train = []
x_train = []
with open("gisette_scale_train.txt", 'r') as f:
    for line in f:
        # Feature value is on the right of the : character
        tokens = line.split()
        # First row element is the class
        y_train.append(int(tokens[0]))
        # feature_labels = set()
        x_unique = []
        # Creates x training set
        for i in range(1,len(tokens)):
            token_split = tokens[i].split(":")
            # label = int(token_split[0])
            # if label not in feature_labels:
            # feature_labels.add(label)
            x = token_split[1]
            x_unique.append(float(x))
        x_train.append(x_unique)


y_test = []
x_test = []

# Removes features that do not appear in the training set
redundant = [112,197,1038,1041,1180,1268,1737,1905,2087,2692,2910,2953,3158,3746,4389,4873]
with open("gisette_scale_test.txt", 'r') as f:
    for line in f:
        # Feature value is on the right of the : character
        tokens = line.split()
        # First row element is the class
        y_test.append(int(tokens[0]))
        # test_labels = set()
        x_unique = []
        # Creates x test set
        for i in range(1,len(tokens)):
            token_split = tokens[i].split(":")
            label = int(token_split[0])
            # if label not in test_labels:
            # test_labels.add(label)
            if label not in redundant:
                x = token_split[1]
                x_unique.append(float(x))
        x_test.append(x_unique)

    
# Removes observations that miss some feature value
for index in [950, 4263, 4310]:
    del x_train[index]
    del y_train[index]

# Casts training and test sets into numpy arrays
y_train = np.array(y_train)
y_test = np.array(y_test)
x_train = np.array(x_train)
x_test = np.array(x_test)

##############################################################################
# Tunes hyperparameter set on another part of the training data, which acts as a test 
# set here
tune_c(x_train, x_test, y_train, y_test, C = [0.001],
       indices_train = (0, 2000), indices_test = (-1000, -1))


