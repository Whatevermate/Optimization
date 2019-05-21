# -*- coding: utf-8 -*-
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import time

###############################################################
def kernel(x1, x2):
    global sigma
    norm = np.linalg.norm(x1-x2)
    kernel = np.exp(-norm**2/(2*(sigma**2)))   
    return kernel


################################### 
#Initialize e for the first time
def initialize_epsilon(n_instances, x_train, y_train):
    global e, b, a, kernel_matrix
    for i in range(n_instances):
        u = 0
        for j in range(n_instances):
            u += y_train[j]*a[j]*kernel_matrix[i,j]
        u = u-b
        e[i] = u - y_train[i]  
    return 0


##########################################                 
# given the new w, b update E vector
def update_epsilon(x_train, y_train, n_instances, i1, i2, alph1, alph2, b_old):
    global e, b, a, kernel_matrix
    k1 = y_train[i1]*(a[i1]-alph1)
    k2 = y_train[i2]*(a[i2]-alph2)
    b_dif = b - b_old
    for i in range(n_instances):
        e[i] += k1*kernel_matrix[i,i1] + k2*kernel_matrix[i,i2] - b_dif  
    return 0
             
    
###################################      
def heuristic_2(i2):
    global e, nonbound
    difference = -1
    for i1 in nonbound:
        result = np.abs(e[i1] - e[i2]) 
        if result > difference:
            difference = result
            i = i1
    return i
    

##################################  
def update_nonbound(i1, i2, C):
    global a, nonbound
    for i in [i1, i2]:
        if i in nonbound:
            if a[i] == 0 or a[i] == C:
                nonbound.remove(i)
        else:
            if a[i] > 0 and a[i] < C:
                nonbound.append(i)
    return 0


###################################  
def takeStep(i1, i2, x_train, y_train, C, eps = 10**(-4)):
    global e, a, nonbound, b, kernel_matrix
    n_instances = np.shape(y_train)[0]
    if i1 == i2:
        return 0
    alph1 = a[i1]
    alph2 = a[i2]
    y1 = y_train[i1]
    y2 = y_train[i2]
    e1 = e[i1]
    e2 = e[i2]
    s = y1*y2
    if y1 != y2:
        l = max(0, alph2-alph1)
        h = min(C, C+alph2-alph1)
    else:
        l = max(0, alph2+alph1-C)
        h = min(C, alph2+alph1)
    if (l == h):
        return 0
    k11 = kernel_matrix[i1,i1]
    k12 = kernel_matrix[i1,i2]
    k22 = kernel_matrix[i2,i2]

    eta = k11 + k22 - 2*k12
    if eta > 0:
        a2 = alph2 + y2*(e1-e2)/eta

        if a2 < l:
            a2 = l
        elif a2 > h:
            a2 = h
    
    else:
        f1 = y1*(e1 + b) - alph1*k11 - s*alph2*k12
        f2 = y2*(e2 + b) - s*alph1*k12 - alph2*k22
        l1 = alph1 + s*(alph2 - l)
        h1 = alph1 + s*(alph2 - h)
        Lobj = l1*f1 + l*f2 + (1/2)*(l1**2)*k11 + (1/2)*(l**2)*k22 + s*l*l1*k12
        Hobj = h1*f1 + h*f2 + (1/2)*(h1**2)*k11 + (1/2)*(h**2)*k22 + s*h*h1*k12
        if Lobj < Hobj - eps:
            a2 = l
        elif Lobj > Hobj+eps:
            a2 = h
        else:
            a2 = alph2
            
    if abs(a2-alph2) < eps*(a2+alph2+eps):
        return 0
    a1 = alph1+s*(alph2-a2)
          
    b1 = e1 + y1*(a1 - alph1)*k11 + y2*(a2-alph2)*k12 + b
    b2 = e2 + y1*(a1 - alph1)*k12 + y2*(a2-alph2)*k22 + b
    # keep track of the previous b
    b_old = b
    
    if 0 < a1 and a1 < C:
        b = b1
    elif 0 < a2 and a2 < C:
        b= b2
    # Average thresholds if both are bound
    else:
        b = (b1+b2)/2
        
    # Update a1, a2 in the alpha array
    a[i1] = a1
    a[i2] = a2
    
    # Update error cache using new Lagrange multipliers
    update_epsilon(x_train, y_train, n_instances, i1, i2, alph1, alph2, b_old)

    # Update nonbound
    update_nonbound(i1, i2, C)
    return 1


###################################  
def examineExample(i2, x_train, y_train, tol, C):
    global e, a, nonbound
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
def smo(x_train, y_train, C, S, tol = 0.001):
    global e, a, nonbound, sigma, b, kernel_matrix
    sigma = S
    b = 0
    # Initializes b, w, e, a and nonbound
    n_instances = np.shape(y_train)[0]
    e = np.zeros(n_instances)
    a = np.zeros(n_instances)
    kernel_matrix = np.zeros((n_instances,n_instances))
        
    for i in range(n_instances):
        for j in range(n_instances):
            kernel_matrix[i,j] = kernel(x_train[i,:],x_train[j,:])
            
    initialize_epsilon(n_instances, x_train, y_train)
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
            indices = list(set(range(0,np.shape(y_train)[0])))
            random.shuffle(indices)
            for i in indices:
                numChanged += examineExample(i, x_train, y_train, tol, C)   
        else:
            # loop over all nonbound examples
            for i in range(len(nonbound)):
                numChanged += examineExample(i, x_train, y_train, tol, C)

        # if all instances have just been checked, check all nonbound instances
        if examineAll == 1:
            examineAll = 0
        # if all nonbound instances satisfy KKT, check all instances again
        elif numChanged == 0:
            examineAll = 1
    return a, b
#####################################################################

def classify(x_test, y_test, x_train, y_train, a, b, C):
    a_used = a[:]
    for i in range(np.shape(a_used)[0]):
        if a_used[i] <1e-10:
            a_used[i] = 0
    index = np.where(a_used>0)
    y_pred = []
    for i in range(np.shape(y_test)[0]):
        x = x_test[i,:]
        u = 0
        for j in index[0]:
            u += y_train[j]*a[j]*kernel(x, x_train[j,:])
        u = u-b
        if u > 0:
            y_pred.append(1)
        else:
            y_pred.append(-1)
    return y_pred


#####################################################################

def tune_c(x_train, x_test, y_train, y_test, C, indices_train, indices_test, S):
    file = open("nonlinear_test.txt", "w")
    x_train = x_train[indices_train[0]:indices_train[1],:]
    y_train = y_train[indices_train[0]:indices_train[1]]
    x_test = x_test[indices_test[0]:indices_test[1],:]
    y_test = y_test[indices_test[0]:indices_test[1]]
    for c in C:
        for s in S:
            print(c,s)
            a , b = smo(x_train, y_train, C = c, S = s, tol = 0.001)
            y_pred = classify(x_test, y_test, x_train, y_train, a, b, c)
            print(y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1score = f1_score(y_test, y_pred)
            file.write('''C = {}, sigma = {}, accuracy on test set = {}%
                       Precision = {}, Recall = {}
                       F score = {}
                       
                       '''.format(c, s, accuracy*100, precision, recall, f1score))
    file.close()
        
#####################################################################
    


# Parses the file and creates the training set
y_train = []
x_train = []
with open("gisette_scale_train.txt", 'r') as f:
    for line in f:
        tokens = line.split()
        y_train.append(int(tokens[0]))
#        feature_labels = set()
        x_unique = []
        for i in range(1,len(tokens)):
            token_split = tokens[i].split(":")
#            label = int(token_split[0])
#            if label not in feature_labels:
#                feature_labels.add(label)
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
        # Creates x training set
        for i in range(1,len(tokens)):
            token_split = tokens[i].split(":")
            label = int(token_split[0])
#            if label not in test_labels:
#                test_labels.add(label)
            if label not in redundant:
                x = token_split[1]
                x_unique.append(float(x))
        x_test.append(x_unique)

    
# Removes observation that miss some feature value
for index in [950, 4263, 4310]:
    del x_train[index]
    del y_train[index]

# Casts training and test sets into numpy arrays
y_train = np.array(y_train)
y_test = np.array(y_test)
x_train = np.array(x_train)
x_test = np.array(x_test)

##############################################################################

t = time.process_time()
tune_c(x_train, x_test, y_train, y_test, C = [1],
       indices_train = (0,100), indices_test = (-100,-1), S = [10] )
elapsed_time = time.process_time() - t

print (elapsed_time)
