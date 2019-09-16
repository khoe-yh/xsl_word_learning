''' 
Associative Uncertainty- (Entropy) & Familiarity-Biased Model

Kachergis, G., Yu, C., & Shiffrin, R. M. (2012, November). Cross-situational 
word learning is better modeled by associations than hypotheses. In Development 
and Learning and Epigenetic Robotics (ICDL), 2012 IEEE International Conference 
on (pp. 1-6). IEEE.

M           a word by object association matrix
words       the list of word id's in a training trial
objects     the list of objects id's in a training trial
parameters  [alpha, chi, labda]
alpha       a decay parameter
chi         the total amount of associative weight to be distributed
labda       a scaling parameter governing differential weighting of uncertainty and prior knowledge  
k           the initial association strength allocated to a new word-object pairing
U           a matrix containing the associative strength to be added for this trial to each word-object pair in M
'''

import numpy as np
from scipy.stats import entropy
from math import exp, log


def learn(M, words, objects, parameters):
    alpha = parameters[0]
    chi = parameters[1]
    labda = parameters[2]
    k = 0.01
    
    # Create empty update matrix U
    U = np.zeros((M.shape[0],M.shape[1]))
    
    # Add initial association strength k for new words and objects
    for w in words:
        for column in range(M.shape[1]):
            if np.sum(M[:,column]) > 0 and M[w-1,column] == 0:
                M[w-1,column] = k
                
                # If number of words and objects are not equal, should simple mirroring 
                # like this really happen?
                if M[column,w-1] == 0:
                    M[column,w-1] = k
        for o in objects:
            if M[w-1,o-1] == 0:
                M[w-1,o-1] = k     
            if M[o-1,w-1] == 0:
                M[o-1,w-1] = k  
                
    # Calculate strength allocation for each word-object pair
    for w in words:
        
        # Calculate Shannon entropy of the probability distribution
        # of the word
        w_entropy = entropy(list(M[w-1,]), base=2)
        
        for o in objects:
            # Calculate Shannon entropy of the probability distribution
            # of the object
            o_entropy = entropy(list(M[:,o-1]), base=2)
            
            # Exponentiate sum of labda and entropy product
            # and multiply by current association strength
            # This implementation follows Kachergis's own implementation in R
            U[w-1, o-1] = (exp(labda * w_entropy) * exp(labda * o_entropy)) * M[w-1, o-1]
            
    # denominator:
    denom = np.sum(U)
    
    # divide U by denom:
    U = U / denom
    
    # distribute chi using U
    U = U * chi
    
    # decrease existing associations by decay parameter alpha
    M = M * alpha
    
    # add distributed associative weight chi to word object pairings
    M = M + U
    
    return M



'''
Calculate the log likelikhood of the model prediciting a subjects responses.

parameters = [alpha, chi, labda]
'''


def subject_log_likelihood(parameters, subject):
    likelihood_elements = []
    items = np.max(subject['trials'])
    M = np.zeros((items, items))
    blocks = len(subject['trials'])

    for i in range(blocks):
        
        # generate associative Matrix based on trial data
        for trial in subject['trials'][i]:
            M = learn(M, trial, trial, parameters)
        
        for j in range(items):
            # probability of correct pairing is its own probability mass divided by total
            # probability mass for word
            correct_prob = M[j, j] / np.sum(M, axis=1)[j]
            
            if subject['testing_accuracy'][i][j] == 1:
                likelihood_elements.append(correct_prob)
            else:
                likelihood_elements.append(1 - correct_prob)
    
    # return negative of log likelihood because scipy.optimize
    # minimizes instead of maximizing
    return -1 * sum(np.log(np.array(likelihood_elements)))

