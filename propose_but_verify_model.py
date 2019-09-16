'''
Propose but verify model, original Trueswell version.

Trueswell, J. C., Medina, T. N., Hafri, A., & Gleitman, L. R. (2013). Propose but verify: 
Fast mapping meets cross-situational word learning. Cognitive psychology, 66(1), 126-156.

alpha             initial recovery probability for new pairing
alpha_confirmed   probability after one confirmation (probability does not increase more after further confirmation) 
'''


from random import shuffle
import numpy as np
from statistics import median
from math import log


def learn(pairing, words, objects, parameters):
    alpha = parameters[0]
    alpha_confirmed = parameters[1]
    
    shuffle(objects)
    
    for i in range(len(words)):
        # If sum of probabilities is not 0, than a pairing exists for the word
        word_prob = np.sum(pairing, axis=1)[words[i]-1]
        # For known words:
        if word_prob != 0:
            # retrieve paired object, i.e. only column that is not 0
            paired_object = np.argmax(pairing[words[i]-1,:])+1
            
            # remember previous guess with stored probability
            prob = pairing[words[i]-1, paired_object-1]
            recalled = np.random.choice([0, 1], p=[1-prob, prob])
            
            # If previous guess is recalled and object is present, i.e. pairing is confirmed: 
            # increase probability
            if paired_object in objects and recalled == 1: 
                pairing[words[i]-1, paired_object-1] = alpha_confirmed
            # Otherwise: forget pairing
            else:
                pairing[words[i]-1, paired_object-1] = 0
            
    
    for o in range(pairing.shape[1]):
        # if object is present
        if o+1 in objects:
            paired_word =  np.argmax(pairing[:,o])+1
            # but paired word is not present
            if paired_word not in words:
                #forget pairing
                pairing[paired_word-1, o] = 0

    objects_unpaired = []
    for o in objects:
        if np.sum(pairing, axis=0)[o-1] == 0:
            objects_unpaired.append(o)
                
    for i in range(len(words)):
        word_prob = np.sum(pairing, axis=1)[words[i]-1]
        
        # For a unpaired word, make a guess
        if word_prob == 0:
            unpaired_object = objects_unpaired.pop()
            pairing[words[i]-1, unpaired_object-1] = alpha
            
    return pairing



'''
Calculate the log likelikhood of the model prediciting a subjects responses.

parameters = [alpha, alpha_confirmed]
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
            
        M_llh = M.copy()
        for j in range(items):
            # probability of correct pairing is equal to the stored probability for correct hypotheses
            # and defaults to a smoothed value of 0.0001 otherwise.
            correct_prob = M_llh[j, j]
            if correct_prob == 0:
                correct_prob += 0.0001
            elif correct_prob == 1:
                correct_prob -= 0.0001
            
            if subject['testing_accuracy'][i][j] == 1:
                likelihood_elements.append(correct_prob)
            else:
                likelihood_elements.append(1 - correct_prob)
    
    # return negative of log likelihood because scipy.optimize
    # minimizes instead of maximizing
    return -1 * sum(np.log(np.array(likelihood_elements)))

'''
Call subject_log_likelihood n times and return median value

n defaults to 10
'''

def subject_median_log_likelihood(parameters, subject, n=10):
    
    subject_log_likelihood_output = []
    for i in range(n):
        log_likelihood = subject_log_likelihood(parameters, subject)
        subject_log_likelihood_output.append(log_likelihood)
        
    return median(subject_log_likelihood_output)


