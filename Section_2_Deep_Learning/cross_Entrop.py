import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    # return np.add(np.multiply(Y, np.log(P)), np.multiply(1-1*Y, np.log(1-1*P)))
    Y = np.array(Y)
    P = np.array(P)
    return -1*np.sum(np.add(np.multiply(Y, np.log(P)), np.multiply(1-1*Y, np.log(1-1*P))))