from __future__ import division
import numpy as np
import pandas as pd

target_users = pd.read_csv("target_users.csv", delimiter='\t')
users = target_users['user_id'].values
training_data = pd.read_csv('training_data.csv',delimiter='\t',index_col=0)
test_data = pd.read_csv('test_data.csv',delimiter=',')

actual = [0] * 10000
for u in range(10000):
    actual[u] = map(int,test_data[test_data['user_id']==users[u]]['recommended_items'].values[0].split(' '))

def apk(actual, predicted, k):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def evaluate_submission(file):
    submission = pd.read_csv(file,delimiter=',')
    predicted = [0] * 10000
    for u in range(10000):
        predicted[u] = map(int, submission[submission['user_id'] == users[u]]['recommended_items'].values[0].split(' '))

    print("Submission: " + file + " - MAP@5: " + str(mapk(actual,predicted,5)))
    return