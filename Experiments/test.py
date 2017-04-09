from __future__ import division
from project_utils import output
from project_utils import interagiti
import numpy as np
import pandas as pd
import graphlab as gl
import random
import project_utils as utils
import contentweight
import contentweightpopularity
import time
from scipy import sparse
import numpy as np
#interactions are not all items that can be recommended
user_profiles = pd.read_csv("user_profile.csv", delimiter='\t')
target_users = pd.read_csv("target_users.csv", delimiter='\t')

# items = gl.SFrame.read_csv('item_profile.csv', delimiter='\t')
# item_profiles = pd.read_csv("item_profile.csv", delimiter='\t')
# ids = item_profiles['id'].values
# index_to_ids = dict(zip(range(ids.size),ids))
#ids_to_index = {v: k for k, v in index_to_ids.items()}
#items_inattivi = item_profiles[item_profiles['active_during_test'] == 0]['id'].values

users = target_users['user_id'].values

uds = user_profiles['user_id'].values
uds_to_index = dict(zip(uds, range(uds.size)))
index_to_uds = {v: k for k, v in uds_to_index.items()}

# interactions = pd.read_csv('interactions.csv', delimiter="\t")
# interactions = interactions.sort_values(['user_id','created_at'],ascending=[True,False])
#
# test = [0] * 10000
#
# for u in range(10000):
#     recent = interactions[interactions['user_id'] == users[u]]['item_id'].head(5)
#
#     a = np.array(recent.values)
#     _, idx = np.unique(a, return_index=True)
#     a = a[np.sort(idx)]
#
#     while a.size < 5:
#         a = np.append(a,[0])
#
#     test[u] = a
#     print(str(u))
#
# result = np.column_stack((users,test))
# output(result)


training_data = pd.read_csv('training_data.csv',delimiter='\t',index_col=0)
test_data = pd.read_csv('test_data.csv',delimiter=',')
#map(int,test_data[test_data['user_id']==285]['recommended_items'].values[0].split(' '))

#pd.read_csv('test_data.csv',delimiter=',').iloc[0]['recommended_items'].split(' ')
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