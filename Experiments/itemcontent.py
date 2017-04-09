from project_utils import *
from __future__ import division
import numpy as np
import pandas as pd
import graphlab as gl
import random
import project_utils as utils

user_profiles = pd.read_csv("user_profile.csv", delimiter='\t')
target_users = pd.read_csv("target_users.csv", delimiter='\t')
interactions = pd.read_csv('interactions.csv', delimiter="\t")
items = pd.read_csv("item_preprocessed.csv", delimiter=',').drop('Unnamed: 0',1)
items = gl.SFrame(items).rename({'id':'item_id'})
observations = gl.SFrame(interactions)

users = gl.SFrame(target_users)
training_data, validation_data = gl.recommender.util.random_split_by_user(observations, 'user_id', 'item_id')
userss = users['user_id'].to_numpy()
most_popular = [2778525, 1244196, 1386412, 657183, 2791339]

def fill(input):
    suggestions = [0] * 10000
    for i in range(10000):
        suggestions[i] = input[input['user_id'] == userss[i]]['item_id'].values
    return suggestions

def fill2(rec):
    suggestions = [0] * 10000

    for i in range(10000):
        sss = rec[rec['user_id'] == userss[i]].sort_values('score',ascending=False).head(5)['item_id'].values
        j = 0

        while sss.size < 5:
            sss = np.append(sss,[most_popular[j]])
            j = j+1

        suggestions[i] = sss
    return suggestions

model = gl.recommender.item_content_recommender.create(items,'item_id',observations,'user_id',target=None)
