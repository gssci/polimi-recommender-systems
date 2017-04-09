import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

user_profiles = pd.read_csv("user_profile.csv", delimiter='\t')
user_profiles = user_profiles.fillna(0)
user_profiles['country'] = user_profiles['country'].replace(0,'zero')
user_profiles['country'] = le.fit_transform(user_profiles['country'])

# v = user_profiles['jobroles'].str.split(',').apply(pd.Series).fillna(0)
# for i in range(44):
#         user_profiles['jobroles' + str(i)] = v[i]
# user_profiles = user_profiles.drop('jobroles',1)
#
# u = user_profiles['edu_fieldofstudies'].str.split(',').apply(pd.Series).fillna(0)
# for i in range(5):
#         user_profiles['edu_fieldofstudies' + str(i)] = u[i]
# user_profiles = user_profiles.drop('edu_fieldofstudies',1)

user_profiles.to_csv('./user_profile_ppd.csv', sep='\t')

# user_profiles = user_profiles.select_dtypes(include=[np.number], exclude=None) #preprocessing per duri
#
# tmp = user_profiles[user_profiles.columns.difference(['user_id'])]
# nbrs = NearestNeighbors(n_neighbors=K, algorithm='auto').fit(tmp)
#
# def faci(x): #nb prende in input una riga del database
#     x = x[x.columns.difference(['user_id'])]
#     distances, indices = nbrs.kneighbors(x)
#     return user_profiles.ix[indices[0]]
#
# def byID(id):
#     return user_profiles[user_profiles['user_id'] == id]




