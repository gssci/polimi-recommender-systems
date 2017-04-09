import Radond_sugg

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

##IMPORT AND PREPROCESS##
users = pd.read_csv("target_users.csv", delimiter="\t")
target = np.genfromtxt("target_users.csv", delimiter="\t", dtype=np.dtype(int), skip_header=1)

user_profiles = pd.read_csv('user_profile_ppd.csv', delimiter="\t")
user_profiles = user_profiles.drop('Unnamed: 0',1)

item_profiles = pd.read_csv("item_profile.csv", delimiter="\t")
item_profiles = item_profiles.fillna(0)
item_profiles['country'] = item_profiles['country'].replace(0,'zero')
item_profiles['country'] = le.fit_transform(item_profiles['country'])

def encode_feature(dataframe, to_encode):
    hot = preprocessing.OneHotEncoder()

    for feature in to_encode:
        a = hot.fit_transform(dataframe[feature].values.reshape((-1,1))).toarray()
        for i in range(np.shape(a)[1]):
            dataframe[feature + str(i)] = a[:,i]
    return

to_encode = ['discipline_id','industry_id','country','region','career_level','employment']
encode_feature(item_profiles,to_encode)
item_profiles = item_profiles.drop(['title','tags','discipline_id','industry_id','latitude','longitude','created_at','country','region','career_level','employment'],1).astype(int)


interactions = pd.read_csv('interactions.csv', delimiter="\t")
item_raccomandabili = item_profiles[item_profiles['active_during_test'] == 1].drop('active_during_test',1).reset_index().drop('index',1)
interazioni_item = interactions['item_id'].drop_duplicates().reset_index().drop('index',1)
# Grab DataFrame rows where column doesn't have certain values
value_list = interazioni_item['item_id'].values
item_raccomandabili = item_raccomandabili[~item_raccomandabili.id.isin(value_list)]

item_raccomandabili['id'].reset_index().drop('index',1).to_csv('./item_racc.csv', sep='\t')






