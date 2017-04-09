import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

interactions_raw = pd.read_csv("interactions.csv", usecols=['user_id', 'item_id','interaction_type'], delimiter="\t")

item_profiles = pd.read_csv("item_profile.csv", delimiter="\t")
item_profiles = item_profiles.fillna(0)

item_profiles['country'] = item_profiles['country'].replace(0,'zero')

item_profiles['country'] = le.fit_transform(item_profiles['country'])

item_profiles.to_csv('./items_processed.csv', sep='\t')