import pandas as pd
import itertools
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn import preprocessing
import sklearn.metrics.pairwise as metrics

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

# Returns the list of tags that appear at least min_k times
def more_than_k_occurrences(t, min_k):

    keys = np.unique(t)
    counts = [0] * len(keys)
    tag_dict = dict(zip(keys, counts))
    for tag in range(len(t)):
        tag_dict[t[tag]] += 1
    tags_more_than_one_occurrence = []
    for tag in tag_dict.keys():
        if tag_dict.get(tag) > min_k:
            tags_more_than_one_occurrence.append(tag)

    return np.unique(tags_more_than_one_occurrence)


# Compute the one hot encoding representation for tags on items
def one_hot_encoding_tags(items):

    print("Computing One Hot Encoding for tags..")
    item_tags = items['tags'].reset_index()
    item_tags['tags'] = item_tags['tags'].apply(lambda x: str(x).split(','))
    tags = item_tags['tags'].tolist()
    flattened_tags = list(itertools.chain.from_iterable(tags))
    min_k = 2
    flattened_tags = np.asarray(more_than_k_occurrences(flattened_tags, min_k))
    flattened_tags = np.delete(flattened_tags, 0)
    tag_indexes = [i for i in range(0, len(flattened_tags))]
    # Dictionary containing for each tag its id
    # Key: tag
    # Value: index
    tags_dict = dict(zip(flattened_tags, tag_indexes))

    items_id_tag = items.drop_duplicates('id')[['id', 'tags']]
    items_id_tag['tags'] = items['tags'].apply(lambda x: x.split(','))
    # Dictionary containing for each item its tags
    # Key: item_id
    # Value: list of tags
    item_tags_dict = dict(zip(items['id'].tolist(), items['tags'].tolist()))
    ids = items['item_index'].tolist()
    items_dict = dict(zip(items['id'], ids))

    row_indexes = []
    col_indexes = []
    tag_values = []

    for item in item_tags_dict.keys():
        tags = str(item_tags_dict.get(item)).split(',')
        item_index = items_dict.get(item)
        cols = []
        for tag in tags:
            tag_index = tags_dict.get(tag)
            if tag_index != None:
                row_indexes.append(item_index)
                cols.append(tags_dict.get(tag))
        col_indexes.extend(cols)
        vals = [1] * len(cols)
        tag_values.extend(vals)

    tags_ohe = sp.coo_matrix((tag_values, (row_indexes, col_indexes)), shape=(len(items), len(tag_indexes)))
    zero_index = tags_dict.get('0')
    tags_ohe = tags_ohe.tocsc()
    if zero_index != None:
        tags_ohe = tags_ohe[:,zero_index] = 0

    return tags_ohe


def one_hot_encoding_titles(items):

    print("Computing One Hot Encoding for titles..")
    item_titles = items['title'].reset_index()
    item_titles['title'] = item_titles['title'].apply(lambda x: str(x).split(','))
    titles = item_titles['title'].tolist()
    flattened_titles = list(itertools.chain.from_iterable(titles))
    min_k = 1
    flattened_titles = np.asarray(more_than_k_occurrences(flattened_titles, min_k))
    flattened_titles = np.delete(flattened_titles, 0)
    title_indexes = [i for i in range(0, len(flattened_titles))]
    # Dictionary containing for each title its id
    # Key: title
    # Value: index
    titles_dict = dict(zip(flattened_titles, title_indexes))

    items_id_title = items.drop_duplicates('id')[['id', 'title']]
    items_id_title['title'] = items['title'].apply(lambda x: x.split(','))
    # Dictionary containing for each item its title
    # Key: item_id
    # Value: list of tags
    item_titles_dict = dict(zip(items['id'].tolist(), items['title'].tolist()))

    ids = items['item_index'].tolist()
    items_dict = dict(zip(items['id'], ids))

    row_indexes = []
    col_indexes = []
    tag_values = []

    for item in item_titles_dict.keys():
        titles = str(item_titles_dict.get(item)).split(',')
        item_index = items_dict.get(item)
        cols = []
        for title in titles:
            title_index = titles_dict.get(title)
            if title_index != None:
                row_indexes.append(item_index)
                cols.append(titles_dict.get(title))
        col_indexes.extend(cols)
        vals = [1] * len(cols)
        tag_values.extend(vals)

    tags_ohe = sp.coo_matrix((tag_values, (row_indexes, col_indexes)), shape=(len(items), len(title_indexes)))
    zero_index = titles_dict.get('0')
    tags_ohe = tags_ohe.tocsc()
    if zero_index != None:
        tags_ohe = tags_ohe[:, zero_index] = 0
    return tags_ohe

VALIDATION = False
# Load datasets
item_profile = pd.read_table("item_profile.csv")
interactions = pd.read_table("training_data.csv")

target_users_tot = pd.read_table("target_users.csv")

target_users = target_users_tot
target_users['recommended_tags_items'] = 0
target_users['points_tag'] = 0
target_users['recommended_titles_items'] = 0
target_users['points_tit'] = 0
target_users['recommended_TT_items'] = 0
target_users['points_tt'] = 0

# Get the list of unique item_ids that were in at least one interaction
item_profile = item_profile.drop_duplicates('id').reset_index()

# Assign to each item an incremental index
item_profile.loc[:, 'item_index'] = 0
item_profile['item_index'] = item_profile.index
item_profile = item_profile[['item_index', 'id', 'active_during_test','country','region','tags','title']].fillna('0')

#Number of elements in tags and titles for each item
item_profile['cont_tags'] = item_profile['tags'].apply(lambda x: len(str(x).split(',')) if x != '0' else 0 )
item_profile['cont_title'] = item_profile['title'].apply(lambda x: len(str(x).split(',')) if x != '0' else 0)

one_hot_tags = one_hot_encoding_tags(item_profile[['id', 'item_index', 'tags']])
one_hot_titles = one_hot_encoding_titles(item_profile[['id', 'item_index', 'title']])

one_hot_tags = one_hot_tags.tocsc()
one_hot_titles = one_hot_titles.tocsc()

interactions = pd.merge(interactions, item_profile, left_on='item_id', right_on='id')[['user_id', 'item_id',
                                                                                   'item_index', 'active_during_test',
                                                                                   'created_at','country']]

interactions = interactions.drop_duplicates(['user_id', 'item_id'])

#interactions = interactions.sort_values(by=['user_id', 'item_id'])

# Get the list of unique user_ids that had at least one interactions
user_ids = target_users.drop_duplicates('user_id').reset_index()
# Assign to each user an incremental index
user_ids.loc[:, 'user_index'] = 0
user_ids['user_index'] = user_ids.index
user_ids = user_ids[['user_id', 'user_index']]

# Append that index to the interactions dataframe
interactions = pd.merge(interactions, user_ids, on='user_id')[['user_id', 'user_index',
                                                               'item_id', 'item_index',
                                                               'created_at', 'active_during_test','country']]

# Find item_index of active items
active_indices = item_profile[item_profile['active_during_test'] != 0].drop_duplicates('item_index')

interactions = interactions.drop('country',axis=1)
# Find complete list of item indices
item_indices_list = np.asarray(item_profile['item_index'].tolist())

item_profile_active = item_profile.iloc[active_indices['item_index'].tolist()]

#item_weight_active = active_indices['points'].tolist()

one_hot_tags_t = one_hot_tags[active_indices['item_index'].tolist(), :].T.copy()

one_hot_titles_t = one_hot_titles[active_indices['item_index'].tolist(), :].T.copy()

interactions = interactions.drop('active_during_test',axis=1)
interactions = pd.merge(interactions,item_profile,how='left',on='item_index').fillna(0)

#active_indices = active_indices.sort('item_index')

num_elem = 50

ai = active_indices.reset_index().drop('index',axis=1)
matrix = np.zeros(shape=[len(target_users),len(active_indices['item_index'])])

for u in range(len(target_users)):
    top_k_ids = []
    print("Making recommendations: ", u, "users done ", len(target_users) - u, " to go")
    user = target_users['user_id'].iloc[u]
    rated_items = interactions[interactions['user_id'] == user]

    if len(rated_items) != 0:
        rated_items_list = rated_items['item_index'].tolist()

        #Computes Tags sim matrix
        current_one_hot_tags = one_hot_tags[rated_items_list, :].copy()
        current_tags_intersection = current_one_hot_tags.dot(one_hot_tags_t)
        current_tags_union = current_tags_intersection.copy().power(0)
        current_tags_union_t = current_tags_union.T.copy()
        rated_cont = rated_items['cont_tags'].tolist()
        current_tags_union = current_tags_union.dot(sp.diags(item_profile_active['cont_tags'].tolist()))
        current_tags_union_t = current_tags_union_t.dot(sp.diags(rated_cont))

        current_tags_union = current_tags_union + current_tags_union_t.T
        current_tags_union = current_tags_union - current_tags_intersection
        current_tags_sim_matrix = current_tags_intersection.multiply(current_tags_union.power(-1))

        #Computes Tags sim vector
        current_tags_sim_vector = current_tags_sim_matrix.sum(axis=0)
        current_tags_sim_vector = np.array(current_tags_sim_vector)[0]

        #Computes Titles sim matrix
        current_one_hot_titles = one_hot_titles[rated_items_list, :].copy()
        current_titles_intersection = current_one_hot_titles.dot(one_hot_titles_t)

        current_titles_union = current_titles_intersection.copy().power(0)
        current_titles_union_t = current_titles_union.T.copy()

        rated_cont = rated_items['cont_title'].tolist()

        current_titles_union = current_titles_union.dot(sp.diags(item_profile_active['cont_title'].tolist()))
        current_titles_union_t = current_titles_union_t.dot(sp.diags(rated_cont))

        current_titles_union = current_titles_union + current_titles_union_t.T
        current_titles_union = current_titles_union - current_titles_intersection

        current_titles_sim_matrix = current_titles_intersection.multiply(current_titles_union.power(-1))
        #computes titles sim vector
        current_titles_sim_vector = current_titles_sim_matrix.sum(axis=0)
        current_titles_sim_vector = np.array(current_titles_sim_vector)[0]  ###########################################

        #rated items by the current user
        rated = interactions[interactions['user_id'] == user]
        active_rated = list(rated[:][rated['active_during_test'] != 0]['item_index'])
        active_rated_index = []

        ar = ai[ai['item_index'].isin(active_rated)].index
        current_titles_sim_vector[ar] = 0
        current_tags_sim_vector[ar] = 0

        top_k_TT_sum = ( 0.65 * current_tags_sim_vector + 0.35 * current_titles_sim_vector) #* item_weight_active
        ma = top_k_TT_sum.max()
        top_k_TT_sum = top_k_TT_sum / ma
        matrix[u,:] = top_k_TT_sum

matrix = sp.csc_matrix(matrix)
save_sparse_csr('CBF_ITEM_ITEM', matrix.tocsr())