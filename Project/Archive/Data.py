import numpy as np
import Radond_sugg
import pandas as pd

k = 8 #length of the tail
#we pick at random the k most clicked jobs

users = np.genfromtxt("target_users.csv", delimiter="\t", dtype=np.dtype(int), skip_header=1)
items = pd.read_csv("item_profile.csv", delimiter="\t")
interactions = pd.read_csv("interactions.csv", usecols=['item_id','interaction_type'], delimiter="\t")
interactions = interactions[interactions['interaction_type'] == 1]

def output(matrix):
    "Creates the file to be submitted given a kind of matrix"
    # output to file
    submission = open('submission.csv', 'w')
    header = 'user_id,recommended_items'
    submission.write(header + '\n')

    for row in matrix:
        for i in range(np.size(row)):
            if i == 0:
                line = str(row[0]) + ','
            elif i == 5:
                line = line + str(row[i])
            else:
                line = line + str(row[i]) + ' '
        submission.write(line + '\n')

    submission.close()
    return
